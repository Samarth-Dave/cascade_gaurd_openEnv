import type {
  AgentTurn,
  CGEdge,
  CGNode,
  CGObservation,
  EventLine,
  TimelineEntry,
} from '../data/types';

const FALLBACK_SECTOR_SUMMARY = {
  power: 1,
  water: 1,
  hospital: 1,
  telecom: 1,
};

export const DEFAULT_DIAGNOSTICS = {
  critical_failures: [] as string[],
  at_risk_nodes: [] as string[],
  dependency_alerts: [] as string[],
  recommended_recovery_order: [] as string[],
  system_pressure: 0,
};

const now = () => new Date().toTimeString().slice(0, 8);

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const asNumber = (value: unknown, fallback: number) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const asOptionalNumber = (value: unknown): number | undefined => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
};

const asString = (value: unknown, fallback = '') =>
  typeof value === 'string' ? value : fallback;

const asStringArray = (value: unknown) =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];

const asNode = (node: CGNode, pendingRecoveries: string[]): CGNode => {
  const source = node as unknown as Record<string, unknown>;
  const lat = asOptionalNumber(source.lat) ?? asOptionalNumber(source.latitude);
  const lon =
    asOptionalNumber(source.lon) ??
    asOptionalNumber(source.lng) ??
    asOptionalNumber(source.longitude);

  return {
    node_id: asString(source.node_id),
    sector: (asString(source.sector, 'power') as CGNode['sector']),
    health: asNumber(source.health, 1),
    load: asNumber(source.load, 0),
    is_operational: typeof source.is_operational === 'boolean' ? source.is_operational : true,
    is_hardened: typeof source.is_hardened === 'boolean' ? source.is_hardened : false,
  in_recovery:
      typeof source.in_recovery === 'boolean'
        ? source.in_recovery
        : pendingRecoveries.includes(asString(source.node_id)),
    is_critical: typeof source.is_critical === 'boolean' ? source.is_critical : false,
    lat,
    lon,
    real_name: typeof source.real_name === 'string' ? source.real_name : undefined,
  observation_delayed:
      typeof source.observation_delayed === 'boolean' ? source.observation_delayed : undefined,
  service_radius_km:
      asOptionalNumber(source.service_radius_km),
  observation_confidence:
      asOptionalNumber(source.observation_confidence),
  coverage_quality:
      asOptionalNumber(source.coverage_quality),
  };
};

const asEdge = (edge: CGEdge, failedSet: Set<string>): CGEdge | null => {
  const source_id = asString(edge.source_id);
  const target_id = asString(edge.target_id);
  if (!source_id || !target_id) return null;

  return {
    source_id,
    target_id,
    active: typeof edge.active === 'boolean' ? edge.active : true,
    kind:
      edge.kind ??
      (failedSet.has(source_id) || failedSet.has(target_id) ? 'cascade' : 'base'),
  };
};

export const normalizeObservation = (obs: CGObservation): CGObservation => {
  const active_failures = asStringArray(obs.active_failures);
  const pending_recoveries = asStringArray(obs.pending_recoveries);
  const failedSet = new Set(active_failures);

  const diagnostics = obs.diagnostics
    ? {
        critical_failures: asStringArray(obs.diagnostics.critical_failures),
        at_risk_nodes: asStringArray(obs.diagnostics.at_risk_nodes),
        dependency_alerts: asStringArray(obs.diagnostics.dependency_alerts),
        recommended_recovery_order: asStringArray(obs.diagnostics.recommended_recovery_order),
        system_pressure: asNumber(obs.diagnostics.system_pressure, 0),
      }
    : DEFAULT_DIAGNOSTICS;

  // ── Bug 4 fix: normalize nodes FIRST so we can compute sector averages from them ──
  // This gives us a reliable fallback when the backend sector_summary is missing, null,
  // or uses different key names than expected.
  const normalizedNodes = Array.isArray(obs.nodes)
    ? obs.nodes
        .map((node) => asNode(node, pending_recoveries))
        .filter((node) => node.node_id.length > 0)
    : [];

  // Compute per-sector average health from nodes (0-1 range, clamped)
  const nodeSectorTotals: Record<string, number[]> = {};
  for (const node of normalizedNodes) {
    if (node.sector) {
      if (!nodeSectorTotals[node.sector]) nodeSectorTotals[node.sector] = [];
      nodeSectorTotals[node.sector].push(node.health);
    }
  }
  const nodeAvgForSector = (sector: string): number => {
    const vals = nodeSectorTotals[sector];
    if (!vals || vals.length === 0) return 1; // fallback: assume healthy if no nodes
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  };

  // Build sector_summary: prefer backend value when it's a valid finite number,
  // fall back to node-computed average (covers null / missing / non-finite cases).
  const rawSS = (obs.sector_summary ?? {}) as unknown as Record<string, unknown>;
  const toSectorVal = (key: string): number => {
    const raw = rawSS[key];
    if (typeof raw === 'number' && Number.isFinite(raw)) {
      return Math.max(0, Math.min(1, raw));
    }
    return nodeAvgForSector(key);
  };

  return {
    ...obs,
    nodes: normalizedNodes,
    edges: Array.isArray(obs.edges)
      ? obs.edges
          .map((edge) => asEdge(edge, failedSet))
          .filter((edge): edge is CGEdge => edge !== null)
      : [],
    sector_summary: {
      power:    toSectorVal('power'),
      water:    toSectorVal('water'),
      hospital: toSectorVal('hospital'),
      telecom:  toSectorVal('telecom'),
    },
    active_failures,
    pending_recoveries,
    budget_remaining: asNumber(obs.budget_remaining, 0),
    step: asNumber(obs.step, 0),
    max_steps: asNumber(obs.max_steps, 10),
    diagnostics,
    reward: asNumber(obs.reward, 0),
    done: typeof obs.done === 'boolean' ? obs.done : false,
    task_id: asString(obs.task_id, ''),
    info: asString(obs.info, ''),
    radius_coverage_events: asStringArray(obs.radius_coverage_events),
    narrative_phase: asString(obs.narrative_phase, 'pre_storm'),
    consecutive_waits: asNumber(obs.consecutive_waits, 0),
    upcoming_stress_level: asString(obs.upcoming_stress_level, 'none'),
    metadata:
      obs.metadata && typeof obs.metadata === 'object'
        ? (obs.metadata as Record<string, unknown>)
        : {},
  };
};

export interface LivePresentation {
  attacker: AgentTurn;
  defender: AgentTurn;
  attackPressure: number;
  defenseScore: number;
  eventLines: EventLine[];
  timelineEntry: TimelineEntry;
  whatsHappening: string;
}

export const deriveLivePresentation = (obs: CGObservation): LivePresentation => {
  const diagnostics = obs.diagnostics ?? DEFAULT_DIAGNOSTICS;
  const criticals = diagnostics.critical_failures.length
    ? diagnostics.critical_failures
    : obs.active_failures;
  const recommended = diagnostics.recommended_recovery_order;
  const alerts = diagnostics.dependency_alerts;
  const atRisk = diagnostics.at_risk_nodes;
  const pressure = diagnostics.system_pressure;
  const cascadeDepth = obs.edges.filter((edge) => edge.kind === 'cascade').length;
  const scoreLabel = Number.isFinite(obs.reward) ? obs.reward.toFixed(3) : '0.000';

  const attackerThink = criticals.length
    ? `Target acquired: ${criticals.slice(0, 2).join(', ')}. System pressure=${pressure.toFixed(2)}. Cascade active across ${cascadeDepth} edges.`
    : `Analyzing topology. No critical failures yet. At-risk nodes: ${atRisk.slice(0, 2).join(', ') || 'none'}. Monitoring budget=${obs.budget_remaining.toFixed(1)}.`;

  const defenderThink = recommended.length
    ? `Recommended recovery order: ${recommended.slice(0, 3).join(' -> ')}. ${alerts.slice(0, 1).join('. ') || 'Dependencies under review.'} Budget=${obs.budget_remaining.toFixed(1)}.`
    : `All sectors stable. Budget=${obs.budget_remaining.toFixed(1)}. Pressure=${pressure.toFixed(2)}. Monitoring.`;

  const attacker: AgentTurn = {
    think: attackerThink,
    action: criticals.length ? `INJECT -> ${criticals[0]}` : 'MONITOR -> topology',
  };

  const defender: AgentTurn = {
    think: defenderThink,
    action: recommended.length ? `recover(${recommended[0]})` : obs.done ? 'COMPLETE' : 'wait',
  };

  const eventLines: EventLine[] = [
    ...criticals.map((nodeId) => ({
      ts: now(),
      level: 'crit' as const,
      msg: `[CRITICAL] ${nodeId} - cascade active`,
    })),
    ...obs.pending_recoveries.map((nodeId) => ({
      ts: now(),
      level: 'ok' as const,
      msg: `[RECOVERING] ${nodeId} - in progress`,
    })),
  ];

  if (eventLines.length === 0) {
    eventLines.push({
      ts: now(),
      level: pressure > 0.3 ? 'warn' : 'info',
      msg: `[STEP ${obs.step}] ${pressure > 0.3 ? 'Pressure rising' : 'All nodes nominal'} - pressure=${pressure.toFixed(2)}`,
    });
  }

  const whatsHappening = criticals.length
    ? `CRITICAL: ${criticals.slice(0, 2).join(', ')} - cascade depth ${cascadeDepth}. Select a node and dispatch an action.`
    : pressure > 0.3
      ? `At-risk nodes: ${atRisk.slice(0, 2).join(', ') || 'monitoring'}. System pressure ${(pressure * 100).toFixed(0)}%. Consider hardening or shedding load.`
      : `Step ${obs.step}/${obs.max_steps}: All sectors stable. Score ${scoreLabel}. ${obs.done ? 'Episode complete.' : 'Monitoring.'}`;

  const tone: TimelineEntry['tone'] = criticals.length
    ? 'crit'
    : pressure > 0.3
      ? 'warn'
      : obs.done
        ? 'ok'
        : 'info';

  const title = criticals.length
    ? `${criticals[0]} critical failure`
    : obs.done
      ? 'Episode complete'
      : pressure > 0.3
        ? 'System under pressure'
        : 'System nominal';

  const description = criticals.length
    ? `Cascade active across ${cascadeDepth} edges. ${alerts[0] ?? 'Dispatch recovery on the recommended node order.'}`
    : pressure > 0.3
      ? `${alerts[0] ?? 'Dependencies under stress.'} At-risk nodes: ${atRisk.slice(0, 3).join(', ') || 'monitoring'}.`
      : obs.done
        ? `All sectors stable. Final reward ${scoreLabel}.`
        : obs.info || 'All sectors stable. Monitoring.';

  const tags = [
    `phase:${obs.narrative_phase ?? 'pre_storm'}`,
    `pressure:${(pressure * 100).toFixed(0)}%`,
    `score:${scoreLabel}`,
    ...(recommended[0] ? [`next:${recommended[0]}`] : []),
  ];

  return {
    attacker,
    defender,
    attackPressure: clamp(Math.max(criticals.length ? 0.25 : 0.08, pressure), 0.08, 0.98),
    defenseScore: clamp(1 - pressure, 0.08, 0.98),
    eventLines,
    timelineEntry: {
      step: obs.step,
      title,
      description,
      tags,
      tone,
    },
    whatsHappening,
  };
};

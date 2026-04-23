import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Navbar } from '@/cascade/components/Navbar';
import { CityModal } from '@/cascade/components/CityModal';
import { Hero } from '@/cascade/components/Hero';
import { LiveMap } from '@/cascade/components/LiveMap';
import { AIDuel } from '@/cascade/components/AIDuel';
import { NeuralGraph } from '@/cascade/components/NeuralGraph';
import { ThreatIntel } from '@/cascade/components/ThreatIntel';
import { Analytics } from '@/cascade/components/Analytics';
import { Timeline } from '@/cascade/components/Timeline';
import { HowImproved } from '@/cascade/components/HowImproved';
import { Logs } from '@/cascade/components/Logs';
import { Footer } from '@/cascade/components/Footer';
import { CITIES, cityByKey, type CityKey } from '@/cascade/data/cities';
import { SCRIPTED_EPISODE, TOTAL_STEPS } from '@/cascade/data/episode';
import type { CGObservation, EventLine, TimelineEntry, AgentTurn } from '@/cascade/data/types';
import { useEnvSocket } from '@/cascade/hooks/useEnvSocket';
import { ENV_BASE } from '@/cascade/config/env';
import {
  deriveLivePresentation,
  normalizeObservation,
} from '@/cascade/lib/liveObservation';
import { useReveal } from '@/cascade/hooks/useReveal';

const SPEED_MS: Record<number, number> = { 0.5: 2400, 1: 1400, 2: 700, 4: 300 };
const MAX_EVENT_LINES = 20;
const MAX_LOG_LINES = 200;
const UI_FLOW_DEBUG = import.meta.env.DEV;

interface BackendLegalAction {
  action_type: string;
  target_node_id: string | null;
  parameters: Record<string, unknown>;
}

const FALLBACK_ACTION_TYPES = [
  'recover', 'harden', 'shed_load', 'coordinate', 'wait',
  'isolate', 'reroute', 'prioritize', 'deploy_repair_crew',
  'emergency_shutdown', 'cross_sector_bridge', 'patch_scada',
  'redistribute_load', 'request_mutual_aid', 'controlled_cascade', 'multi_sector_lockdown',
];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const toLegalAction = (value: unknown): BackendLegalAction | null => {
  if (!isRecord(value)) return null;
  if (typeof value.action_type !== 'string' || value.action_type.length === 0) return null;
  return {
    action_type: value.action_type,
    target_node_id: typeof value.target_node_id === 'string' ? value.target_node_id : null,
    parameters: isRecord(value.parameters) ? value.parameters : {},
  };
};

const actionCatalogFromMetadata = (metadata?: Record<string, unknown>) => {
  // Bug 3 fix: always start with the full FALLBACK_ACTION_TYPES (all 16) as base.
  // Metadata catalogs from older backend versions may only carry a subset.
  const sampleMap: Record<string, BackendLegalAction> = {};
  if (isRecord(metadata?.legal_action_samples)) {
    for (const [actionType, raw] of Object.entries(metadata.legal_action_samples)) {
      const parsed = toLegalAction(raw);
      if (parsed) sampleMap[actionType] = parsed;
    }
  }

  // Merge: FALLBACK_ACTION_TYPES covers all 16 guaranteed actions;
  // any extra types from the backend catalog are appended.
  const backendCatalog = Array.isArray(metadata?.action_catalog)
    ? metadata.action_catalog.filter((v): v is string => typeof v === 'string')
    : [];
  const merged = [
    ...FALLBACK_ACTION_TYPES,
    ...backendCatalog.filter((t) => !FALLBACK_ACTION_TYPES.includes(t)),
  ];

  return {
    actionTypes: merged,
    actionSamples: sampleMap,
  };
};

const uiFlowLog = (...args: unknown[]) => {
  if (!UI_FLOW_DEBUG) return;
  console.debug('[Index]', ...args);
};

export default function Index() {
  useReveal();

  const [activeCityKey, setActiveCityKey] = useState<CityKey>('london');
  const [selectedCityKey, setSelectedCityKey] = useState<CityKey>('london');
  const city = cityByKey(activeCityKey);
  // Fix 2: track the task chosen in the city modal (defaults to city.taskId)
  const [selectedTaskId, setSelectedTaskId] = useState<string>(city.taskId);
  const [hasLaunched, setHasLaunched] = useState(false);
  const [modalOpen, setModalOpen] = useState(true);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<0.5 | 1 | 2 | 4>(1);
  const [busy, setBusy] = useState(false);
  const [events, setEvents] = useState<EventLine[]>([]);
  const [logs, setLogs] = useState<EventLine[]>([]);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [rewardSeries, setRewardSeries] = useState<number[]>([0]);
  const [attacker, setAttacker] = useState<AgentTurn>(SCRIPTED_EPISODE[0].attacker);
  const [defender, setDefender] = useState<AgentTurn>(SCRIPTED_EPISODE[0].defender);
  const [attackPressure, setAttackPressure] = useState(SCRIPTED_EPISODE[0].attackPressure);
  const [defenseScore, setDefenseScore] = useState(SCRIPTED_EPISODE[0].defenseScore);
  const [whatsHappening, setWH] = useState(SCRIPTED_EPISODE[0].whatsHappening);
  const [obs, setObs] = useState<CGObservation>(normalizeObservation(SCRIPTED_EPISODE[0].obs));
  const [backendActionCatalog, setBackendActionCatalog] = useState<string[]>([]);
  const liveStepsRef = useRef<Set<number>>(new Set());
  const prevConnStateRef = useRef<'connecting' | 'live' | 'sim' | 'error'>('sim');
  const playRef = useRef<number | null>(null);
  // Fix 3: safety timeout ref — fires 4 s after busy is set to true with no response
  const busyTimeoutRef = useRef<number | null>(null);

  const clearBusyTimeout = useCallback(() => {
    if (busyTimeoutRef.current !== null) {
      window.clearTimeout(busyTimeoutRef.current);
      busyTimeoutRef.current = null;
    }
  }, []);

  const setBusyWithTimeout = useCallback((value: boolean) => {
    if (!value) {
      clearBusyTimeout();
      setBusy(false);
      return;
    }

    clearBusyTimeout();
    busyTimeoutRef.current = window.setTimeout(() => {
      busyTimeoutRef.current = null;
      console.warn('[CascadeGuard] busy timeout — forcing reset');
      setBusy(false);
    }, 4000);
    setBusy(true);
  }, [clearBusyTimeout]);

  const applyScripted = useCallback((idx: number, replace = false) => {
    const scripted = SCRIPTED_EPISODE[Math.max(0, Math.min(SCRIPTED_EPISODE.length - 1, idx))];
    const normalized = normalizeObservation(scripted.obs);

    setObs(normalized);
    setStepIdx(normalized.step);
    setAttacker(scripted.attacker);
    setDefender(scripted.defender);
    setAttackPressure(scripted.attackPressure);
    setDefenseScore(scripted.defenseScore);
    setWH(scripted.whatsHappening);
    setEvents((prev) =>
      replace
        ? scripted.events.slice(0, MAX_EVENT_LINES)
        : [...scripted.events, ...prev].slice(0, MAX_EVENT_LINES),
    );
    setLogs((prev) =>
      replace
        ? scripted.events.slice(-MAX_LOG_LINES)
        : [...prev, ...scripted.events].slice(-MAX_LOG_LINES),
    );
    setTimeline((prev) => (replace ? [scripted.timeline] : [...prev, scripted.timeline]));
    setRewardSeries((prev) =>
      replace ? [0, normalized.reward].slice(-60) : [...prev, normalized.reward].slice(-60),
    );
  }, []);

  const resetToSimulation = useCallback(() => {
    liveStepsRef.current.clear();
    setBusyWithTimeout(false);
    setPlaying(false);
    applyScripted(0, true);
  }, [applyScripted, setBusyWithTimeout]);

  const appendEventLines = useCallback((entries: EventLine[], replace = false) => {
    setEvents((prev) =>
      replace ? entries.slice(0, MAX_EVENT_LINES) : [...entries, ...prev].slice(0, MAX_EVENT_LINES),
    );
    setLogs((prev) =>
      replace ? entries.slice(-MAX_LOG_LINES) : [...prev, ...entries].slice(-MAX_LOG_LINES),
    );
  }, []);

  const handleObs = useCallback((incoming: CGObservation) => {
    const normalized = normalizeObservation(incoming);
    const live = deriveLivePresentation(normalized);
    const replace = liveStepsRef.current.size === 0;

    liveStepsRef.current.add(normalized.step);

    setObs(normalized);
    setStepIdx(normalized.step ?? 0);
    // Fix 3: response arrived — cancel the 4-second safety timeout and clear busy
    clearBusyTimeout();
    setBusy(false);
    // Only stop auto-play when we've genuinely reached the episode end (step >= max_steps).
    // Do NOT stop on obs.done alone — the backend sets done=true during crisis_peak even
    // before step 30, which was causing the play button to stop prematurely (Bug 2 fix).
    if (normalized.step >= normalized.max_steps) setPlaying(false);

    setAttacker(live.attacker);
    setDefender(live.defender);
    setAttackPressure(live.attackPressure);
    setDefenseScore(live.defenseScore);
    setWH(live.whatsHappening);
    appendEventLines(live.eventLines, replace);
    setTimeline((prev) => {
      if (replace) return [live.timelineEntry];
      const next = prev.filter((entry) => entry.step !== live.timelineEntry.step);
      return [...next, live.timelineEntry];
    });
    setRewardSeries((prev) =>
      replace
        ? [0, normalized.reward].slice(-60)
        : [...prev, normalized.reward].slice(-60),
    );

    const coordsCount = normalized.nodes.filter(
      (node) => Number.isFinite(node.lat) && Number.isFinite(node.lon),
    ).length;
    const legalActionTypes = Array.isArray(normalized.metadata?.legal_action_types)
      ? normalized.metadata.legal_action_types
      : [];
    uiFlowLog('socket -> state', {
      step: normalized.step,
      reward: normalized.reward,
      nodes: normalized.nodes.length,
      nodes_with_coords: coordsCount,
      cascade_edges: normalized.edges.filter((edge) => edge.kind === 'cascade').length,
      legal_action_types: legalActionTypes,
    });
  }, [appendEventLines, clearBusyTimeout]);

  const handleSocketError = useCallback((message: string) => {
    setBusyWithTimeout(false);
    appendEventLines([
      {
        ts: new Date().toTimeString().slice(0, 8),
        level: 'crit',
        msg: message,
      },
    ]);
  }, [appendEventLines, setBusyWithTimeout]);

  // Fix 2: use the user-selected task id (from CityModal dropdown) instead of the
  // hardcoded city.taskId so the episode respects the chosen difficulty/scenario.
  const sock = useEnvSocket(selectedTaskId, hasLaunched, handleObs, handleSocketError);
  const connState = sock.state;
  const sendAction = sock.send;
  const resetSocket = sock.reset;
  const reconnectSocket = sock.reconnect;

  useEffect(() => {
    if (!hasLaunched) return;
    resetToSimulation();
  }, [activeCityKey, hasLaunched, resetToSimulation]);

  useEffect(() => {
    if (!hasLaunched) {
      prevConnStateRef.current = connState;
      return;
    }

    const prev = prevConnStateRef.current;
    if (connState === 'live' && prev !== 'live') {
      liveStepsRef.current.clear();
    }
    if (connState === 'sim' && prev !== 'sim') {
      resetToSimulation();
    }
    if (connState !== 'live') {
      setBusyWithTimeout(false);
    }

    prevConnStateRef.current = connState;
  }, [connState, hasLaunched, resetToSimulation, setBusyWithTimeout]);

  useEffect(() => {
    let cancelled = false;

    const loadActions = async () => {
      try {
        const response = await fetch(`${ENV_BASE}/actions`);
        if (!response.ok) return;

        const payload = await response.json();
        const actions = Array.isArray(payload?.actions)
          ? payload.actions.filter((value: unknown): value is string => typeof value === 'string')
          : [];
        if (!cancelled && actions.length > 0) {
          setBackendActionCatalog(actions);
          uiFlowLog('loaded action catalog', { count: actions.length, actions: actions.slice(0, 10) });
        }
      } catch (error) {
        uiFlowLog('action catalog fetch failed', String(error));
      }
    };

    void loadActions();

    return () => {
      cancelled = true;
    };
  }, []);

  // Fix 3: 4-second busy safety timeout — set once when we enter busy state,
  // cancelled in handleObs when a real WS response arrives.
  useEffect(() => {
    if (!busy) {
      // Not busy — ensure any lingering timeout is cleared
      if (busyTimeoutRef.current !== null) {
        window.clearTimeout(busyTimeoutRef.current);
        busyTimeoutRef.current = null;
      }
      return;
    }

    if (busyTimeoutRef.current === null) busyTimeoutRef.current = window.setTimeout(() => {
      busyTimeoutRef.current = null;
      console.warn('[CascadeGuard] busy timeout — forcing reset');
      setBusy(false);
    }, 4000);

    return () => {
      if (busyTimeoutRef.current !== null) {
        window.clearTimeout(busyTimeoutRef.current);
        busyTimeoutRef.current = null;
      }
    };
  }, [busy]);

  // Play loop — no longer stops on obs.done alone.
  // The backend sets done=true during crisis_peak before step 30 which was halting
  // the episode. We now only stop at step >= max_steps (true episode end).
  useEffect(() => {
    if (!playing) {
      if (playRef.current) window.clearTimeout(playRef.current);
      return;
    }

    playRef.current = window.setTimeout(() => {
      if (connState === 'live') {
        // Stop only when we've genuinely hit the step limit.
        if (obs.step >= obs.max_steps) {
          setPlaying(false);
          return;
        }
        // If busy, wait — the 4-second watchdog (above) will unblock us if WS drops.
        if (busy) return;

        appendEventLines([
          {
            ts: new Date().toTimeString().slice(0, 8),
            level: 'info',
            msg: '[STEP] Advance one step (wait)',
          },
        ]);
        setBusyWithTimeout(true);
        sendAction('wait', null);
        return;
      }

      // Simulation mode — advance scripted episode
      setStepIdx((currentStep) => {
        const nextStep = Math.min(TOTAL_STEPS - 1, currentStep + 1);
        applyScripted(nextStep);
        if (nextStep >= TOTAL_STEPS - 1) setPlaying(false);
        return nextStep;
      });
    }, SPEED_MS[speed]);

    return () => {
      if (playRef.current) window.clearTimeout(playRef.current);
    };
  }, [appendEventLines, applyScripted, busy, connState, obs.step, obs.max_steps, playing, sendAction, setBusyWithTimeout, speed]);

  const onStep = () => {
    setPlaying(false);
    if (connState === 'live') {
      appendEventLines([
        {
          ts: new Date().toTimeString().slice(0, 8),
          level: 'info',
          msg: '[STEP] Advance one step (wait)',
        },
      ]);
      setBusyWithTimeout(true);
      sendAction('wait', null);
      return;
    }

    setStepIdx((currentStep) => {
      const nextStep = Math.min(TOTAL_STEPS - 1, currentStep + 1);
      applyScripted(nextStep);
      return nextStep;
    });
  };

  const onReset = () => {
    setPlaying(false);
    setBusyWithTimeout(false);
    liveStepsRef.current.clear();

    if (connState === 'live') {
      appendEventLines(
        [
          {
            ts: new Date().toTimeString().slice(0, 8),
            level: 'info',
            msg: '[RESET] New episode started',
          },
        ],
        true,
      );
      setTimeline([]);
      setRewardSeries([0]);
      setStepIdx(0);
      resetSocket();
      return;
    }

    resetToSimulation();
  };

  const onDispatch = (
    action: string,
    target: string | null,
    parameters: Record<string, unknown> = {},
  ) => {
    if (connState === 'live') {
      appendEventLines([
        {
          ts: new Date().toTimeString().slice(0, 8),
          level: 'info',
          msg: `[DISPATCH] ${action}${target ? ` -> ${target}` : ''}`,
        },
      ]);
      setBusyWithTimeout(true);
      sendAction(action, target, parameters);
      return;
    }

    appendEventLines([
      {
        ts: new Date().toTimeString().slice(0, 8),
        level: 'info',
        msg: `dispatch ${action}${target ? ` -> ${target}` : ''}`,
      },
    ]);
    setStepIdx((currentStep) => {
      const nextStep = Math.min(TOTAL_STEPS - 1, currentStep + 1);
      applyScripted(nextStep);
      return nextStep;
    });
  };

  const score = useMemo(
    () => (connState === 'live' ? obs.reward : rewardSeries.reduce((sum, value) => sum + value, 0)),
    [connState, obs.reward, rewardSeries],
  );
  const maxSteps = useMemo(
    () => (connState === 'live' ? (obs.max_steps || TOTAL_STEPS) : TOTAL_STEPS),
    [connState, obs.max_steps],
  );
  const cascadeEdgeKeys = useMemo(
    () => obs.edges.filter((edge) => edge.kind === 'cascade').map((edge) => `${edge.source_id}->${edge.target_id}`),
    [obs.edges],
  );
  const criticalNodeIds = useMemo(
    () => obs.nodes.filter((node) => !node.is_operational).map((node) => node.node_id),
    [obs.nodes],
  );
  const nodesAtRisk =
    obs.diagnostics?.at_risk_nodes?.length || obs.nodes.filter((node) => node.health < 0.5).length;

  const { actionTypes: availableActionTypes, actionSamples } = useMemo(
    () => actionCatalogFromMetadata(obs.metadata),
    [obs.metadata],
  );

  const dispatchActionTypes = backendActionCatalog.length > 0
    ? backendActionCatalog
    : availableActionTypes;

  const dl = (filename: string, content: string, type = 'text/plain;charset=utf-8') => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const stamp = () => {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
  };

  const escapeHtml = (s: string) => s.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' } as Record<string, string>)[c]);

  const onDownloadReport = () => {
    const generatedAt = new Date().toISOString();
    const sectorRow = (label: string, v: number) => {
      const pct = Math.round(v * 100);
      const tone = v >= 0.8 ? 'ok' : v >= 0.5 ? 'warn' : 'crit';
      return `<tr><td>${label}</td><td class="num">${pct}%</td><td><div class="bar"><div class="bar-fill ${tone}" style="width:${pct}%"></div></div></td></tr>`;
    };
    const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>CascadeGuard · Episode 042 · ${escapeHtml(city.name)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root { --ink:#0d0d0d; --bg:#fafafa; --line:rgba(0,0,0,.08); --muted:#666; --accent:#1446f5; --green:#00a878; --amber:#d97706; --red:#e8192c; --purple:#7c3aed; }
  *{box-sizing:border-box} html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);font-family:'DM Sans',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
  .page{max-width:880px;margin:0 auto;padding:48px 56px 80px}
  .mono{font-family:'DM Mono',ui-monospace,monospace}
  .hero{position:relative;border-radius:20px;overflow:hidden;color:#fff;padding:36px 40px;background:radial-gradient(120% 140% at 0% 0%,#1446f5 0%,#0d0d0d 60%,#000 100%)}
  .hero::after{content:"";position:absolute;inset:0;background-image:radial-gradient(rgba(255,255,255,.08) 1px,transparent 1px);background-size:22px 22px;opacity:.4;pointer-events:none}
  .pill{display:inline-flex;align-items:center;gap:8px;padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.12);font-size:11px;letter-spacing:.18em;text-transform:uppercase}
  .pill .dot{width:7px;height:7px;border-radius:50%;background:#00ff9c;box-shadow:0 0 8px #00ff9c}
  h1{font-size:42px;line-height:1.05;letter-spacing:-.03em;margin:14px 0 8px;font-weight:700}
  h1 .alt{color:#7aa2ff}
  .sub{color:rgba(255,255,255,.7);font-size:14px}
  .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:22px;position:relative;z-index:2}
  .stat{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:14px}
  .stat .k{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:rgba(255,255,255,.55)}
  .stat .v{font-size:22px;font-weight:700;margin-top:4px;font-family:'DM Mono',monospace}
  section{margin-top:36px}
  h2{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:0 0 14px;font-family:'DM Mono',monospace}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td,th{padding:9px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
  td.num{font-family:'DM Mono',monospace;width:60px}
  .bar{height:6px;border-radius:99px;background:rgba(0,0,0,.06);overflow:hidden}
  .bar-fill{height:100%;border-radius:99px}
  .bar-fill.ok{background:var(--green)} .bar-fill.warn{background:var(--amber)} .bar-fill.crit{background:var(--red)}
  .tl{position:relative;padding-left:22px}
  .tl::before{content:"";position:absolute;left:6px;top:6px;bottom:6px;width:2px;background:linear-gradient(180deg,var(--accent),var(--purple))}
  .tl-item{position:relative;padding:10px 0 14px}
  .tl-item::before{content:"";position:absolute;left:-19px;top:14px;width:10px;height:10px;border-radius:50%;background:#fff;border:2px solid var(--accent)}
  .tl-item.crit::before{border-color:var(--red)} .tl-item.ok::before{border-color:var(--green)} .tl-item.warn::before{border-color:var(--amber)}
  .tag{display:inline-block;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-family:'DM Mono',monospace}
  .tl-item .title{font-weight:600;font-size:14px;margin:2px 0 3px}
  .tl-item .desc{color:var(--muted);font-size:13px}
  .logs{background:#0d0d0d;color:#e7e7e7;border-radius:14px;padding:18px 20px;font-family:'DM Mono',monospace;font-size:12px;line-height:1.7;max-height:none}
  .logs .ts{color:rgba(255,255,255,.4)} .logs .lv{display:inline-block;padding:1px 6px;border-radius:4px;margin:0 8px;font-size:10px;font-weight:700}
  .lv.info{background:rgba(255,255,255,.1);color:#fff} .lv.ok{background:rgba(0,168,120,.18);color:#00a878}
  .lv.warn{background:rgba(217,119,6,.18);color:#f59e0b} .lv.crit{background:rgba(232,25,44,.2);color:#ff6b78}
  footer{margin-top:46px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:12px;display:flex;justify-content:space-between}
  @media print{ body{background:#fff} .page{padding:24px} .hero{break-inside:avoid} section{break-inside:avoid} }
</style></head><body>
<div class="page">
  <div class="hero">
    <span class="pill"><span class="dot"></span>CascadeGuard · Episode Report</span>
    <h1>Cascade<span class="alt">Guard</span> · ${escapeHtml(city.name)}</h1>
    <div class="sub">Episode 042 · ${escapeHtml(city.taskId)} · generated ${escapeHtml(generatedAt)}</div>
    <div class="grid">
      <div class="stat"><div class="k">Steps</div><div class="v">${stepIdx + 1}/${maxSteps}</div></div>
      <div class="stat"><div class="k">Episode Score</div><div class="v">${score.toFixed(2)}</div></div>
      <div class="stat"><div class="k">Budget Left</div><div class="v">$${obs.budget_remaining.toLocaleString()}</div></div>
      <div class="stat"><div class="k">Nodes at Risk</div><div class="v">${nodesAtRisk}</div></div>
    </div>
  </div>

  <section>
    <h2>/ sector health</h2>
    <table><tbody>
      ${sectorRow('Power', obs.sector_summary.power)}
      ${sectorRow('Water', obs.sector_summary.water)}
      ${sectorRow('Hospital', obs.sector_summary.hospital)}
      ${sectorRow('Telecom', obs.sector_summary.telecom)}
    </tbody></table>
  </section>

  <section>
    <h2>/ step-by-step trace</h2>
    <div class="tl">
      ${timeline.length === 0
        ? '<div class="tl-item"><div class="title">No timeline yet</div><div class="desc">Press Play on the Live Map to begin the episode.</div></div>'
        : timeline.map(t => `
        <div class="tl-item ${escapeHtml(t.tone)}">
          <span class="tag">step ${String(t.step).padStart(2,'0')} · ${escapeHtml(t.tone)}</span>
          <div class="title">${escapeHtml(t.title)}</div>
          <div class="desc">${escapeHtml(t.description)}</div>
        </div>`).join('')}
    </div>
  </section>

  <section>
    <h2>/ event log (${logs.length})</h2>
    <div class="logs">
      ${logs.length === 0 ? '<div class="ts">// no entries</div>'
        : logs.map(l => `<div><span class="ts">${escapeHtml(l.ts)}</span><span class="lv ${escapeHtml(l.level)}">${escapeHtml(l.level.toUpperCase())}</span>${escapeHtml(l.msg)}</div>`).join('')}
    </div>
  </section>

  <footer>
    <span>CascadeGuard · Infrastructure Resilience Intelligence</span>
    <span class="mono">Meta × HuggingFace Hackathon · OpenEnv · 2025</span>
  </footer>
</div>
</body></html>`;
    dl(`cascadeguard_${city.key}_ep042_${stamp()}.html`, html, 'text/html;charset=utf-8');
  };

  const onDownloadLogs = () => {
    const header = [
      '╔══════════════════════════════════════════════════════════════════╗',
      '║  CascadeGuard · Event Log                                       ║',
      '╠══════════════════════════════════════════════════════════════════╣',
      `║  Episode  : 042 · ${city.taskId}`.padEnd(67) + '║',
      `║  City     : ${city.name}`.padEnd(67) + '║',
      `║  Step     : ${stepIdx + 1}/${maxSteps}`.padEnd(67) + '║',
      `║  Score    : ${score.toFixed(2)}`.padEnd(67) + '║',
      `║  Budget   : $${obs.budget_remaining.toLocaleString()}`.padEnd(67) + '║',
      `║  Exported : ${new Date().toISOString()}`.padEnd(67) + '║',
      '╚══════════════════════════════════════════════════════════════════╝',
      '',
    ].join('\n');
    const body = logs.map((line) => `${line.ts}  [${line.level.toUpperCase().padEnd(4)}]  ${line.msg}`).join('\n');
    dl(`cascadeguard_${city.key}_logs_${stamp()}.log`, header + body + '\n');
  };

  const onDownloadJson = () => {
    const payload = {
      $schema: 'https://cascadeguard.dev/schema/episode.v1.json',
      meta: {
        product: 'CascadeGuard',
        version: '1.0.0',
        episode_id: '042',
        task_id: city.taskId,
        city: { key: city.key, name: city.name, center: city.center, zoom: city.zoom },
        connection: connState,
        exported_at: new Date().toISOString(),
      },
      summary: {
        steps_taken: stepIdx + 1,
        max_steps: maxSteps,
        episode_score: Number(score.toFixed(4)),
        budget_remaining: obs.budget_remaining,
        nodes_at_risk: nodesAtRisk,
        critical_node_ids: criticalNodeIds,
        sector_health: obs.sector_summary,
      },
      reward_series: rewardSeries.map((value, index) => ({ step: index, reward: Number(value.toFixed(4)) })),
      observation: obs,
      timeline,
      events: logs,
    };
    dl(`cascadeguard_${city.key}_ep042_${stamp()}.json`, JSON.stringify(payload, null, 2), 'application/json');
  };

  const onDownloadCsv = () => {
    const cumulative = rewardSeries.reduce<number[]>((acc, value) => {
      acc.push((acc[acc.length - 1] ?? 0) + value);
      return acc;
    }, []);
    const lines = [
      '# CascadeGuard reward curve',
      `# episode=042 task=${city.taskId} city=${city.name} (${city.key})`,
      `# steps=${stepIdx + 1}/${maxSteps} final_score=${score.toFixed(4)} budget_remaining=${obs.budget_remaining}`,
      `# exported=${new Date().toISOString()}`,
      'step,reward,cumulative_score',
      ...rewardSeries.map((value, index) => `${index},${value.toFixed(4)},${cumulative[index].toFixed(4)}`),
    ];
    dl(`cascadeguard_${city.key}_reward_${stamp()}.csv`, lines.join('\n'), 'text/csv;charset=utf-8');
  };

  return (
    <div className="relative min-h-screen text-foreground">
      <div className="cg-dotgrid fixed inset-0 -z-10 pointer-events-none" />
      <Navbar
        city={city}
        onOpenCity={() => {
          setSelectedCityKey(activeCityKey);
          setModalOpen(true);
        }}
      />
      <CityModal
        open={modalOpen}
        selected={selectedCityKey}
        onClose={() => setModalOpen(false)}
        onSelect={setSelectedCityKey}
        onLaunch={(key, taskId) => {
          const relaunchSameCity = hasLaunched && key === activeCityKey && taskId === selectedTaskId;
          setSelectedCityKey(key);
          setActiveCityKey(key);
          setSelectedTaskId(taskId);
          setHasLaunched(true);
          setModalOpen(false);
          if (relaunchSameCity) {
            liveStepsRef.current.clear();
            resetToSimulation();
            reconnectSocket(taskId);
          }
        }}
      />
      <Hero city={city.name} step={stepIdx} episodeId="042" />
      <LiveMap
        city={city}
        nodes={obs.nodes}
        edges={obs.edges}
        pendingRecoveries={obs.pending_recoveries}
        sectorSummary={obs.sector_summary}
        step={stepIdx}
        maxSteps={maxSteps}
        budget={obs.budget_remaining}
        score={score}
        hasCritical={criticalNodeIds.length > 0}
        events={events}
        connState={connState}
        isPlaying={playing}
        speed={speed}
        onPlayPause={() => setPlaying((value) => !value)}
        onStep={onStep}
        onReset={onReset}
        onSpeed={setSpeed}
        onDispatch={onDispatch}
        availableActionTypes={dispatchActionTypes}
        actionSamples={actionSamples}
        busy={busy}
        whatsHappening={whatsHappening}
        episodeId="042"
      />
      <AIDuel
        attacker={attacker}
        defender={defender}
        attackPressure={attackPressure}
        defenseScore={defenseScore}
      />
      <NeuralGraph
        cascadeDepth={cascadeEdgeKeys.length}
        episodeScore={score}
        nodesAtRisk={nodesAtRisk}
        criticalNodeIds={criticalNodeIds}
        cascadeEdgeKeys={cascadeEdgeKeys}
      />
      <ThreatIntel terminalLines={events} rewardSeries={rewardSeries} step={stepIdx} />
      <Analytics
        episodeScore={score}
        budgetRemaining={obs.budget_remaining}
        nodesAtRisk={nodesAtRisk}
        stepsTaken={stepIdx}
        sectorHealth={obs.sector_summary}
      />
      <Timeline entries={timeline} />
      <HowImproved />
      <Logs
        logs={logs}
        onDownloadReport={onDownloadReport}
        onDownloadLogs={onDownloadLogs}
        onDownloadJson={onDownloadJson}
        onDownloadCsv={onDownloadCsv}
      />
      <Footer />
    </div>
  );
}

void CITIES;

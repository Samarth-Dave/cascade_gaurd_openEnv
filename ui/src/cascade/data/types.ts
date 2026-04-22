export type Sector = 'power' | 'water' | 'hospital' | 'telecom' | 'storage' | 'grid';
export type NodeStatus = 'healthy' | 'degraded' | 'critical' | 'repair' | 'isolated' | 'operational';

export interface CGNode {
  node_id: string;
  sector: Sector;
  health: number;       // 0..1
  load: number;         // 0..1
  is_operational: boolean;
  is_hardened: boolean;
  in_recovery?: boolean;
  is_critical: boolean;
  lat?: number;
  lon?: number;
  real_name?: string;
  observation_delayed?: boolean;
  service_radius_km?: number;
  observation_confidence?: number;
  coverage_quality?: number;
}
export interface CGEdge {
  source_id: string;
  target_id: string;
  active: boolean;
  kind?: 'base' | 'cascade' | 'reroute' | 'isolate';
}
export interface CGObservation {
  nodes: CGNode[];
  edges: CGEdge[];
  sector_summary: { power:number; water:number; hospital:number; telecom:number };
  active_failures: string[];
  pending_recoveries: string[];
  budget_remaining: number;
  step: number;
  max_steps: number;
  diagnostics: {
    critical_failures: string[];
    at_risk_nodes: string[];
    dependency_alerts: string[];
    recommended_recovery_order: string[];
    system_pressure: number;
  };
  reward: number;
  done: boolean;
  task_id?: string;
  info?: string;
  radius_coverage_events?: string[];
  narrative_phase?: string;
  consecutive_waits?: number;
  upcoming_stress_level?: string;
  metadata?: Record<string, unknown>;
}

export interface AgentTurn {
  think: string;
  action: string;
}

export interface EventLine {
  ts: string;
  level: 'info' | 'warn' | 'crit' | 'ok';
  msg: string;
}

export interface TimelineEntry {
  step: number;
  title: string;
  description: string;
  tags: string[];
  tone: 'info' | 'warn' | 'crit' | 'ok';
}

export interface ScriptedStep {
  obs: CGObservation;
  attacker: AgentTurn;
  defender: AgentTurn;
  events: EventLine[];
  timeline: TimelineEntry;
  attackPressure: number; // 0..1
  defenseScore: number;   // 0..1
  whatsHappening: string;
}

export const statusFromNode = (n: CGNode, pendingRecoveries: string[]): NodeStatus => {
  if (n.in_recovery || pendingRecoveries.includes(n.node_id)) return 'repair';
  if (!n.is_operational) return 'critical';
  if (n.health < 0.35) return 'degraded';
  if (n.is_hardened) return 'operational';
  return 'healthy';
};

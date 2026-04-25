export type Task =
  | "task_easy"
  | "task_medium"
  | "task_hard"
  | "task_blackout"
  | "task_cyberattack";

export type NodeStatus = "healthy" | "degraded" | "critical" | "isolated";
export type EpisodeOutcome = "CONTAINED" | "FAILED" | "IN_PROGRESS";
export type WsEventType =
  | "step_result"
  | "failure_event"
  | "cascade_trigger"
  | "repair_complete"
  | "episode_done";
export type WsLevel = "INFO" | "WARN" | "CRIT" | "OK";

export interface SectorHealth {
  power: number;
  water: number;
  hospital: number;
  telecom: number;
}

export interface InfraNode {
  id: string;
  label: string;
  sector: string;
  health: number;
  health_pct: number;
  status: NodeStatus;
  is_critical: boolean;
  is_operational: boolean;
  is_hardened: boolean;
  lat: number;
  lng: number;
  service_radius_km: number;
  metadata: Record<string, unknown>;
}

export interface InfraEdge {
  id: string;
  source: string;
  target: string;
  kind: "dependency" | "cascade" | "recovery";
  status: "active" | "inactive";
}

export interface GraderBreakdown {
  hospital_health: number;
  cascade_penalty: number;
  budget_efficiency: number;
  infrastructure_stability: number;
  centrality_protection: number;
}

export interface EpisodeState {
  session_id: string;
  task: Task;
  step: number;
  max_steps: number;
  budget: number;
  episode_score: number;
  cascade_depth: number;
  done: boolean;
  sector_health: SectorHealth;
  nodes: InfraNode[];
  edges: InfraEdge[];
}

export interface EpisodeResetResponse {
  session_id: string;
  state: EpisodeState;
  nodes: InfraNode[];
  edges: InfraEdge[];
  sector_health: SectorHealth;
  budget: number;
  max_steps: number;
  task: Task;
}

export interface AgentAction {
  action: string;
  target: string | null;
}

export interface StepResult {
  new_state: EpisodeState;
  reward_step: number;
  episode_score: number;
  done: boolean;
  grader_breakdown: GraderBreakdown;
  nodes: InfraNode[];
  edges: InfraEdge[];
  agent_thinking: string | null;
  agent_action: AgentAction | null;
}

export interface TrajectoryItem {
  step: number;
  action: string;
  target: string | null;
  reward_step: number;
  score_after: number;
  cascade_depth: number;
  thinking: string | null;
  created_at: string;
}

export interface EpisodeHistoryItem {
  session_id: string;
  task: Task;
  score: number;
  budget_used: number;
  steps: number;
  cascade_peak: number;
  outcome: EpisodeOutcome;
  started_at: string;
  updated_at: string;
  trajectory: TrajectoryItem[];
}

export interface RewardCurvePoint {
  episode: number;
  grpo_score: number;
  random_baseline: number;
  zero_shot_baseline: number;
}

export interface BaselineComparison {
  random: number;
  zero_shot: number;
  grpo_trained: number;
}

export interface DatasetStats {
  total: number;
  task_distribution: Record<string, number>;
  action_distribution: Record<string, number>;
  avg_score: number;
}

export interface DatasetRow {
  id: string;
  task: string;
  seed: string;
  step: string;
  phase: string;
  system: string;
  prompt: string;
  thinking: string;
  completion: string;
  action: string;
  target: string;
  episode_score: string;
  reward_step: string;
  sector_health_power: string;
  sector_health_water: string;
  sector_health_hospital: string;
  sector_health_telecom: string;
  cascade_depth: string;
  budget_remaining: string;
  steps_taken: string;
  max_steps: string;
}

export interface GraphResponse {
  source: string;
  city: string;
  nodes: Array<Record<string, unknown>>;
  edges: Array<Record<string, unknown>>;
  available_cities: string[];
  session_id: string | null;
}

export interface HealthResponse {
  status: string;
  env_connected: boolean;
  model_loaded: boolean;
}

export interface WsEvent {
  type: WsEventType;
  level: WsLevel;
  timestamp: string;
  message: string;
  payload: Record<string, unknown>;
}

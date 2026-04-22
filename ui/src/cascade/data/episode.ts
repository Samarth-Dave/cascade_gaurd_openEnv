import type { CGEdge, CGNode, CGObservation, ScriptedStep, Sector } from './types';

// Base 11-node London infrastructure layout
const NODES: { id: string; sector: Sector; real: string; lat: number; lon: number }[] = [
  { id:'PWR_1', sector:'power',    real:'Battersea Substation',   lat:51.4836, lon:-0.1440 },
  { id:'PWR_2', sector:'power',    real:'Greenwich Substation',   lat:51.4934, lon:0.0098 },
  { id:'GRD_1', sector:'grid',     real:'City Grid Node 1',       lat:51.5155, lon:-0.0922 },
  { id:'GRD_2', sector:'grid',     real:'Westminster Grid',       lat:51.4994, lon:-0.1245 },
  { id:'WAT_1', sector:'water',    real:'Thames Pump Station',    lat:51.5045, lon:-0.0865 },
  { id:'WAT_2', sector:'water',    real:'Coppermills Reservoir',  lat:51.5870, lon:-0.0367 },
  { id:'HOSP_1',sector:'hospital', real:'St Thomas Hospital',     lat:51.4980, lon:-0.1186 },
  { id:'HOSP_2',sector:'hospital', real:'Royal London Hospital',  lat:51.5176, lon:-0.0594 },
  { id:'TEL_1', sector:'telecom',  real:'BT Tower Hub',           lat:51.5215, lon:-0.1389 },
  { id:'TEL_2', sector:'telecom',  real:'Canary Wharf Telecom',   lat:51.5054, lon:-0.0235 },
  { id:'BAT_1', sector:'storage',  real:'East London Battery Bank',lat:51.5390, lon:-0.0150 },
];

const BASE_EDGES: { s:string; t:string }[] = [
  { s:'PWR_1', t:'GRD_1' },
  { s:'PWR_1', t:'GRD_2' },
  { s:'PWR_2', t:'GRD_1' },
  { s:'GRD_1', t:'WAT_1' },
  { s:'GRD_1', t:'HOSP_1' },
  { s:'GRD_2', t:'HOSP_1' },
  { s:'GRD_2', t:'TEL_1' },
  { s:'WAT_1', t:'HOSP_1' },
  { s:'WAT_2', t:'WAT_1' },
  { s:'TEL_1', t:'HOSP_2' },
  { s:'TEL_2', t:'HOSP_2' },
  { s:'BAT_1', t:'GRD_1' },
  { s:'BAT_1', t:'PWR_2' },
];

// Helper to build observation snapshots quickly
type Snap = Partial<Record<string, { health?: number; op?: boolean; rec?: boolean; crit?: boolean; hard?: boolean; load?: number }>>;
function buildObs(step:number, snap:Snap, opts: {
  budget: number;
  failures: string[];
  pending: string[];
  cascadeEdges?: [string,string][];
  rerouteEdges?: [string,string][];
  isolateEdges?: [string,string][];
  sector: { power:number; water:number; hospital:number; telecom:number };
  pressure: number;
  reward: number;
  atRisk?: string[];
  recOrder?: string[];
  alerts?: string[];
  done?: boolean;
}): CGObservation {
  const nodes: CGNode[] = NODES.map(n => {
    const s = snap[n.id] ?? {};
    return {
      node_id: n.id, sector: n.sector, real_name: n.real, lat: n.lat, lon: n.lon,
      health: s.health ?? 1,
      load: s.load ?? 0.5,
      is_operational: s.op ?? true,
      is_hardened: s.hard ?? false,
      in_recovery: s.rec ?? false,
      is_critical: s.crit ?? false,
    };
  });
  const baseEdges: CGEdge[] = BASE_EDGES.map(e => ({ source_id:e.s, target_id:e.t, active:true, kind:'base' }));
  const cascade: CGEdge[] = (opts.cascadeEdges ?? []).map(([s,t]) => ({ source_id:s, target_id:t, active:true, kind:'cascade' }));
  const reroute: CGEdge[] = (opts.rerouteEdges ?? []).map(([s,t]) => ({ source_id:s, target_id:t, active:true, kind:'reroute' }));
  const isolate: CGEdge[] = (opts.isolateEdges ?? []).map(([s,t]) => ({ source_id:s, target_id:t, active:true, kind:'isolate' }));
  return {
    nodes,
    edges: [...baseEdges, ...cascade, ...reroute, ...isolate],
    sector_summary: opts.sector,
    active_failures: opts.failures,
    pending_recoveries: opts.pending,
    budget_remaining: opts.budget,
    step, max_steps: 30,
    diagnostics: {
      critical_failures: opts.failures,
      at_risk_nodes: opts.atRisk ?? [],
      dependency_alerts: opts.alerts ?? [],
      recommended_recovery_order: opts.recOrder ?? [],
      system_pressure: opts.pressure,
    },
    reward: opts.reward,
    done: opts.done ?? false,
  };
}

const ts = (s:number) => {
  const base = new Date(2025, 10, 1, 14, 32, 0);
  base.setSeconds(base.getSeconds() + s*7);
  return base.toTimeString().slice(0,8);
};

export const SCRIPTED_EPISODE: ScriptedStep[] = [
  // STEP 0 — all healthy baseline
  {
    obs: buildObs(0, {}, {
      budget: 10000, failures: [], pending: [],
      sector:{power:1,water:1,hospital:1,telecom:1}, pressure:0.05, reward:0,
    }),
    attacker: {
      think: 'Surveying network. Battersea substation shows highest load coupling to St Thomas Hospital. Probing for thermal headroom.',
      action: 'recon target=PWR_1 vector=thermal_overload',
    },
    defender: {
      think: 'All sectors nominal. Pre-positioning monitors near PWR_1 due to elevated coupling factor with HOSP_1.',
      action: 'monitor target=PWR_1 priority=elevated',
    },
    events: [
      { ts: ts(0), level:'info', msg:'Episode 042 initialized · London · task_hard' },
      { ts: ts(0), level:'info', msg:'All 11 nodes operational · 4 sectors green' },
      { ts: ts(0), level:'ok',   msg:'Defender online · GRPO policy ep42 loaded' },
    ],
    timeline: { step:0, title:'Episode initialized', description:'Baseline scan complete — all sectors healthy.', tags:['init','baseline'], tone:'info' },
    attackPressure: 0.08, defenseScore: 0.92,
    whatsHappening: 'All systems nominal. Defender is positioning monitors. No active threats.',
  },
  // STEP 1 — PWR_1 cascade trigger
  {
    obs: buildObs(1, {
      PWR_1: { health: 0.05, op:false, crit:true, load:0.95 },
      GRD_1: { health: 0.42, load:0.88 },
      GRD_2: { health: 0.55, load:0.79 },
      HOSP_1:{ health: 0.58 },
    }, {
      budget: 10000, failures:['PWR_1'], pending:[],
      cascadeEdges:[['PWR_1','GRD_1'],['PWR_1','GRD_2'],['GRD_1','HOSP_1']],
      sector:{power:0.5, water:0.92, hospital:0.61, telecom:0.94}, pressure:0.78, reward:-0.18,
      atRisk:['GRD_1','GRD_2','HOSP_1'], recOrder:['PWR_1','GRD_1','HOSP_1'],
      alerts:['Power→Grid cascade detected', 'HOSP_1 dependency at risk'],
    }),
    attacker: {
      think: 'PWR_1 collapsed under thermal load. Cascade is propagating into GRD_1 and GRD_2. St Thomas backup transfer window opens — exploit it.',
      action: 'amplify target=GRD_1 vector=load_imbalance',
    },
    defender: {
      think: 'Critical: PWR_1 down, two grid nodes degrading, hospital health 0.58. Highest expected return: dispatch recovery to PWR_1 immediately.',
      action: 'recover target=PWR_1 cost=1200',
    },
    events: [
      { ts: ts(1), level:'crit', msg:'PWR_1 (Battersea) CRITICAL FAILURE · health 0.05' },
      { ts: ts(1), level:'warn', msg:'Cascade detected · GRD_1, GRD_2 degrading' },
      { ts: ts(1), level:'warn', msg:'HOSP_1 (St Thomas) health 0.58 — at risk' },
      { ts: ts(1), level:'info', msg:'Defender dispatch: recover PWR_1' },
    ],
    timeline: { step:1, title:'PWR_1 critical failure', description:'Battersea substation collapsed. Cascade propagating into grid layer.', tags:['cascade','power','critical'], tone:'crit' },
    attackPressure: 0.62, defenseScore: 0.38,
    whatsHappening: 'PWR_1 has failed catastrophically. Cascade is propagating through GRD_1 and GRD_2 toward St Thomas Hospital.',
  },
  // STEP 2 — Recover PWR_1 initiated
  {
    obs: buildObs(2, {
      PWR_1: { health: 0.18, op:false, rec:true, load:0.40 },
      GRD_1: { health: 0.46, load:0.84 },
      GRD_2: { health: 0.58, load:0.74 },
      HOSP_1:{ health: 0.55 },
    }, {
      budget: 8800, failures:['PWR_1'], pending:['PWR_1'],
      cascadeEdges:[['PWR_1','GRD_1'],['GRD_1','HOSP_1']],
      rerouteEdges:[['BAT_1','GRD_1']],
      sector:{power:0.55, water:0.90, hospital:0.58, telecom:0.93}, pressure:0.71, reward:-0.05,
      atRisk:['HOSP_1','GRD_1'], recOrder:['HOSP_1','GRD_1'],
      alerts:['BAT_1 reroute supplying GRD_1','HOSP_1 still below threshold'],
    }),
    attacker: {
      think: 'Defender chose recovery, predictable. Pivot: degrade HOSP_1 directly via dependency starvation while PWR_1 is offline.',
      action: 'starve target=HOSP_1 vector=power_dependency',
    },
    defender: {
      think: 'Recovery on PWR_1 in progress. Reroute via BAT_1 active. HOSP_1 still degrading — schedule hospital recovery next step.',
      action: 'coordinate target=HOSP_1 cost=400',
    },
    events: [
      { ts: ts(2), level:'info', msg:'Recover PWR_1 initiated · ETA 2 steps · cost $1,200' },
      { ts: ts(2), level:'ok',   msg:'BAT_1 reroute active → GRD_1' },
      { ts: ts(2), level:'warn', msg:'HOSP_1 health 0.55 — coordinate dispatched' },
    ],
    timeline: { step:2, title:'Recover PWR_1 initiated', description:'Battery bank rerouting load. Hospital still degrading.', tags:['recovery','reroute'], tone:'warn' },
    attackPressure: 0.55, defenseScore: 0.49,
    whatsHappening: 'Defender dispatched recovery to PWR_1 and is coordinating support for HOSP_1. BAT_1 is rerouting power.',
  },
  // STEP 3 — HOSP_1 recovery started
  {
    obs: buildObs(3, {
      PWR_1: { health: 0.46, op:false, rec:true, load:0.30 },
      GRD_1: { health: 0.61, load:0.70 },
      GRD_2: { health: 0.72, load:0.60 },
      HOSP_1:{ health: 0.48, rec:true },
    }, {
      budget: 8400, failures:['PWR_1'], pending:['PWR_1','HOSP_1'],
      rerouteEdges:[['BAT_1','GRD_1'],['GRD_2','HOSP_1']],
      sector:{power:0.62, water:0.92, hospital:0.62, telecom:0.94}, pressure:0.55, reward:0.12,
      atRisk:['HOSP_1'], recOrder:['HOSP_1'],
      alerts:['HOSP_1 recovery underway','GRD_2 carrying hospital load'],
    }),
    attacker: {
      think: 'Defender is closing both gaps. Try TEL_1 disruption to fragment situational awareness.',
      action: 'jam target=TEL_1 vector=signal_flood',
    },
    defender: {
      think: 'PWR_1 climbing through 0.46. HOSP_1 recovery scheduled. Harden TEL_1 preemptively to deny the next attack vector.',
      action: 'harden target=TEL_1 cost=600',
    },
    events: [
      { ts: ts(3), level:'info', msg:'HOSP_1 recovery started · ETA 1 step' },
      { ts: ts(3), level:'ok',   msg:'PWR_1 health rising · 0.46' },
      { ts: ts(3), level:'info', msg:'TEL_1 hardened · attack-surface reduced' },
    ],
    timeline: { step:3, title:'HOSP_1 recovery started', description:'St Thomas brought into recovery. Telecom hardened against probe.', tags:['recovery','harden','hospital'], tone:'warn' },
    attackPressure: 0.42, defenseScore: 0.61,
    whatsHappening: 'Both PWR_1 and HOSP_1 are recovering. Defender hardened TEL_1 to block the attacker\'s next pivot.',
  },
  // STEP 4 — recoveries advancing
  {
    obs: buildObs(4, {
      PWR_1: { health: 0.78, op:true, rec:true, load:0.45 },
      GRD_1: { health: 0.84, load:0.55 },
      GRD_2: { health: 0.88, load:0.50 },
      HOSP_1:{ health: 0.79, rec:true },
      TEL_1: { hard:true },
    }, {
      budget: 7800, failures:[], pending:['PWR_1','HOSP_1'],
      rerouteEdges:[['BAT_1','GRD_1']],
      sector:{power:0.81, water:0.95, hospital:0.81, telecom:0.97}, pressure:0.28, reward:0.41,
      atRisk:[], recOrder:[],
      alerts:['No active failures'],
    }),
    attacker: {
      think: 'Defender response was efficient. Backing off, preparing secondary vector against water layer for next window.',
      action: 'wait reason=recompute_strategy',
    },
    defender: {
      think: 'System pressure dropping fast. Hold the recovery program; do not over-spend. Continue monitoring.',
      action: 'monitor target=ALL priority=normal',
    },
    events: [
      { ts: ts(4), level:'ok', msg:'PWR_1 back online · health 0.78' },
      { ts: ts(4), level:'ok', msg:'HOSP_1 health 0.79 · recovery continuing' },
      { ts: ts(4), level:'info', msg:'Attacker idle · no new probes detected' },
    ],
    timeline: { step:4, title:'Recoveries advancing', description:'PWR_1 back online. Hospital nearing nominal.', tags:['recovery','stable'], tone:'ok' },
    attackPressure: 0.22, defenseScore: 0.78,
    whatsHappening: 'PWR_1 is back online. HOSP_1 health is climbing. Pressure is dropping rapidly.',
  },
  // STEP 5 — fully restored
  {
    obs: buildObs(5, {
      PWR_1: { health: 0.96, op:true, load:0.55 },
      HOSP_1:{ health: 0.97 },
      TEL_1: { hard:true },
    }, {
      budget: 7800, failures:[], pending:[],
      sector:{power:0.97, water:0.98, hospital:0.98, telecom:0.99}, pressure:0.08, reward:0.62,
    }),
    attacker: { think:'Window closed. Re-evaluating.', action:'wait reason=cooldown' },
    defender: { think:'All sectors restored. Maintain hardened posture, retain budget for next wave.', action:'monitor target=ALL priority=baseline' },
    events: [
      { ts: ts(5), level:'ok', msg:'PWR_1 fully repaired · health 0.96' },
      { ts: ts(5), level:'ok', msg:'All sectors restored · no active failures' },
      { ts: ts(5), level:'info', msg:'Budget remaining $7,800 / $10,000' },
    ],
    timeline: { step:5, title:'All sectors restored', description:'Episode stabilized — defender retained 78% of budget.', tags:['restored','stable'], tone:'ok' },
    attackPressure: 0.10, defenseScore: 0.88,
    whatsHappening: 'All sectors restored. Defender retained $7,800 of budget and hardened TEL_1.',
  },
  // STEP 6-10 monitoring with mild perturbations
  ...([6,7,8,9,10].map((step, idx): ScriptedStep => {
    const reward = 0.62 + (idx+1)*0.04;
    return {
      obs: buildObs(step, {
        TEL_1: { hard:true },
        WAT_1: { health: 0.92 - idx*0.01 },
      }, {
        budget: 7800, failures:[], pending:[],
        sector:{power:0.97, water:0.96 - idx*0.01, hospital:0.98, telecom:0.99}, pressure:0.06 + idx*0.01, reward,
      }),
      attacker: {
        think: idx<2 ? 'Probing water sector for follow-up vector.' : 'Defender posture remains tight. No exploitable pressure.',
        action: idx<2 ? 'recon target=WAT_1 vector=flow_anomaly' : 'wait reason=no_opening',
      },
      defender: {
        think: idx===1 ? 'Minor flow anomaly on WAT_1. Coordinate to confirm — cheap action.' : 'System nominal. Conserve budget.',
        action: idx===1 ? 'coordinate target=WAT_1 cost=400' : 'monitor target=ALL priority=baseline',
      },
      events: idx===1 ? [
        { ts: ts(step), level:'warn', msg:'WAT_1 flow anomaly detected · 0.91' },
        { ts: ts(step), level:'info', msg:'Coordinate dispatched to WAT_1' },
      ] : [
        { ts: ts(step), level:'info', msg:`Step ${step} · monitoring · pressure low` },
      ],
      timeline: {
        step,
        title: idx===1 ? 'Water anomaly contained' : `Step ${step} · monitoring`,
        description: idx===1 ? 'WAT_1 minor anomaly resolved with low-cost coordinate.' : 'Sectors holding nominal values.',
        tags: idx===1 ? ['water','contain'] : ['monitor','stable'],
        tone: idx===1 ? 'warn' : 'ok',
      },
      attackPressure: Math.max(0.05, 0.18 - idx*0.03),
      defenseScore: Math.min(0.96, 0.88 + idx*0.015),
      whatsHappening: idx===1 ? 'A minor water flow anomaly is being contained.' : 'Episode in monitoring phase. All sectors healthy.',
    };
  })),
];

export const TOTAL_STEPS = SCRIPTED_EPISODE.length; // 11

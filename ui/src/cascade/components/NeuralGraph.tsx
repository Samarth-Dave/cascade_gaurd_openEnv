import { useEffect, useRef } from 'react';
import type { Sector } from '../data/types';
import { SECTOR_PATHS } from './icons';

interface Props {
  cascadeDepth: number;
  episodeScore: number;
  nodesAtRisk: number;
  criticalNodeIds: string[];
  cascadeEdgeKeys: string[]; // 'A->B'
}

const ZONES = [
  { x:60,  y:90,  w:140, h:330, color:'hsl(var(--amber))', label:'POWER' },
  { x:220, y:90,  w:200, h:430, color:'hsl(var(--accent))', label:'GRID' },
  { x:440, y:90,  w:180, h:430, color:'hsl(var(--green))', label:'WATER' },
  { x:640, y:90,  w:160, h:200, color:'hsl(var(--purple))', label:'HOSPITAL' },
  { x:640, y:300, w:160, h:160, color:'hsl(var(--purple))', label:'TELECOM' },
  { x:820, y:140, w:120, h:300, color:'hsl(0 0% 35%)', label:'STORAGE' },
];

type GraphNode = { id: string; x: number; y: number; sector: Sector; name: string; status: string };
const NODES: GraphNode[] = [
  { id:'GEN_SUBSTATION_A', x:130, y:160, sector:'power',    name:'GEN_SUBSTATION_A', status:'operational' },
  { id:'GEN_SUBSTATION_B', x:130, y:340, sector:'power',    name:'GEN_SUBSTATION_B', status:'operational' },
  { id:'SUBST_BACKUP',     x:130, y:250, sector:'storage',  name:'SUBST_BACKUP',     status:'standby' },
  { id:'GRID_N1',          x:320, y:160, sector:'grid',     name:'GRID_N1',          status:'nominal' },
  { id:'GRID_N2',          x:320, y:300, sector:'grid',     name:'GRID_N2',          status:'nominal' },
  { id:'GRID_N3',          x:320, y:440, sector:'grid',     name:'GRID_N3',          status:'nominal' },
  { id:'PUMP_STATION',     x:530, y:160, sector:'water',    name:'PUMP_STATION',     status:'nominal' },
  { id:'RESERVOIR',        x:530, y:340, sector:'water',    name:'RESERVOIR',        status:'nominal' },
  { id:'WATER_TREAT',      x:530, y:440, sector:'water',    name:'WATER_TREAT',      status:'nominal' },
  { id:'HOSP_CENTRAL',     x:720, y:170, sector:'hospital', name:'HOSP_CENTRAL',     status:'critical-load' },
  { id:'TELECOM_HUB',      x:720, y:370, sector:'telecom',  name:'TELECOM_HUB',      status:'nominal' },
  { id:'BATT_BANK',        x:880, y:220, sector:'storage',  name:'BATT_BANK',        status:'discharging' },
  { id:'HOSP_BACKUP',      x:880, y:380, sector:'hospital', name:'HOSP_BACKUP',      status:'standby' },
];

// Static topology only — edge "kind" is derived per-step from live state.
const EDGES: [string, string][] = [
  ['GEN_SUBSTATION_A','GRID_N1'],
  ['GEN_SUBSTATION_B','GRID_N3'],
  ['SUBST_BACKUP','GRID_N2'],
  ['GRID_N1','PUMP_STATION'],
  ['GRID_N2','RESERVOIR'],
  ['GRID_N3','WATER_TREAT'],
  ['PUMP_STATION','HOSP_CENTRAL'],
  ['RESERVOIR','HOSP_CENTRAL'],
  ['HOSP_CENTRAL','TELECOM_HUB'],
  ['BATT_BANK','GRID_N1'],
  ['HOSP_BACKUP','HOSP_CENTRAL'],
];

// Backup/reroute edges only become active when their downstream target is critical.
const REROUTE_EDGES = new Set(['SUBST_BACKUP->GRID_N2','BATT_BANK->GRID_N1','HOSP_BACKUP->HOSP_CENTRAL']);

const NODE_W = 122, NODE_H = 56;

export function NeuralGraph({ cascadeDepth, episodeScore, nodesAtRisk, criticalNodeIds, cascadeEdgeKeys }: Props) {
  const ref = useRef<SVGSVGElement | null>(null);
  const offsetRef = useRef(0);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const svg = ref.current;
      if (svg) {
        const flowing = svg.querySelectorAll<SVGLineElement>('line[data-flow="1"]');
        if (flowing.length) {
          offsetRef.current = (offsetRef.current - 1) % 200;
          flowing.forEach(l => l.setAttribute('stroke-dashoffset', String(offsetRef.current)));
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const nodeMap = Object.fromEntries(NODES.map(n => [n.id, n]));

  const isCascadeEdge = (a:string, b:string) =>
    cascadeEdgeKeys.includes(`${a}->${b}`) || cascadeEdgeKeys.includes(`${b}->${a}`);

  // A reroute edge is "live" only when its downstream/peer node is currently critical.
  const isActiveReroute = (a:string, b:string) =>
    REROUTE_EDGES.has(`${a}->${b}`) &&
    (criticalNodeIds.includes(a) || criticalNodeIds.includes(b));

  return (
    <section id="graph" className="relative px-5 py-20">
      <div className="max-w-[1280px] mx-auto cg-reveal">
        <div className="flex items-end justify-between mb-6 flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/55 mb-2">/ neural topology</div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Critical Infrastructure Graph</h2>
          </div>
          <div className="text-sm text-foreground/55 max-w-md">A reasoning view of sector dependencies. Red edges show the live cascade path; teal edges show defender reroutes.</div>
        </div>
        <div className="rounded-2xl bg-card border border-foreground/10 cg-shadow-card p-4 sm:p-6 overflow-hidden">
          <div className="rounded-xl bg-[#0d0f12] p-3">
            <svg ref={ref} viewBox="0 0 960 500" className="w-full h-auto" role="img" aria-label="Neural topology of city infrastructure">
              <defs>
                <marker id="cg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M0 0L10 5L0 10z" fill="hsl(var(--red))" />
                </marker>
                <filter id="cg-glow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="2.5" result="b" />
                  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </defs>

              {/* zones */}
              {ZONES.map(z => (
                <g key={z.label}>
                  <rect x={z.x} y={z.y} width={z.w} height={z.h} rx="14" fill={z.color} fillOpacity="0.06" stroke={z.color} strokeOpacity="0.35" strokeDasharray="4 6" />
                  <text x={z.x + 10} y={z.y + 18} fill={z.color} opacity="0.85" fontSize="10" fontFamily="DM Mono" letterSpacing="2">{z.label}</text>
                </g>
              ))}

              {/* edges — kind derived from live state each render */}
              {EDGES.map(([a,b]) => {
                const A = nodeMap[a], B = nodeMap[b];
                const cascade = isCascadeEdge(a,b);
                const reroute = !cascade && isActiveReroute(a,b);
                const stroke = cascade ? 'hsl(var(--red))'
                  : reroute ? 'hsl(var(--accent-2))'
                  : 'hsl(0 0% 100% / 0.28)';
                const flow = cascade || reroute;
                return (
                  <line key={`${a}-${b}`} x1={A.x} y1={A.y} x2={B.x} y2={B.y}
                        stroke={stroke} strokeWidth={cascade ? 2.4 : reroute ? 1.8 : 1.2}
                        strokeDasharray={flow ? '8 6' : '4 6'}
                        data-flow={flow ? '1' : '0'}
                        filter={cascade ? 'url(#cg-glow)' : undefined}
                        markerEnd={cascade ? 'url(#cg-arrow)' : undefined} />
                );
              })}

              {/* nodes */}
              {NODES.map(n => {
                const isCrit = criticalNodeIds.includes(n.id);
                return (
                  <g key={n.id} transform={`translate(${n.x - NODE_W/2} ${n.y - NODE_H/2})`}>
                    {isCrit && <circle cx={NODE_W/2} cy={NODE_H/2} r={45} fill="none" stroke="hsl(var(--red))" strokeOpacity="0.55">
                      <animate attributeName="r" values="34;52;34" dur="1.6s" repeatCount="indefinite" />
                      <animate attributeName="opacity" values="0.6;0;0.6" dur="1.6s" repeatCount="indefinite" />
                    </circle>}
                    <rect width={NODE_W} height={NODE_H} rx="10" fill="hsl(0 0% 100% / 0.06)" stroke={isCrit ? 'hsl(var(--red))' : 'hsl(0 0% 100% / 0.18)'} />
                    <svg x={10} y={18} width={20} height={20} viewBox="0 0 24 24" fill="none" stroke={isCrit ? 'hsl(var(--red))' : '#ffffff'} strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" dangerouslySetInnerHTML={{ __html: SECTOR_PATHS[n.sector] }} />
                    <text x="36" y="22" fill="#fff" fontSize="10" fontFamily="DM Mono" letterSpacing="0.5">{n.name}</text>
                    <text x="36" y="40" fill={isCrit ? 'hsl(var(--red))' : 'hsl(0 0% 100% / 0.55)'} fontSize="9" fontFamily="DM Mono">{isCrit ? 'CRITICAL' : n.status}</text>
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-4 text-[11px] font-mono text-foreground/65">
            <span className="inline-flex items-center gap-2"><span className="inline-block h-[2px] w-6" style={{ borderTop:'2px dashed hsl(0 0% 50%)' }}/> Base</span>
            <span className="inline-flex items-center gap-2"><span className="inline-block h-[2px] w-6 bg-[hsl(var(--red))]"/> Cascade</span>
            <span className="inline-flex items-center gap-2"><span className="inline-block h-[2px] w-6 bg-[hsl(var(--accent-2))]"/> Reroute</span>
          </div>

          <div className="mt-4 grid grid-cols-3 gap-3">
            {[
              { k:'Cascade depth', v:cascadeDepth, color:'hsl(var(--red))' },
              { k:'Episode score', v:episodeScore.toFixed(2), color:'hsl(var(--accent))' },
              { k:'Nodes at risk', v:nodesAtRisk, color:'hsl(var(--amber))' },
            ].map(s => (
              <div key={s.k} className="rounded-xl bg-foreground/[0.03] border border-foreground/10 px-4 py-3">
                <div className="font-mono text-[10px] uppercase tracking-wider text-foreground/55">{s.k}</div>
                <div className="mt-1 text-2xl font-bold" style={{ color: s.color }}>{s.v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

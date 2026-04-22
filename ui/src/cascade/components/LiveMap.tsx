import { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import type { CGEdge, CGNode } from '../data/types';
import { statusFromNode } from '../data/types';
import type { City } from '../data/cities';
import type { ConnState } from '../hooks/useEnvSocket';
import type { EventLine } from '../data/types';
import { RingGauge } from './RingGauge';
import { sectorSvg } from './icons';
import { ENV_WS } from '../config/env';

interface Props {
  city: City;
  nodes: CGNode[];
  edges: CGEdge[];
  pendingRecoveries: string[];
  sectorSummary: { power:number; water:number; hospital:number; telecom:number };
  step: number;
  maxSteps: number;
  budget: number;
  score: number;
  hasCritical: boolean;
  events: EventLine[];
  connState: ConnState;
  isPlaying: boolean;
  speed: 0.5 | 1 | 2 | 4;
  onPlayPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onSpeed: (s: 0.5|1|2|4) => void;
  onDispatch: (action: string, target: string | null) => void;
  busy: boolean;
  whatsHappening: string;
  episodeId: string;
}

const STATUS_CLASS: Record<string,string> = {
  healthy:'cg-mk-healthy', degraded:'cg-mk-degraded', critical:'cg-mk-critical',
  repair:'cg-mk-repair', isolated:'cg-mk-isolated', operational:'cg-mk-operational',
};
type ActionDef = { key: string; label: string; cost: number; hover: string; path: string };
const ACTIONS: ActionDef[] = [
  { key:'recover',    label:'Recover Node',   cost:1200, hover:'hover:bg-[hsl(var(--purple))]/10 hover:text-[hsl(var(--purple))]',
    path:'<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>' },
  { key:'harden',     label:'Harden Node',    cost:600,  hover:'hover:bg-[hsl(var(--accent))]/10 hover:text-[hsl(var(--accent))]',
    path:'<path d="M12 2 4 5v6c0 5 3.5 9 8 11 4.5-2 8-6 8-11V5z"/>' },
  { key:'shed_load',  label:'Shed Load',      cost:300,  hover:'hover:bg-[hsl(var(--amber))]/10 hover:text-[hsl(var(--amber))]',
    path:'<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>' },
  { key:'coordinate', label:'Coordinate',     cost:400,  hover:'hover:bg-[hsl(var(--green))]/10 hover:text-[hsl(var(--green))]',
    path:'<path d="M5 12a7 7 0 0 1 14 0"/><path d="M8 12a4 4 0 0 1 8 0"/><circle cx="12" cy="12" r="1.5"/><path d="M12 14v6"/>' },
  { key:'wait',       label:'Wait / Monitor', cost:0,    hover:'hover:bg-foreground/5',
    path:'<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>' },
];

const ActionIcon = ({ path, className = 'h-3.5 w-3.5' }: { path: string; className?: string }) => (
  <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth={1.9} strokeLinecap="round" strokeLinejoin="round" dangerouslySetInnerHTML={{ __html: path }} />
);

export function LiveMap(p: Props) {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapObj = useRef<L.Map | null>(null);
  const layerNodes = useRef<L.LayerGroup | null>(null);
  const layerEdges = useRef<L.LayerGroup | null>(null);

  const [target, setTarget] = useState<string | null>(null);

  // Init Leaflet
  useEffect(() => {
    if (!mapRef.current || mapObj.current) return;
    const m = L.map(mapRef.current, {
      center: p.city.center, zoom: p.city.zoom, zoomControl: true, scrollWheelZoom: false, attributionControl: true,
    });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19, attribution: '© OpenStreetMap · © CARTO',
    }).addTo(m);
    layerNodes.current = L.layerGroup().addTo(m);
    layerEdges.current = L.layerGroup().addTo(m);
    mapObj.current = m;
    return () => { m.remove(); mapObj.current = null; };
  }, []);

  // Re-center on city change
  useEffect(() => {
    mapObj.current?.setView(p.city.center, p.city.zoom, { animate: true });
  }, [p.city.key, p.city.center, p.city.zoom]);

  // Compute coordinates with fallback
  const placedNodes = useMemo(() => {
    return p.nodes.map((n, i) => {
      let lat = n.lat ?? 0, lon = n.lon ?? 0;
      if (!lat || !lon) {
        const angle = (i / Math.max(1, p.nodes.length)) * Math.PI * 2;
        const r = 0.015 + (i % 3) * 0.008;
        lat = p.city.center[0] + Math.sin(angle) * r;
        lon = p.city.center[1] + Math.cos(angle) * r;
      }
      return { ...n, lat, lon };
    });
  }, [p.nodes, p.city.center]);

  // Render nodes
  useEffect(() => {
    if (!layerNodes.current) return;
    layerNodes.current.clearLayers();
    placedNodes.forEach(n => {
      const status = statusFromNode(n, p.pendingRecoveries);
      const cls = STATUS_CLASS[status] ?? 'cg-mk-healthy';
      const isCrit = status === 'critical';
      const html = `
        <div class="cg-marker ${cls}">
          ${isCrit ? '<span class="cg-ring"></span><span class="cg-ring delay"></span>' : ''}
          <span class="cg-marker-svg">${sectorSvg(n.sector, 20)}</span>
          <span class="cg-marker-label">${n.node_id}</span>
        </div>`;
      const icon = L.divIcon({ html, className: 'cg-marker-icon', iconSize: [40,40], iconAnchor: [20,20] });
      const mk = L.marker([n.lat!, n.lon!], { icon });
      mk.on('click', () => setTarget(n.node_id));
      mk.addTo(layerNodes.current!);
    });
  }, [placedNodes, p.pendingRecoveries]);

  // Render edges
  useEffect(() => {
    if (!layerEdges.current) return;
    layerEdges.current.clearLayers();
    const byId = new Map(placedNodes.map(n => [n.node_id, n]));
    p.edges.forEach(e => {
      const a = byId.get(e.source_id), b = byId.get(e.target_id);
      if (!a || !b) return;
      const kind = e.kind ?? 'base';
      const styles: Record<string, L.PolylineOptions> = {
        base:    { color: '#ffffff', weight: 1, opacity: 0.45, dashArray: '4,6' },
        cascade: { color: '#e8192c', weight: 2.5, opacity: 0.95, dashArray: '8,6' },
        reroute: { color: '#00a878', weight: 2.2, opacity: 0.95, dashArray: '6,6' },
        isolate: { color: '#1446f5', weight: 2, opacity: 0.9, dashArray: '2,8' },
      };
      L.polyline([[a.lat!, a.lon!],[b.lat!, b.lon!]], styles[kind]).addTo(layerEdges.current!);
    });
  }, [p.edges, placedNodes]);

  const conn = p.connState;
  const connBanner =
    conn === 'live' ? { dot:'bg-[hsl(var(--green))]', text:`Connected to OpenEnv · ${ENV_WS}`, tag:'LIVE',  bg:'bg-[hsl(var(--green))]/10 border-[hsl(var(--green))]/25 text-[hsl(var(--green))]' }
  : conn === 'sim'  ? { dot:'bg-[hsl(var(--accent))]', text:'Running scripted simulation — start the OpenEnv server to enable live mode.', tag:'SIM',  bg:'bg-[hsl(var(--accent))]/10 border-[hsl(var(--accent))]/25 text-[hsl(var(--accent))]' }
  : conn === 'error'? { dot:'bg-[hsl(var(--red))]',    text:'Connection error — fell back to simulation.',  tag:'ERR',  bg:'bg-[hsl(var(--red))]/10 border-[hsl(var(--red))]/25 text-[hsl(var(--red))]' }
  :                   { dot:'bg-foreground/40',         text:`Connecting to OpenEnv at ${ENV_WS}…`, tag:'…', bg:'bg-foreground/[0.05] border-foreground/15 text-foreground/65' };

  const speeds: (0.5|1|2|4)[] = [0.5,1,2,4];

  return (
    <section id="map" className="relative px-5 py-16">
      <div className="max-w-[1400px] mx-auto cg-reveal">
        <div className="flex items-end justify-between mb-5 flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/55 mb-2">/ live map</div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Cascade Containment in Real Time</h2>
          </div>
          <div className="text-sm text-foreground/55 max-w-md">Click any node on the map to target it, then dispatch an action from the panel.</div>
        </div>

        <div className="rounded-2xl bg-card border border-foreground/10 cg-shadow-card overflow-hidden">
          {/* Header bar */}
          <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-foreground/10 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="font-mono text-[11px] tracking-wider text-foreground/65">{p.episodeId} · {p.city.name} · {p.city.taskId}</span>
              {p.hasCritical && (
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-[hsl(var(--red))]/15 text-[hsl(var(--red))] text-[10px] font-bold font-mono cg-red-pulse">
                  <span className="h-1.5 w-1.5 rounded-full bg-[hsl(var(--red))]" /> CRITICAL
                </span>
              )}
            </div>
            <div className="flex items-center gap-4 font-mono text-[11px] text-foreground/65">
              <span>STEP <span className="text-foreground font-semibold">{String(p.step).padStart(3,'0')}/{String(p.maxSteps).padStart(3,'0')}</span></span>
              <span>SCORE <span className="text-foreground font-semibold">{p.score.toFixed(2)}</span></span>
              <span>BUDGET <span className="text-foreground font-semibold">${p.budget.toLocaleString()}</span></span>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2 px-4 py-2.5 border-b border-foreground/10 flex-wrap">
            <button onClick={p.onReset} className="h-8 px-3 rounded-md border border-foreground/15 hover:bg-foreground/5 text-xs font-medium inline-flex items-center gap-1.5">
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>
              Reset
            </button>
            <button onClick={p.onPlayPause} className="h-8 px-3 rounded-md bg-foreground text-background hover:bg-[hsl(var(--accent))] text-xs font-semibold inline-flex items-center gap-1.5">
              {p.isPlaying
                ? <><svg viewBox="0 0 24 24" className="h-3 w-3" fill="currentColor"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg> Pause</>
                : <><svg viewBox="0 0 24 24" className="h-3 w-3" fill="currentColor"><path d="M6 4l14 8-14 8z"/></svg> Play</>}
            </button>
            <button onClick={p.onStep} disabled={p.busy} className="h-8 px-3 rounded-md border border-foreground/15 hover:bg-foreground/5 text-xs font-medium inline-flex items-center gap-1.5 disabled:opacity-50">
              Step ›
            </button>
            <div className="ml-1 flex items-center gap-1 rounded-md border border-foreground/15 p-0.5">
              {speeds.map(s => (
                <button key={s} onClick={() => p.onSpeed(s)} className={`h-6 px-2 rounded text-[11px] font-mono ${p.speed===s ? 'bg-foreground text-background' : 'text-foreground/60 hover:text-foreground'}`}>{s}×</button>
              ))}
            </div>
            <span className="ml-auto font-mono text-[10px] uppercase tracking-wider text-foreground/55">
              {conn==='live' ? 'LIVE ENV' : 'SIMULATION'}
            </span>
          </div>

          {/* Step progress bar */}
          <div className="px-4 py-2 border-b border-foreground/10">
            <div className="h-1 rounded-full bg-foreground/[0.08] overflow-hidden">
              <div className="h-full bg-[hsl(var(--accent))] transition-all duration-500" style={{ width: `${(p.step / p.maxSteps) * 100}%` }} />
            </div>
            <div className="mt-1 flex justify-between font-mono text-[10px] text-foreground/55">
              <span>Step {p.step} of {p.maxSteps}</span>
              <span>{p.episodeId} · {p.city.name}</span>
            </div>
          </div>

          {/* Connection banner */}
          <div className={`flex items-center gap-2 px-4 py-2 border-b ${connBanner.bg}`}>
            <span className={`h-2 w-2 rounded-full ${connBanner.dot}`} />
            <span className="text-[12px]">{connBanner.text}</span>
            <span className="ml-auto font-mono text-[10px] font-bold tracking-wider">{connBanner.tag}</span>
          </div>

          {/* Map */}
          <div className="relative">
            <div ref={mapRef} className="w-full" style={{ height: 540 }} />
            {/* Action Dispatch */}
            <div className="hidden md:block absolute top-3 left-3 z-[400] w-[224px] rounded-xl bg-card border border-foreground/15 cg-shadow-card p-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="relative inline-flex h-2 w-2 rounded-full bg-[hsl(var(--green))] cg-live-pulse" />
                <span className="text-xs font-bold">Dispatch Action</span>
              </div>
              <label className="block text-[10px] font-mono uppercase tracking-wider text-foreground/55 mb-1">Target Node</label>
              <select value={target ?? ''} onChange={(e) => setTarget(e.target.value || null)}
                      className="w-full h-8 px-2 rounded-md border border-foreground/15 bg-background text-xs font-mono">
                <option value="">— click a node or pick —</option>
                {p.nodes.map(n => <option key={n.node_id} value={n.node_id}>{n.node_id} · {n.sector}</option>)}
              </select>
              <div className="mt-2 space-y-1">
                {ACTIONS.map(a => (
                  <button key={a.key}
                    disabled={p.busy}
                    onClick={() => p.onDispatch(a.key, a.key==='wait' ? null : target)}
                    className={`w-full flex items-center justify-between gap-2 h-8 px-2 rounded-md text-[12px] font-medium transition-colors disabled:opacity-50 ${a.hover}`}>
                    <span className="inline-flex items-center gap-1.5"><ActionIcon path={a.path} /><span>{a.label}</span></span>
                    <span className="font-mono text-[10px] text-foreground/50">${a.cost}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Sector Health Sidebar */}
            <div className="hidden md:block absolute top-3 right-3 z-[400] w-[220px] rounded-xl bg-card border border-foreground/15 cg-shadow-card p-3">
              <div className="text-xs font-bold mb-2">Sector Health</div>
              <div className="space-y-2">
                {([
                  ['Power',    p.sectorSummary.power],
                  ['Water',    p.sectorSummary.water],
                  ['Hospital', p.sectorSummary.hospital],
                  ['Telecom',  p.sectorSummary.telecom],
                ] as const).map(([k,v]) => (
                  <div key={k} className="flex items-center justify-between gap-2">
                    <span className="text-xs text-foreground/70">{k}</span>
                    <RingGauge value={v} size={28} />
                  </div>
                ))}
              </div>
            </div>

            {/* Event Feed */}
            <div className="hidden md:block absolute bottom-3 left-3 z-[400] w-[290px] rounded-xl border border-white/10 cg-shadow-card overflow-hidden" style={{ background:'rgba(13,13,13,0.92)' }}>
              <div className="flex items-center gap-2 px-3 py-2 border-b border-white/5">
                <span className="relative inline-flex h-2 w-2 rounded-full bg-[hsl(var(--green))] cg-live-pulse" />
                <span className="text-[11px] font-bold text-white">Live Event Feed</span>
              </div>
              <div className="cg-scroll cg-scroll-dark max-h-[200px] overflow-y-auto px-3 py-2 font-mono text-[11px] text-white/85 space-y-1">
                {p.events.length === 0 && <div className="text-white/40">// no events yet</div>}
                {p.events.map((e, i) => (
                  <div key={i} className="cg-slide-in">
                    <span className="text-white/35">{e.ts}</span>{' '}
                    <span className={
                      e.level==='ok' ? 'text-[hsl(var(--green))]' :
                      e.level==='warn' ? 'text-[hsl(var(--amber))]' :
                      e.level==='crit' ? 'text-[hsl(var(--red))]' :
                      'text-white/85'
                    }>{e.msg}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="px-4 py-3 border-t border-foreground/10 bg-foreground/[0.02] flex flex-wrap gap-x-6 gap-y-2 text-[11px] font-mono text-foreground/65">
            <div className="flex items-center gap-3">
              <span className="opacity-60">Lines:</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-6 border-t border-dashed border-foreground/40"/>Base</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-6 h-[2px] bg-[hsl(var(--red))]"/>⚡ Cascade</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-6 h-[2px] bg-[hsl(var(--accent-2))]"/>↩ Reroute</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-6 h-[2px] bg-[hsl(var(--accent))]"/>✂ Isolate</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="opacity-60">Nodes:</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded bg-[hsl(var(--green))]"/>Healthy</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded bg-[hsl(var(--amber))]"/>Degraded</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded bg-[hsl(var(--red))]"/>Critical (pulses)</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded bg-[hsl(0_0%_35%)]"/>Isolated</span>
              <span className="inline-flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded bg-[hsl(var(--accent))]"/>Repairing</span>
            </div>
          </div>

          {/* What's Happening */}
          <div className="px-4 py-3 flex items-start gap-3 border-t border-foreground/10">
            <div className="h-7 w-7 rounded-full bg-[hsl(var(--accent))]/15 text-[hsl(var(--accent))] grid place-items-center mt-0.5">
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v4M12 18v4M2 12h4M18 12h4M5 5l3 3M16 16l3 3M19 5l-3 3M8 16l-3 3"/></svg>
            </div>
            <div>
              <div className="font-mono text-[10px] uppercase tracking-wider text-foreground/55">What&apos;s happening</div>
              <div className="text-sm text-foreground/85">{p.whatsHappening}</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

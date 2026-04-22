import { useEffect, useRef, useState } from 'react';
import { TRAINING_CURVE } from '../data/training';
import { RingGauge } from './RingGauge';

interface Props {
  episodeScore: number;
  budgetRemaining: number;
  nodesAtRisk: number;
  stepsTaken: number;
  sectorHealth: { power:number; water:number; hospital:number; telecom:number };
}

function CountUp({ value, decimals = 0, prefix = '', suffix = '' }: { value:number; decimals?:number; prefix?:string; suffix?:string }) {
  const [v, setV] = useState(0);
  const ref = useRef<HTMLSpanElement | null>(null);
  const seenRef = useRef(false);
  useEffect(() => {
    if (!ref.current) return;
    const io = new IntersectionObserver((es) => {
      es.forEach(e => { if (e.isIntersecting) seenRef.current = true; });
    }, { threshold: 0.4 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, []);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const from = v;
    const to = value;
    const dur = 700;
    const tick = (t:number) => {
      const k = Math.min(1, (t - start)/dur);
      const eased = 1 - Math.pow(1-k, 3);
      setV(from + (to - from) * eased);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return <span ref={ref}>{prefix}{v.toFixed(decimals)}{suffix}</span>;
}

export function Analytics({ episodeScore, budgetRemaining, nodesAtRisk, stepsTaken, sectorHealth }: Props) {
  const W = 860, H = 190;
  const pad = { l: 30, r: 16, t: 14, b: 22 };
  const N = TRAINING_CURVE.length;
  const xAt = (i:number) => pad.l + (W - pad.l - pad.r) * (i / (N-1));
  const yAt = (v:number) => pad.t + (H - pad.t - pad.b) * (1 - Math.max(0, Math.min(1, v)));
  const path = TRAINING_CURVE.map((v,i) => `${i===0?'M':'L'}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ');
  const area = `${path} L${xAt(N-1).toFixed(1)},${yAt(0).toFixed(1)} L${xAt(0).toFixed(1)},${yAt(0).toFixed(1)} Z`;

  const stats = [
    { k:'Episode score',   node:<CountUp value={episodeScore} decimals={2} />, color:'hsl(var(--accent))' },
    { k:'Budget remaining',node:<CountUp value={budgetRemaining} prefix="$" />, color:'hsl(var(--accent-2))' },
    { k:'Nodes at risk',   node:<CountUp value={nodesAtRisk} />,                color:'hsl(var(--red))' },
    { k:'Steps taken',     node:<CountUp value={stepsTaken} suffix=" / 30" />,  color:'hsl(var(--amber))' },
  ];

  const sectors = [
    { k:'Power',    v:sectorHealth.power,    color:'hsl(var(--amber))' },
    { k:'Water',    v:sectorHealth.water,    color:'hsl(var(--green))' },
    { k:'Hospitals',v:sectorHealth.hospital, color:'hsl(var(--purple))' },
    { k:'Telecom',  v:sectorHealth.telecom,  color:'hsl(var(--accent))' },
  ];

  return (
    <section id="analytics" className="relative px-5 py-20">
      <div className="max-w-[1280px] mx-auto cg-reveal">
        <div className="flex items-end justify-between mb-6 flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/55 mb-2">/ analytics</div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Episode Telemetry</h2>
          </div>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          {stats.map(s => (
            <div key={s.k} className="rounded-2xl bg-card border border-foreground/10 cg-shadow-card p-4 border-t-4" style={{ borderTopColor: s.color }}>
              <div className="font-mono text-[10px] uppercase tracking-wider text-foreground/55">{s.k}</div>
              <div className="mt-2 text-3xl font-bold tracking-tight" style={{ color: s.color }}>{s.node}</div>
            </div>
          ))}
        </div>

        <div className="rounded-2xl bg-card border border-foreground/10 cg-shadow-card p-5 mb-6">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-wider text-foreground/55">GRPO training curve</div>
              <div className="font-semibold text-base">Episodes 1 → 50</div>
            </div>
            <div className="font-mono text-[11px] text-foreground/55">peak {Math.max(...TRAINING_CURVE).toFixed(2)} · final {TRAINING_CURVE[N-1].toFixed(2)}</div>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
            <defs>
              <linearGradient id="cg-train-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(var(--accent))" stopOpacity="0.35" />
                <stop offset="100%" stopColor="hsl(var(--accent))" stopOpacity="0" />
              </linearGradient>
            </defs>
            {[0, 0.25, 0.5, 0.75, 1].map(t => (
              <line key={t} x1={pad.l} x2={W-pad.r} y1={yAt(t)} y2={yAt(t)} stroke="hsl(var(--border) / 0.12)" />
            ))}
            <line x1={pad.l} x2={W-pad.r} y1={yAt(0.23)} y2={yAt(0.23)} stroke="hsl(var(--red))" strokeOpacity="0.6" strokeDasharray="4 4" />
            <text x={pad.l} y={yAt(0.23)-4} fontFamily="DM Mono" fontSize="9" fill="hsl(var(--red))" opacity="0.7">baseline 0.23</text>
            <path d={area} fill="url(#cg-train-fill)" />
            <path d={path} fill="none" stroke="hsl(var(--accent))" strokeWidth={2} />
            <circle cx={xAt(N-1)} cy={yAt(TRAINING_CURVE[N-1])} r={3.5} fill="hsl(var(--accent))" />
            <text x={xAt(N-1)-6} y={yAt(TRAINING_CURVE[N-1])-8} textAnchor="end" fontFamily="DM Mono" fontSize="10" fill="hsl(var(--ink))">ep50 · {TRAINING_CURVE[N-1].toFixed(2)}</text>
            <text x={xAt(0)} y={H-6} fontFamily="DM Mono" fontSize="9" fill="hsl(var(--ink-2))">ep1</text>
            <text x={xAt(N-1)} y={H-6} textAnchor="end" fontFamily="DM Mono" fontSize="9" fill="hsl(var(--ink-2))">ep50</text>
          </svg>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {sectors.map(s => {
            const status = s.v >= 0.7 ? 'healthy' : s.v >= 0.4 ? 'degraded' : 'critical';
            return (
              <div key={s.k} className="rounded-2xl bg-card border border-foreground/10 cg-shadow-card p-4 flex items-center gap-4">
                <RingGauge value={s.v} size={72} showPct={false} />
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-foreground/55">{s.k}</div>
                  <div className="text-2xl font-bold leading-tight">{Math.round(s.v*100)}%</div>
                  <div className={`text-[11px] font-mono uppercase tracking-wider ${status==='healthy'?'text-[hsl(var(--green))]':status==='degraded'?'text-[hsl(var(--amber))]':'text-[hsl(var(--red))]'}`}>{status}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

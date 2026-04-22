import { useEffect, useRef } from 'react';
import type { EventLine } from '../data/types';

interface Props {
  terminalLines: EventLine[];
  rewardSeries: number[];
  step: number;
}

export function ThreatIntel({ terminalLines, rewardSeries, step }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const termRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = c.offsetWidth, h = c.offsetHeight;
      c.width = w * dpr; c.height = h * dpr;
      const ctx = c.getContext('2d')!;
      ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.clearRect(0,0,w,h);
      // grid
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      for (let i=0;i<=10;i++){
        const y = h * (i/10);
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
      }
      for (let i=0;i<=10;i++){
        const x = w * (i/10);
        ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();
      }
      // baseline 0.23
      const baselineY = h - h*0.23;
      ctx.setLineDash([4,4]);
      ctx.strokeStyle = 'rgba(232,25,44,0.6)';
      ctx.beginPath(); ctx.moveTo(0,baselineY); ctx.lineTo(w,baselineY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(232,25,44,0.85)';
      ctx.font = '10px DM Mono';
      ctx.fillText('baseline 0.23 (random)', 8, baselineY - 6);

      // curve
      const series = rewardSeries.length ? rewardSeries : [0];
      const N = Math.max(2, series.length);
      const padL = 6, padR = 6;
      const xAt = (i:number) => padL + (w - padL - padR) * (i / (N-1 || 1));
      const yAt = (v:number) => h - h * Math.max(0, Math.min(1, (v + 0.5) / 1.5));
      // fill area
      const grad = ctx.createLinearGradient(0,0,0,h);
      grad.addColorStop(0, 'rgba(0,168,120,0.45)');
      grad.addColorStop(1, 'rgba(0,168,120,0)');
      ctx.beginPath();
      ctx.moveTo(xAt(0), h);
      series.forEach((v,i) => ctx.lineTo(xAt(i), yAt(v)));
      ctx.lineTo(xAt(series.length-1), h);
      ctx.closePath();
      ctx.fillStyle = grad; ctx.fill();
      // line
      ctx.beginPath();
      series.forEach((v,i) => i===0 ? ctx.moveTo(xAt(i), yAt(v)) : ctx.lineTo(xAt(i), yAt(v)));
      ctx.strokeStyle = 'hsl(161 100% 45%)';
      ctx.lineWidth = 2;
      ctx.stroke();
      // last point
      const lv = series[series.length-1];
      ctx.beginPath(); ctx.arc(xAt(series.length-1), yAt(lv), 3.2, 0, Math.PI*2);
      ctx.fillStyle = '#fff'; ctx.fill();
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    return () => ro.disconnect();
  }, [rewardSeries]);

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight;
  }, [terminalLines]);

  const current = rewardSeries[rewardSeries.length-1] ?? 0;
  const peak = rewardSeries.reduce((m,v) => Math.max(m,v), 0);

  return (
    <section id="intel" className="relative px-5 py-20">
      <div className="max-w-[1280px] mx-auto cg-reveal">
        <div className="flex items-end justify-between mb-6 flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/55 mb-2">/ threat intelligence</div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Live Reasoning · Live Reward</h2>
          </div>
          <div className="text-sm text-foreground/55 max-w-md">Adversary command stream on the left; reward signal vs the random-policy baseline on the right.</div>
        </div>
        <div className="grid lg:grid-cols-2 gap-5">
          {/* Terminal */}
          <div className="rounded-2xl border border-[hsl(var(--red))]/25 overflow-hidden cg-shadow-card" style={{ background:'#080808' }}>
            <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5">
              <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
              <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
              <span className="h-3 w-3 rounded-full bg-[#28c840]" />
              <span className="ml-3 font-mono text-[11px] text-white/55">attacker — ssh adversary@cg-net</span>
              <span className="ml-auto font-mono text-[10px] text-white/35">step {String(step).padStart(3,'0')}</span>
            </div>
            <div ref={termRef} className="cg-scroll cg-scroll-dark h-[300px] overflow-y-auto px-4 py-3 font-mono text-[12px] leading-relaxed text-white/80">
              {terminalLines.map((l, i) => (
                <div key={i} className="cg-slide-in">
                  <span className="text-white/35">{l.ts}</span>{' '}
                  <span className={
                    l.level==='ok'   ? 'text-[hsl(var(--green))]'
                  : l.level==='warn' ? 'text-[hsl(var(--amber))]'
                  : l.level==='crit' ? 'text-[hsl(var(--red))]'
                  : 'text-white/85'
                  }>{l.msg}</span>
                </div>
              ))}
              <div className="text-[hsl(var(--red))] cg-blink">$ </div>
            </div>
          </div>
          {/* Reward Canvas */}
          <div className="rounded-2xl border border-[hsl(var(--accent-2))]/25 overflow-hidden cg-shadow-card" style={{ background:'#0a1614' }}>
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
              <div className="font-mono text-[11px] text-white/55">reward_curve.live</div>
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="text-white/45">CURRENT <span className="text-[hsl(var(--green))]">{current.toFixed(2)}</span></span>
                <span className="text-white/45">PEAK <span className="text-white">{peak.toFixed(2)}</span></span>
                <span className="text-white/45">BASELINE <span className="text-[hsl(var(--red))]">0.23</span></span>
              </div>
            </div>
            <div className="px-3 pt-3 pb-2">
              <canvas ref={canvasRef} className="block w-full h-[266px]" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

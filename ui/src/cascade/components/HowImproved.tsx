import { useEffect, useRef, useState } from 'react';

const PILLARS = [
  { n:'01', title:'Reasoning', sub:'Chain-of-Thought Planning', desc:'Defender narrates intent before action, enabling reward shaping over the reasoning trace itself.', value:88, color:'hsl(var(--accent))' },
  { n:'02', title:'Reward Signal', sub:'Dense Per-Step Rewards', desc:'Health deltas, dependency satisfaction, and budget efficiency combine into a dense signal.', value:94, color:'hsl(var(--accent-2))' },
  { n:'03', title:'Curriculum', sub:'Progressive Task Difficulty', desc:'Easy → medium → hard tasks unlock as the policy crosses score thresholds.', value:72, color:'hsl(var(--amber))' },
  { n:'04', title:'Efficiency', sub:'Budget-Aware Decisions', desc:'Penalty for over-spending teaches the agent to retain budget for future cascades.', value:81, color:'hsl(var(--purple))' },
];

function Bar({ value, color, fill }: { value:number; color:string; fill:boolean }) {
  return (
    <div className="h-1.5 w-full rounded-full bg-foreground/[0.08] overflow-hidden">
      <div className="h-full rounded-full transition-[width] duration-1000 ease-out" style={{ width: fill ? `${value}%` : '0%', background: color }} />
    </div>
  );
}

export function HowImproved() {
  const sectionRef = useRef<HTMLElement | null>(null);
  const [fill, setFill] = useState(false);
  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const io = new IntersectionObserver((es) => {
      es.forEach(e => { if (e.isIntersecting) { setFill(true); io.disconnect(); } });
    }, { threshold: 0.25 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <section ref={sectionRef} className="relative px-5 py-20">
      <div className="max-w-[1280px] mx-auto cg-reveal">
        <div className="flex items-end justify-between mb-6 flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/55 mb-2">/ training</div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">How the Agent Improved</h2>
          </div>
          <div className="text-sm text-foreground/55 max-w-md">Four pillars of GRPO training drove the 0.23 → 0.81 jump.</div>
        </div>
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          {PILLARS.map(p => (
            <div key={p.n} className="rounded-2xl bg-card border border-foreground/10 cg-shadow-card p-5">
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-xs text-foreground/45">{p.n}</span>
                <span className="font-mono text-xs font-semibold" style={{ color: p.color }}>{p.value}%</span>
              </div>
              <div className="font-bold text-lg">{p.title}</div>
              <div className="text-sm text-foreground/55 mb-2">{p.sub}</div>
              <p className="text-[13px] text-foreground/70 mb-4 leading-relaxed">{p.desc}</p>
              <Bar value={p.value} color={p.color} fill={fill} />
            </div>
          ))}
        </div>

        <div className="rounded-2xl bg-card border border-foreground/10 cg-shadow-card p-5">
          <div className="font-mono text-[10px] uppercase tracking-wider text-foreground/55 mb-3">before / after — episode score on task_hard</div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { k:'Random policy',  v:'0.23', note:'Acts uniformly at random.', color:'hsl(var(--red))', highlight:false },
              { k:'Zero-shot LLM',   v:'0.49', note:'Qwen2.5-7B base, no fine-tuning.', color:'hsl(var(--amber))', highlight:false },
              { k:'GRPO trained',    v:'0.81', note:'Qwen2.5-7B + LoRA · 50 episodes.', color:'hsl(var(--accent))', highlight:true },
            ].map(x => (
              <div key={x.k} className={`rounded-xl p-4 border ${x.highlight ? 'border-[hsl(var(--accent))]/40 bg-[hsl(var(--accent))]/5' : 'border-foreground/10 bg-foreground/[0.02]'}`}>
                <div className="font-mono text-[10px] uppercase tracking-wider text-foreground/55">{x.k}</div>
                <div className="text-3xl font-bold mt-1" style={{ color: x.color }}>{x.v}</div>
                <div className="text-xs text-foreground/60 mt-1">{x.note}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

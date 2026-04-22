import type { TimelineEntry } from '../data/types';

interface Props {
  entries: TimelineEntry[];
}
const tone = (t: TimelineEntry['tone']) => t==='crit' ? 'hsl(var(--red))' : t==='warn' ? 'hsl(var(--amber))' : t==='ok' ? 'hsl(var(--green))' : 'hsl(var(--accent))';

export function Timeline({ entries }: Props) {
  return (
    <section className="relative px-5 py-20">
      <div className="max-w-[1280px] mx-auto cg-reveal">
        <div className="flex items-end justify-between mb-6 flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/55 mb-2">/ trace</div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Step-by-Step Episode Trace</h2>
          </div>
        </div>
        {entries.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-foreground/15 p-10 text-center font-mono text-sm text-foreground/55">
            Press Play to begin episode simulation.
          </div>
        ) : (
          <div className="relative pl-8">
            <div className="absolute left-3 top-2 bottom-2 w-px" style={{ background: 'linear-gradient(to bottom, hsl(var(--accent)), hsl(var(--green)), hsl(var(--amber)))' }} />
            <div className="space-y-5">
              {entries.map((e, i) => (
                <div key={i} className="relative cg-slide-in">
                  <span className="absolute -left-[22px] top-1.5 h-3 w-3 rounded-full" style={{ background: tone(e.tone), boxShadow: `0 0 0 4px hsl(0 0% 100%), 0 0 0 5px ${tone(e.tone)}30` }} />
                  <div className="rounded-xl bg-card border border-foreground/10 cg-shadow-card p-4">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-[10px] tracking-wider uppercase text-foreground/55">step {String(e.step).padStart(2,'0')}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: `${tone(e.tone)}1a`, color: tone(e.tone) }}>{e.tone.toUpperCase()}</span>
                    </div>
                    <div className="font-semibold text-base">{e.title}</div>
                    <div className="text-sm text-foreground/65 mt-1">{e.description}</div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {e.tags.map(t => (
                        <span key={t} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-foreground/[0.05] text-foreground/65">#{t}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

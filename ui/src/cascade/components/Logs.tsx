import type { EventLine } from '../data/types';
import { useEffect, useRef } from 'react';

interface Props {
  logs: EventLine[];
  onDownloadReport: () => void;
  onDownloadLogs: () => void;
  onDownloadJson: () => void;
  onDownloadCsv: () => void;
}

const levelStyle = (l: EventLine['level']) =>
  l==='ok'   ? 'bg-[hsl(var(--green))]/15 text-[hsl(var(--green))]'
: l==='warn' ? 'bg-[hsl(var(--amber))]/15 text-[hsl(var(--amber))]'
: l==='crit' ? 'bg-[hsl(var(--red))]/15 text-[hsl(var(--red))]'
:              'bg-white/10 text-white/80';

const levelLabel = (l: EventLine['level']) => l==='ok' ? 'OK' : l==='crit' ? 'CRIT' : l.toUpperCase();

export function Logs({ logs, onDownloadReport, onDownloadLogs, onDownloadJson, onDownloadCsv }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [logs]);
  return (
    <section id="logs" className="relative px-5 py-20">
      <div className="max-w-[1280px] mx-auto cg-reveal">
        <div className="flex items-end justify-between mb-6 flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/55 mb-2">/ logs</div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Episode Logs &amp; Reports</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={onDownloadReport} className="h-10 px-4 rounded-full text-white text-sm font-semibold cg-shadow-card inline-flex items-center gap-2" style={{ background:'linear-gradient(90deg, hsl(var(--accent)), hsl(var(--purple)))' }}>
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9 14l2 2 4-4"/></svg>
              Download Branded Report (.html)
            </button>
            <button onClick={onDownloadLogs} className="h-10 px-4 rounded-full border border-foreground/15 bg-background hover:bg-foreground/5 text-sm font-semibold inline-flex items-center gap-2">
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 6h16M4 12h16M4 18h10"/></svg>
              Event Log
            </button>
            <button onClick={onDownloadJson} className="h-10 px-4 rounded-full border border-foreground/15 bg-background hover:bg-foreground/5 text-sm font-semibold inline-flex items-center gap-2">
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h2"/><path d="M16 3h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-2"/></svg>
              Episode JSON
            </button>
            <button onClick={onDownloadCsv}  className="h-10 px-4 rounded-full border border-foreground/15 bg-background hover:bg-foreground/5 text-sm font-semibold inline-flex items-center gap-2">
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-6 4 4 5-8"/></svg>
              Reward Curve
            </button>
          </div>
        </div>
        <div className="rounded-2xl overflow-hidden border border-white/10 cg-shadow-card" style={{ background:'#060606' }}>
          <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5">
            <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
            <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
            <span className="h-3 w-3 rounded-full bg-[#28c840]" />
            <span className="ml-3 font-mono text-[11px] text-white/55">cascadeguard.log — tail -f</span>
          </div>
          <div ref={ref} className="cg-scroll cg-scroll-dark max-h-[380px] overflow-y-auto px-4 py-3 font-mono text-[12px] leading-relaxed text-white/85 space-y-1">
            {logs.length === 0 && <div className="text-white/40">// no entries yet — press Play in the Live Map.</div>}
            {logs.map((l, i) => (
              <div key={i} className="cg-slide-in flex items-start gap-2">
                <span className="text-white/35">{l.ts}</span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${levelStyle(l.level)}`}>{levelLabel(l.level)}</span>
                <span className="flex-1">{l.msg}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

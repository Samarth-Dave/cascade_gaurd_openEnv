interface Props {
  city: string;
  step: number;
  episodeId: string;
}
export function Hero({ city, step, episodeId }: Props) {
  return (
    <section id="hero" className="relative min-h-[100svh] flex items-center justify-center px-6 pt-[60px]">
      <div className="max-w-[1100px] w-full text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-foreground/10 bg-background/60 backdrop-blur text-[11px] font-mono tracking-wide cg-fade-up">
          <span className="relative inline-flex h-2 w-2 rounded-full bg-[hsl(var(--green))] cg-live-pulse" />
          <span className="text-foreground/70">Live Episode Active —</span>
          <span className="text-foreground font-semibold">{episodeId} · {city} · task_hard · step {String(step).padStart(3,'0')}</span>
        </div>
        <h1 className="mt-6 font-bold tracking-[-0.06em] leading-[0.88] cg-fade-up" style={{ fontSize: 'clamp(4rem, 11vw, 9rem)', animationDelay: '.08s' }}>
          <span className="block">CASCADE</span>
          <span className="block text-[hsl(var(--accent))]">GUARD</span>
        </h1>
        <p className="mt-7 mx-auto max-w-[720px] text-base sm:text-lg text-foreground/55 font-light leading-relaxed cg-fade-up" style={{ animationDelay: '.16s' }}>
          GRPO-trained AI autonomously defends critical city infrastructure against adversarial cascade failures — in real time.
        </p>
        <div className="mt-9 flex flex-wrap justify-center gap-3 cg-fade-up" style={{ animationDelay: '.24s' }}>
          <a href="#map" className="inline-flex items-center gap-2 h-11 px-5 rounded-full bg-[hsl(var(--accent))] text-white text-sm font-semibold hover:opacity-90 transition-opacity">
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor"><path d="M5 4l14 8-14 8z"/></svg>
            Live Map
          </a>
          <a href="#duel" className="inline-flex items-center gap-2 h-11 px-5 rounded-full border border-foreground/15 bg-background hover:bg-foreground/5 text-sm font-semibold transition-colors">
            AI Duel
          </a>
          <a href="#graph" className="inline-flex items-center gap-2 h-11 px-5 rounded-full border border-[hsl(var(--red))]/30 bg-[hsl(var(--red))]/5 text-[hsl(var(--red))] hover:bg-[hsl(var(--red))]/10 text-sm font-semibold transition-colors">
            Neural Graph
          </a>
        </div>
      </div>
    </section>
  );
}

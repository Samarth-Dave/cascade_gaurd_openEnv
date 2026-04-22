import type { AgentTurn } from '../data/types';

interface Props {
  attacker: AgentTurn;
  defender: AgentTurn;
  attackPressure: number;
  defenseScore: number;
}
export function AIDuel({ attacker, defender, attackPressure, defenseScore }: Props) {
  return (
    <section id="duel" className="relative px-5 py-20">
      <div className="max-w-[1280px] mx-auto cg-reveal">
        <div className="flex items-end justify-between mb-6 flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-foreground/55 mb-2">/ ai duel</div>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Adversarial vs Defensive Reasoning</h2>
          </div>
          <div className="text-sm text-foreground/55 max-w-md">Two LLMs reason in parallel each step. Attacker probes for cascade vectors; GRPO-trained defender allocates a finite budget to contain them.</div>
        </div>
        <div className="relative grid lg:grid-cols-2 gap-0 rounded-2xl overflow-hidden border border-foreground/10 cg-shadow-card min-h-[480px]">
          {/* Attacker */}
          <div className="bg-[#0d0d0d] text-white p-6 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="px-2 py-1 rounded-md bg-[hsl(var(--red))]/15 text-[hsl(var(--red))] text-[10px] font-mono font-bold tracking-wider">ATTACKER</span>
              <span className="font-mono text-[10px] text-white/40">adversarial_attacker.py</span>
            </div>
            <h3 className="text-xl font-bold mb-1">Adversarial Agent</h3>
            <p className="text-white/45 text-xs mb-4">Probes infrastructure for cascade vectors and dependency exploits.</p>
            <div className="flex-1 space-y-3 overflow-y-auto cg-scroll cg-scroll-dark pr-1">
              <div className="cg-cot-in rounded-lg bg-white/[0.04] border border-white/10 p-3">
                <div className="text-[10px] font-mono text-white/40 mb-1">⟨think⟩</div>
                <div className="text-[12.5px] leading-relaxed text-white/75">{attacker.think}</div>
              </div>
              <div className="cg-cot-in rounded-lg bg-[hsl(var(--red))]/15 border border-[hsl(var(--red))]/30 p-3" style={{ animationDelay: '.1s' }}>
                <div className="text-[10px] font-mono text-[hsl(var(--red))] mb-1">⟨action⟩</div>
                <div className="text-[12.5px] leading-relaxed font-mono text-white">{attacker.action}</div>
              </div>
            </div>
            <div className="mt-5">
              <div className="flex items-center justify-between text-[11px] font-mono text-white/55 mb-1.5">
                <span>Attack pressure</span>
                <span>{Math.round(attackPressure*100)}%</span>
              </div>
              <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${attackPressure*100}%`, background: 'linear-gradient(90deg, hsl(var(--red))/0.6, hsl(var(--red)))' }} />
              </div>
            </div>
          </div>
          {/* Defender */}
          <div className="bg-[hsl(var(--accent))]/[0.06] p-6 flex flex-col border-l border-foreground/10">
            <div className="flex items-center justify-between mb-4">
              <span className="px-2 py-1 rounded-md bg-[hsl(var(--accent))]/15 text-[hsl(var(--accent))] text-[10px] font-mono font-bold tracking-wider">DEFENDER</span>
              <span className="font-mono text-[10px] text-foreground/45">Qwen2.5-7B · LoRA ep42</span>
            </div>
            <h3 className="text-xl font-bold mb-1">GRPO-Trained Agent</h3>
            <p className="text-foreground/55 text-xs mb-4">Reasons over budget, sector health, and dependency risk to dispatch optimal actions.</p>
            <div className="flex-1 space-y-3 overflow-y-auto cg-scroll pr-1">
              <div className="cg-cot-in rounded-lg bg-[hsl(var(--accent))]/8 border border-[hsl(var(--accent))]/20 p-3">
                <div className="text-[10px] font-mono text-[hsl(var(--accent))] mb-1">⟨think⟩</div>
                <div className="text-[12.5px] leading-relaxed text-foreground/85">{defender.think}</div>
              </div>
              <div className="cg-cot-in rounded-lg bg-[hsl(var(--green))]/10 border border-[hsl(var(--green))]/25 p-3" style={{ animationDelay: '.1s' }}>
                <div className="text-[10px] font-mono text-[hsl(var(--green))] mb-1">⟨action⟩</div>
                <div className="text-[12.5px] leading-relaxed font-mono text-foreground">{defender.action}</div>
              </div>
            </div>
            <div className="mt-5">
              <div className="flex items-center justify-between text-[11px] font-mono text-foreground/55 mb-1.5">
                <span>Defense score</span>
                <span>{Math.round(defenseScore*100)}%</span>
              </div>
              <div className="h-2 rounded-full bg-foreground/10 overflow-hidden">
                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${defenseScore*100}%`, background: 'linear-gradient(90deg, hsl(var(--accent)), hsl(var(--green)))' }} />
              </div>
            </div>
          </div>
          {/* VS badge */}
          <div className="hidden lg:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10 h-14 w-14 items-center justify-center rounded-full bg-background border border-foreground/15 cg-shadow-pop font-bold text-sm tracking-wider">
            VS
          </div>
        </div>
      </div>
    </section>
  );
}

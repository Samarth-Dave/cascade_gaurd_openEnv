import type { SectorHealth } from "@/types";

const toneMap: Record<keyof SectorHealth, string> = {
  power: "hsl(var(--red))",
  water: "hsl(var(--green))",
  hospital: "hsl(var(--purple))",
  telecom: "hsl(var(--accent))",
};

export function MetricRing({
  label,
  value,
  sector,
}: {
  label: string;
  value: number;
  sector: keyof SectorHealth;
}) {
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const safeValue = Math.max(0, Math.min(1, value));
  const offset = circumference * (1 - safeValue);

  return (
    <div className="rounded-2xl border border-black/10 bg-white/85 p-4 shadow-[0_18px_40px_-28px_rgba(0,0,0,0.35)]">
      <div className="flex items-center gap-4">
        <svg width="84" height="84" viewBox="0 0 84 84" className="shrink-0">
          <circle cx="42" cy="42" r={radius} fill="none" stroke="rgba(0,0,0,0.08)" strokeWidth="8" />
          <circle
            cx="42"
            cy="42"
            r={radius}
            fill="none"
            stroke={toneMap[sector]}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            transform="rotate(-90 42 42)"
          />
          <text x="42" y="39" textAnchor="middle" className="fill-current font-mono text-[11px]">
            {Math.round(safeValue * 100)}%
          </text>
          <text x="42" y="52" textAnchor="middle" className="fill-current text-[8px] uppercase tracking-[0.2em] text-foreground/50">
            live
          </text>
        </svg>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">{label}</div>
          <div className="mt-1 text-xl font-semibold">{safeValue >= 0.8 ? "Stable" : safeValue >= 0.5 ? "Degraded" : "Critical"}</div>
        </div>
      </div>
    </div>
  );
}

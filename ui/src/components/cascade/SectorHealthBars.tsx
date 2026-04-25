import type { SectorHealth } from "@/types";

const sectorMeta: Array<{ key: keyof SectorHealth; label: string; color: string }> = [
  { key: "power", label: "Power", color: "hsl(var(--red))" },
  { key: "water", label: "Water", color: "hsl(var(--green))" },
  { key: "hospital", label: "Hospital", color: "hsl(var(--purple))" },
  { key: "telecom", label: "Telecom", color: "hsl(var(--accent))" },
];

export function SectorHealthBars({ sectorHealth }: { sectorHealth: SectorHealth }) {
  return (
    <div className="space-y-3">
      {sectorMeta.map((sector) => {
        const value = Math.max(0, Math.min(1, sectorHealth[sector.key]));
        return (
          <div key={sector.key} className="space-y-1.5">
            <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.16em] text-foreground/55">
              <span>{sector.label}</span>
              <span>{Math.round(value * 100)}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-black/8">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${value * 100}%`, backgroundColor: sector.color }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

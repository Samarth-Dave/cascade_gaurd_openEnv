import { cn } from "@/lib/utils";

export function StateBadge({
  label,
  tone = "neutral",
}: {
  label: string;
  tone?: "neutral" | "info" | "warn" | "danger" | "success";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.16em]",
        tone === "neutral" && "border-black/10 bg-black/5 text-foreground/70",
        tone === "info" && "border-[hsl(var(--accent))]/25 bg-[hsl(var(--accent))]/10 text-[hsl(var(--accent))]",
        tone === "warn" && "border-[hsl(var(--amber))]/30 bg-[hsl(var(--amber))]/12 text-[hsl(var(--amber))]",
        tone === "danger" && "border-[hsl(var(--red))]/30 bg-[hsl(var(--red))]/10 text-[hsl(var(--red))]",
        tone === "success" && "border-[hsl(var(--green))]/30 bg-[hsl(var(--green))]/10 text-[hsl(var(--green))]",
      )}
    >
      {label}
    </span>
  );
}

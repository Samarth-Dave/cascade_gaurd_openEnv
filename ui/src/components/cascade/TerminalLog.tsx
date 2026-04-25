import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";
import type { WsLevel } from "@/types";

export interface LogLine {
  id: string;
  timestamp: string;
  level: WsLevel;
  message: string;
}

const levelClass: Record<WsLevel, string> = {
  INFO: "text-[hsl(var(--accent))]",
  WARN: "text-[hsl(var(--amber))]",
  CRIT: "text-[hsl(var(--red))]",
  OK: "text-[hsl(var(--green))]",
};

export function TerminalLog({ lines }: { lines: LogLine[] }) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [lines]);

  return (
    <div className="flex h-[320px] flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#0e0e0e]">
      <div className="border-b border-white/10 px-4 py-3 font-mono text-[11px] uppercase tracking-[0.18em] text-white/55">
        Live Terminal
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto px-4 py-4 font-mono text-[12px]">
        {lines.length === 0 ? (
          <div className="text-white/40">00:00:00 [INFO] Awaiting episode events</div>
        ) : (
          lines.map((line) => (
            <div key={line.id} className="leading-relaxed text-white/85">
              <span className="text-white/35">{line.timestamp}</span>{" "}
              <span className={cn(levelClass[line.level])}>[{line.level}]</span>{" "}
              <span>{line.message}</span>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

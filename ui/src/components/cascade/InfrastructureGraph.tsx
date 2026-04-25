import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { InfraEdge, InfraNode } from "@/types";

const statusColor: Record<string, string> = {
  healthy: "#00a878",
  degraded: "#d97706",
  critical: "#e8192c",
  isolated: "#6b7280",
};

function fallbackPosition(index: number, total: number) {
  const angle = (Math.PI * 2 * index) / Math.max(total, 1);
  return {
    x: 80 + Math.cos(angle) * 180,
    y: 80 + Math.sin(angle) * 180,
  };
}

export function InfrastructureGraph({
  nodes,
  edges,
  onAction,
}: {
  nodes: InfraNode[];
  edges: InfraEdge[];
  onAction: (action: string, target: string) => void;
}) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const positions = useMemo(() => {
    const validNodes = nodes.filter((node) => Number.isFinite(node.lat) && Number.isFinite(node.lng) && (node.lat !== 0 || node.lng !== 0));
    const latValues = validNodes.map((node) => node.lat);
    const lngValues = validNodes.map((node) => node.lng);
    const latMin = Math.min(...latValues, 0);
    const latMax = Math.max(...latValues, 1);
    const lngMin = Math.min(...lngValues, 0);
    const lngMax = Math.max(...lngValues, 1);
    return Object.fromEntries(
      nodes.map((node, index) => {
        if (validNodes.length > 1 && (node.lat !== 0 || node.lng !== 0)) {
          const x = 70 + ((node.lng - lngMin) / Math.max(lngMax - lngMin, 0.001)) * 520;
          const y = 90 + (1 - (node.lat - latMin) / Math.max(latMax - latMin, 0.001)) * 380;
          return [node.id, { x, y }];
        }
        return [node.id, fallbackPosition(index, nodes.length)];
      }),
    );
  }, [nodes]);

  const selectedNode = nodes.find((node) => node.id === selectedNodeId) ?? null;

  return (
    <div className="relative overflow-hidden rounded-[28px] border border-black/10 bg-[radial-gradient(circle_at_top_left,rgba(20,70,245,0.12),transparent_40%),linear-gradient(180deg,#ffffff,rgba(255,255,255,0.92))] p-4 shadow-[0_30px_80px_-40px_rgba(0,0,0,0.35)]">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/50">Infrastructure Graph</div>
          <div className="mt-1 text-lg font-semibold">Live dependency map</div>
        </div>
        <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-foreground/45">
          {nodes.length} nodes / {edges.length} edges
        </div>
      </div>

      {nodes.length === 0 ? (
        <div className="grid h-[520px] place-items-center rounded-[24px] border border-dashed border-black/15 bg-white/60 text-center">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-foreground/45">No active episode</div>
            <p className="mt-2 max-w-sm text-sm text-foreground/60">
              Start an episode to render the live infrastructure graph and node controls.
            </p>
          </div>
        </div>
      ) : (
        <div className="relative h-[520px] overflow-hidden rounded-[24px] border border-black/8 bg-[linear-gradient(135deg,rgba(20,70,245,0.06),rgba(255,255,255,0.9))]">
          <svg viewBox="0 0 660 520" className="h-full w-full">
            <defs>
              <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
                <path d="M 26 0 L 0 0 0 26" fill="none" stroke="rgba(0,0,0,0.04)" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="660" height="520" fill="url(#grid)" />
            {edges.map((edge) => {
              const source = positions[edge.source];
              const target = positions[edge.target];
              if (!source || !target) return null;
              const stroke =
                edge.kind === "cascade"
                  ? "#e8192c"
                  : edge.kind === "recovery"
                    ? "#00a878"
                    : "rgba(0,0,0,0.28)";
              return (
                <line
                  key={edge.id}
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  stroke={stroke}
                  strokeWidth={edge.kind === "dependency" ? 2 : 3}
                  strokeDasharray={edge.kind === "cascade" ? "10 6" : undefined}
                  className={cn(edge.kind === "cascade" && "animate-[cg-dashFlow_1.4s_linear_infinite]")}
                />
              );
            })}
            {nodes.map((node) => {
              const position = positions[node.id];
              return (
                <g
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  className="cursor-pointer"
                  transform={`translate(${position.x}, ${position.y})`}
                >
                  {node.status === "critical" ? (
                    <circle r="24" fill="none" stroke="#e8192c" strokeWidth="2" className="animate-[cg-ringOut_1.6s_ease-out_infinite]" />
                  ) : null}
                  <circle r="18" fill={statusColor[node.status]} stroke="white" strokeWidth="3" />
                  <text y="42" textAnchor="middle" className="fill-current font-mono text-[10px] tracking-[0.14em]">
                    {node.id}
                  </text>
                </g>
              );
            })}
          </svg>

          {selectedNode ? (
            <div className="absolute bottom-4 left-4 w-[280px] rounded-2xl border border-black/10 bg-white/92 p-4 shadow-[0_18px_40px_-24px_rgba(0,0,0,0.45)] backdrop-blur">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">{selectedNode.id}</div>
                  <div className="mt-1 text-base font-semibold">{selectedNode.label}</div>
                </div>
                <span className="rounded-full px-2 py-1 font-mono text-[10px] uppercase tracking-[0.14em]" style={{ backgroundColor: `${statusColor[selectedNode.status]}18`, color: statusColor[selectedNode.status] }}>
                  {selectedNode.status}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 text-sm text-foreground/65">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Sector</div>
                  <div>{selectedNode.sector}</div>
                </div>
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Health</div>
                  <div>{selectedNode.health_pct}%</div>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {["ISOLATE", "REPAIR", "REROUTE"].map((action) => (
                  <Button key={action} size="sm" variant="outline" onClick={() => onAction(action, selectedNode.id)}>
                    {action}
                  </Button>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

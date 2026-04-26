import { useMemo, useState } from "react";
import type { LatLngTuple, PathOptions } from "leaflet";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";

import { Button } from "@/components/ui/button";
import type { InfraEdge, InfraNode } from "@/types";

const MUMBAI_CENTER: LatLngTuple = [19.0, 72.85];
const ZOOM = 12;

function hasGeoCoordinates(node: InfraNode) {
  return Number.isFinite(node.lat) && Number.isFinite(node.lng) && (node.lat !== 0 || node.lng !== 0);
}

function getNodeColor(node: InfraNode): string {
  if (node.status === "isolated") return "#6b7280";
  if (!node.is_operational || node.health_pct < 30) return "#e8192c";
  if (node.health_pct < 70) return "#d97706";
  return "#00a878";
}

function getNodeVisualStatus(node: InfraNode): "healthy" | "degraded" | "critical" | "isolated" {
  if (node.status === "isolated" || node.metadata.isolated === true) {
    return "isolated";
  }
  if (!node.is_operational || node.status === "critical" || node.health_pct < 30) {
    return "critical";
  }
  if (node.status === "degraded" || node.health_pct < 70) {
    return "degraded";
  }
  return "healthy";
}

function getEdgeStyle(kind: InfraEdge["kind"]): PathOptions {
  if (kind === "cascade") {
    return { color: "#e8192c", weight: 3, opacity: 0.95, dashArray: "8 8" };
  }
  if (kind === "recovery") {
    return { color: "#00a878", weight: 3, opacity: 0.95, dashArray: "8 8" };
  }
  return { color: "#6b7280", weight: 2.5, opacity: 0.75 };
}

function getNodeActionGuard(action: string, node: InfraNode) {
  if (action === "REPAIR" && node.is_operational) {
    return "Repair is only useful once the node has actually failed.";
  }
  if (action === "ISOLATE" && node.is_operational) {
    return "Isolate only applies to failed nodes.";
  }
  return null;
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

  const geoNodes = useMemo(() => nodes.filter(hasGeoCoordinates), [nodes]);
  const nodePositions = useMemo(
    () =>
      Object.fromEntries(
        geoNodes.map((node) => [node.id, [node.lat, node.lng] as LatLngTuple]),
      ),
    [geoNodes],
  );
  const selectedNode = nodes.find((node) => node.id === selectedNodeId) ?? null;
  const selectedNodeVisualStatus = selectedNode ? getNodeVisualStatus(selectedNode) : null;

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
      ) : geoNodes.length === 0 ? (
        <div className="grid h-[520px] place-items-center rounded-[24px] border border-dashed border-black/15 bg-white/60 text-center">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-foreground/45">Missing geo data</div>
            <p className="mt-2 max-w-sm text-sm text-foreground/60">
              No node coordinates were available for this episode, so the live map cannot be plotted yet.
            </p>
          </div>
        </div>
      ) : (
        <div className="relative h-[520px] overflow-hidden rounded-[24px] border border-black/8">
          <MapContainer center={MUMBAI_CENTER} zoom={ZOOM} className="h-full w-full" scrollWheelZoom>
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {edges.map((edge) => {
              const source = nodePositions[edge.source];
              const target = nodePositions[edge.target];
              if (!source || !target) {
                return null;
              }

              return (
                <Polyline
                  key={edge.id}
                  positions={[source, target]}
                  pathOptions={getEdgeStyle(edge.kind)}
                />
              );
            })}

            {geoNodes.map((node) => {
              const color = getNodeColor(node);
              const position = nodePositions[node.id];
              if (!position) {
                return null;
              }

              const visualStatus = getNodeVisualStatus(node);
              // The key includes the visual bucket + integer health so React
              // remounts the marker whenever the node transitions between
              // healthy / degraded / critical / isolated. Without this,
              // react-leaflet sometimes keeps a stale Leaflet path style and
              // the map appears "frozen" even though state has updated.
              const markerKey = `${node.id}-${visualStatus}-${node.health_pct}`;

              return (
                <CircleMarker
                  key={markerKey}
                  center={position}
                  radius={10}
                  eventHandlers={{ click: () => setSelectedNodeId(node.id) }}
                  pathOptions={{
                    color: "#ffffff",
                    weight: 2,
                    fillColor: color,
                    fillOpacity: 0.95,
                    className: visualStatus === "critical" ? "node-critical-pulse" : undefined,
                  }}
                >
                  <Tooltip direction="top" offset={[0, -8]}>
                    <div className="space-y-1">
                      <div className="font-semibold">{node.id}</div>
                      <div>Sector: {node.sector}</div>
                      <div>Health: {node.health_pct}%</div>
                      <div>Status: {node.status}</div>
                    </div>
                  </Tooltip>
                </CircleMarker>
              );
            })}
          </MapContainer>

          {selectedNode ? (
            <div className="pointer-events-auto absolute bottom-4 left-4 z-10 w-[280px] rounded-2xl border border-black/10 bg-white/92 p-4 shadow-[0_18px_40px_-24px_rgba(0,0,0,0.45)] backdrop-blur">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">{selectedNode.id}</div>
                  <div className="mt-1 text-base font-semibold">{selectedNode.label}</div>
                </div>
                <span
                  className="rounded-full px-2 py-1 font-mono text-[10px] uppercase tracking-[0.14em]"
                  style={{
                    backgroundColor: `${getNodeColor(selectedNode)}18`,
                    color: getNodeColor(selectedNode),
                  }}
                >
                  {selectedNodeVisualStatus}
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
                {["ISOLATE", "REPAIR", "REROUTE"].map((action) => {
                  const guard = getNodeActionGuard(action, selectedNode);
                  return (
                    <Button
                      key={action}
                      size="sm"
                      variant="outline"
                      onClick={() => onAction(action, selectedNode.id)}
                      disabled={guard !== null}
                      title={guard ?? undefined}
                    >
                      {action}
                    </Button>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

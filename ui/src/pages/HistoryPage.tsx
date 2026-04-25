import { useQuery } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";
import { useState } from "react";

import { apiClient } from "@/api/client";
import { StateBadge } from "@/components/cascade/StateBadge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export default function HistoryPage() {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const historyQuery = useQuery({
    queryKey: ["episode-history"],
    queryFn: apiClient.getEpisodeHistory,
  });

  if (historyQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 rounded-[28px]" />
        <Skeleton className="h-24 rounded-[28px]" />
        <Skeleton className="h-24 rounded-[28px]" />
      </div>
    );
  }

  if (historyQuery.isError) {
    return (
      <Alert className="border-[hsl(var(--red))]/30 bg-[hsl(var(--red))]/8">
        <AlertTitle>Could not load episode history</AlertTitle>
        <AlertDescription className="flex items-center gap-3">
          <span>The backend may be offline or no episodes have been recorded yet.</span>
          <Button size="sm" variant="outline" onClick={() => void historyQuery.refetch()}>
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Episode History</div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">Completed containment runs</h1>
      </div>

      {historyQuery.data.length === 0 ? (
        <Card className="rounded-[28px] border-black/10 bg-white/88 p-6 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
          <p className="text-sm text-foreground/60">No completed episodes yet. Run a simulation and finish it to populate this table.</p>
        </Card>
      ) : (
        <div className="space-y-4">
          {historyQuery.data.map((episode, index) => {
            const isExpanded = expandedId === episode.session_id;
            return (
              <Card key={episode.session_id} className="rounded-[28px] border-black/10 bg-white/88 p-0 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
                <button
                  className="grid w-full gap-4 rounded-[28px] px-5 py-4 text-left md:grid-cols-[48px_1.1fr_repeat(4,minmax(0,0.7fr))_auto]"
                  onClick={() => setExpandedId(isExpanded ? null : episode.session_id)}
                >
                  <div className="font-mono text-sm text-foreground/45">#{index + 1}</div>
                  <div>
                    <div className="font-semibold">{episode.task}</div>
                    <div className="text-sm text-foreground/55">{episode.session_id.slice(0, 8)}</div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Score</div>
                    <div className="text-lg font-semibold">{episode.score.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Budget Used</div>
                    <div className="text-lg font-semibold">{episode.budget_used.toFixed(1)}</div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Steps</div>
                    <div className="text-lg font-semibold">{episode.steps}</div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Cascade Peak</div>
                    <div className="text-lg font-semibold">{episode.cascade_peak}</div>
                  </div>
                  <div className="flex items-center justify-end gap-3">
                    <StateBadge
                      label={episode.outcome}
                      tone={episode.outcome === "CONTAINED" ? "success" : episode.outcome === "FAILED" ? "danger" : "warn"}
                    />
                    <ChevronDown className={`h-4 w-4 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                  </div>
                </button>

                {isExpanded ? (
                  <div className="border-t border-black/8 px-5 py-4">
                    <div className="space-y-3">
                      {episode.trajectory.map((item) => (
                        <div key={`${episode.session_id}-${item.step}`} className="grid gap-3 rounded-2xl border border-black/8 bg-black/[0.03] px-4 py-3 md:grid-cols-[90px_1.1fr_repeat(3,minmax(0,0.7fr))]">
                          <div className="font-mono text-[12px] uppercase tracking-[0.16em] text-foreground/45">Step {item.step}</div>
                          <div className="font-semibold">
                            {item.action} {item.target ?? "null"}
                          </div>
                          <div className="text-sm text-foreground/65">reward {item.reward_step.toFixed(3)}</div>
                          <div className="text-sm text-foreground/65">score {item.score_after.toFixed(3)}</div>
                          <div className="text-sm text-foreground/65">cascade {item.cascade_depth}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

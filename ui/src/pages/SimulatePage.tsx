import { useQuery } from "@tanstack/react-query";
import { ChevronDown, RefreshCw } from "lucide-react";
import { useState } from "react";

import { apiClient } from "@/api/client";
import { InfrastructureGraph } from "@/components/cascade/InfrastructureGraph";
import { SectorHealthBars } from "@/components/cascade/SectorHealthBars";
import { StateBadge } from "@/components/cascade/StateBadge";
import { TerminalLog } from "@/components/cascade/TerminalLog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { useCascade } from "@/context/CascadeContext";
import type { Task } from "@/types";

const taskOptions: Task[] = [
  "task_easy",
  "task_medium",
  "task_hard",
  "task_blackout",
  "task_cyberattack",
];

const manualActions = ["MONITOR", "REPAIR", "ISOLATE", "REROUTE"];

export default function SimulatePage() {
  const {
    config,
    setConfig,
    state,
    lastStepResult,
    logs,
    agentThinking,
    autoRun,
    setAutoRun,
    pending,
    error,
    isConnected,
    resetEpisode,
    stepEpisode,
    runAgentStep,
    sessionId,
  } = useCascade();
  const [manualAction, setManualAction] = useState("REPAIR");
  const [manualTarget, setManualTarget] = useState<string | null>(null);
  const [thinkingOpen, setThinkingOpen] = useState(true);

  const graphQuery = useQuery({
    queryKey: ["environment-graph", sessionId],
    queryFn: () => apiClient.getEnvironmentGraph(sessionId ?? undefined),
    enabled: !state,
  });

  const availableTargets = state?.nodes ?? [];

  return (
    <div className="space-y-6">
      {error ? (
        <Alert className="border-[hsl(var(--red))]/30 bg-[hsl(var(--red))]/8">
          <AlertTitle>Backend request failed</AlertTitle>
          <AlertDescription className="flex flex-wrap items-center gap-3">
            <span>{error}</span>
            <Button size="sm" variant="outline" onClick={() => resetEpisode(config.task, config.seed)}>
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)_300px]">
        <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Live Simulation</div>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">Contain the cascade</h1>

          <div className="mt-6 space-y-4">
            <div className="space-y-2">
              <Label>Task</Label>
              <Select value={config.task} onValueChange={(value) => setConfig({ ...config, task: value as Task })}>
                <SelectTrigger>
                  <SelectValue placeholder="Select task" />
                </SelectTrigger>
                <SelectContent>
                  {taskOptions.map((task) => (
                    <SelectItem key={task} value={task}>
                      {task}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Seed</Label>
              <Input
                type="number"
                value={config.seed}
                onChange={(event) => setConfig({ ...config, seed: Number(event.target.value || 42) })}
              />
            </div>

            <Button className="w-full" onClick={() => resetEpisode(config.task, config.seed)} disabled={pending}>
              {pending ? "Starting..." : "Start Episode"}
            </Button>
          </div>

          <div className="mt-6 space-y-4 rounded-2xl border border-black/8 bg-black/[0.03] p-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Episode Stats</span>
              <StateBadge label={isConnected ? "live" : "offline"} tone={isConnected ? "success" : "warn"} />
            </div>
            {state ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Score</div>
                    <div className="text-2xl font-semibold">{state.episode_score.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Budget</div>
                    <div className="text-2xl font-semibold">{state.budget.toFixed(1)}</div>
                  </div>
                </div>
                <div>
                  <div className="mb-1.5 flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.16em] text-foreground/45">
                    <span>Steps</span>
                    <span>
                      {state.step}/{state.max_steps}
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-black/8">
                    <div
                      className="h-full rounded-full bg-[hsl(var(--accent))]"
                      style={{ width: `${(state.step / Math.max(state.max_steps, 1)) * 100}%` }}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Cascade depth</div>
                    <div className="text-lg font-semibold">{state.cascade_depth}</div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Session</div>
                    <div className="text-sm font-medium">{state.session_id.slice(0, 8)}</div>
                  </div>
                </div>
                <SectorHealthBars sectorHealth={state.sector_health} />
              </div>
            ) : graphQuery.isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-12 rounded-xl" />
                <Skeleton className="h-12 rounded-xl" />
                <Skeleton className="h-24 rounded-xl" />
              </div>
            ) : (
              <div className="text-sm text-foreground/60">Start an episode to reveal score, budget, cascade depth, and sector health.</div>
            )}
          </div>

          <div className="mt-6 space-y-3">
            <Button className="w-full" variant="secondary" onClick={() => void runAgentStep()} disabled={!sessionId || pending || state?.done}>
              Agent Step
            </Button>
            <div className="flex items-center justify-between rounded-2xl border border-black/8 bg-black/[0.03] px-4 py-3">
              <div>
                <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Auto-Run</div>
                <div className="text-sm text-foreground/60">Fire every 1.5 seconds until done</div>
              </div>
              <Switch checked={autoRun} onCheckedChange={setAutoRun} disabled={!sessionId || !!state?.done} />
            </div>
          </div>

          <Collapsible open={thinkingOpen} onOpenChange={setThinkingOpen} className="mt-6 rounded-2xl border border-black/8 bg-black/[0.03]">
            <CollapsibleTrigger asChild>
              <button className="flex w-full items-center justify-between px-4 py-3 text-left">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Agent Thinking</div>
                  <div className="text-sm text-foreground/60">Latest chain-of-thought summary</div>
                </div>
                <ChevronDown className={`h-4 w-4 transition-transform ${thinkingOpen ? "rotate-180" : ""}`} />
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="border-t border-black/8 px-4 py-3 font-mono text-[12px] leading-6 text-foreground/75">
              {agentThinking || "No agent reasoning yet."}
            </CollapsibleContent>
          </Collapsible>
        </Card>

        <div>
          {graphQuery.isLoading && !state ? (
            <Skeleton className="h-[620px] rounded-[28px]" />
          ) : (
            <InfrastructureGraph
              nodes={state?.nodes ?? []}
              edges={state?.edges ?? []}
              onAction={(action, target) => void stepEpisode(action, target)}
            />
          )}
        </div>

        <div className="space-y-6">
          <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Manual Action</div>
            <div className="mt-4 space-y-4">
              <div className="space-y-2">
                <Label>Action</Label>
                <Select value={manualAction} onValueChange={setManualAction}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select action" />
                  </SelectTrigger>
                  <SelectContent>
                    {manualActions.map((action) => (
                      <SelectItem key={action} value={action}>
                        {action}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Target Node</Label>
                <Select value={manualTarget ?? ""} onValueChange={setManualTarget}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select target" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableTargets.map((node) => (
                      <SelectItem key={node.id} value={node.id}>
                        {node.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button className="w-full" onClick={() => void stepEpisode(manualAction, manualTarget)} disabled={!sessionId || pending}>
                Execute
              </Button>
            </div>
          </Card>

          <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Last Action Result</div>
            {lastStepResult ? (
              <div className="mt-4 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-lg font-semibold">
                      {lastStepResult.agent_action?.action ?? manualAction}
                    </div>
                    <div className="text-sm text-foreground/60">
                      {lastStepResult.agent_action?.target ?? manualTarget ?? "No target"}
                    </div>
                  </div>
                  <div className={`text-xl font-semibold ${lastStepResult.reward_step >= 0 ? "text-[hsl(var(--green))]" : "text-[hsl(var(--red))]"}`}>
                    {lastStepResult.reward_step >= 0 ? "+" : ""}
                    {lastStepResult.reward_step.toFixed(3)}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="rounded-2xl border border-black/8 bg-black/[0.03] p-3">
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Episode Score</div>
                    <div className="mt-1 text-xl font-semibold">{lastStepResult.episode_score.toFixed(3)}</div>
                  </div>
                  <div className="rounded-2xl border border-black/8 bg-black/[0.03] p-3">
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Cascade Depth</div>
                    <div className="mt-1 text-xl font-semibold">{lastStepResult.new_state.cascade_depth}</div>
                  </div>
                </div>
                <div className="space-y-2 text-sm">
                  {Object.entries(lastStepResult.grader_breakdown).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between rounded-xl border border-black/8 px-3 py-2">
                      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/45">
                        {key.replace(/_/g, " ")}
                      </span>
                      <span className="font-semibold">{value.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="mt-4 text-sm text-foreground/60">Run a manual or agent step to inspect the grader feedback.</div>
            )}
          </Card>

          <TerminalLog lines={logs} />
        </div>
      </div>
    </div>
  );
}

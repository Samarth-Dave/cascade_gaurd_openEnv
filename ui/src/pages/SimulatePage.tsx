import { useQuery } from "@tanstack/react-query";
import { ChevronDown, Download, Loader2, Play, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

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
import type { InfraNode, Task } from "@/types";

const taskOptions: Task[] = [
  "task_easy",
  "task_medium",
  "task_hard",
  "task_blackout",
  "task_cyberattack",
];

type ManualActionTargetKind = "none" | "node" | "two_nodes" | "sector" | "two_sectors";

interface ManualActionSpec {
  id: string;
  label: string;
  description: string;
  targetKind: ManualActionTargetKind;
  cost?: string;
}

const manualActions: ManualActionSpec[] = [
  { id: "recover", label: "recover", description: "Recover a failed node (upstream must be operational)", targetKind: "node" },
  { id: "harden", label: "harden", description: "Harden a node before storms", cost: "2.0", targetKind: "node" },
  { id: "isolate", label: "isolate", description: "Quarantine a failed node to stop cascade", targetKind: "node" },
  { id: "shed_load", label: "shed_load", description: "Shed load from non-critical, non-hospital node (load > 0.5)", targetKind: "node" },
  { id: "coordinate", label: "coordinate", description: "Clear observation delay for a sector", targetKind: "node" },
  { id: "reroute", label: "reroute(A,B)", description: "Reroute supply: source A to target B", cost: "0.6", targetKind: "two_nodes" },
  { id: "prioritize", label: "prioritize", description: "Boost a single node for 3 steps", targetKind: "node" },
  { id: "deploy_repair_crew", label: "deploy_repair_crew", description: "Speed up an in-progress repair", cost: "1.2", targetKind: "node" },
  { id: "emergency_shutdown", label: "emergency_shutdown", description: "Force shutdown when health < 30%", cost: "0.2", targetKind: "node" },
  { id: "cross_sector_bridge", label: "cross_sector_bridge(SA,SB)", description: "Bridge observability between two sectors", cost: "1.5", targetKind: "two_sectors" },
  { id: "patch_scada", label: "patch_scada", description: "Patch a node's SCADA controller", cost: "0.8", targetKind: "node" },
  { id: "redistribute_load", label: "redistribute_load(A,B)", description: "Move load from node A to node B (same sector)", cost: "0.5", targetKind: "two_nodes" },
  { id: "request_mutual_aid", label: "request_mutual_aid(SECTOR)", description: "Sector-wide health boost (one per episode)", cost: "2.5", targetKind: "sector" },
  { id: "controlled_cascade", label: "controlled_cascade", description: "Sacrifice a node to stabilise neighbors", targetKind: "node" },
  { id: "multi_sector_lockdown", label: "multi_sector_lockdown", description: "Last-resort 2-step lockdown (no cascade)", cost: "2.0", targetKind: "none" },
  { id: "wait", label: "wait", description: "Skip this step (only when nothing actionable)", targetKind: "none" },
];

const SECTOR_OPTIONS = ["power", "water", "hospital", "telecom"];

function findManualAction(id: string): ManualActionSpec {
  return manualActions.find((a) => a.id === id) ?? manualActions[0];
}

function getActionGuard(spec: ManualActionSpec, target: string | null, target2: string | null, node: InfraNode | null) {
  if (spec.targetKind === "none") {
    return { disabled: false, message: null };
  }
  if (spec.targetKind === "sector") {
    if (!target) return { disabled: true, message: "Pick a sector before executing this action." };
    return { disabled: false, message: null };
  }
  if (spec.targetKind === "two_sectors") {
    if (!target || !target2) return { disabled: true, message: "Pick both sectors before executing." };
    if (target === target2) return { disabled: true, message: "Sector A and sector B must differ." };
    return { disabled: false, message: null };
  }
  if (spec.targetKind === "two_nodes") {
    if (!target || !target2) return { disabled: true, message: "Pick both source and target nodes before executing." };
    if (target === target2) return { disabled: true, message: "Source and target must differ." };
    return { disabled: false, message: null };
  }
  if (spec.targetKind === "node") {
    if (!node) return { disabled: true, message: "Select a target node before executing this action." };
    if (spec.id === "recover" && node.is_operational) {
      return { disabled: true, message: "Recover only applies to failed nodes." };
    }
    if (spec.id === "isolate" && node.is_operational) {
      return { disabled: true, message: "Isolate only applies to failed nodes." };
    }
    if (spec.id === "harden" && node.is_hardened) {
      return { disabled: true, message: "This node is already hardened." };
    }
    if (spec.id === "emergency_shutdown" && node.health_pct >= 30) {
      return { disabled: true, message: "Emergency shutdown requires node health below 30%." };
    }
    if (spec.id === "shed_load" && node.sector === "hospital") {
      return { disabled: true, message: "Cannot shed load from hospital nodes." };
    }
    if (spec.id === "shed_load" && node.is_critical) {
      return { disabled: true, message: "Cannot shed load from critical nodes." };
    }
    return { disabled: false, message: null };
  }
  return { disabled: false, message: null };
}

export default function SimulatePage() {
  const {
    config,
    setConfig,
    state,
    currentNodes,
    currentEdges,
    lastStepResult,
    logs,
    agentThinking,
    autoRun,
    setAutoRun,
    pending,
    isResetting,
    isAgentStepping,
    error,
    isConnected,
    resetEpisode,
    stepEpisode,
    runAgentStep,
    sessionId,
  } = useCascade();
  const [manualAction, setManualAction] = useState("recover");
  const [manualTarget, setManualTarget] = useState<string | null>(null);
  const [manualTarget2, setManualTarget2] = useState<string | null>(null);
  const [thinkingOpen, setThinkingOpen] = useState(true);
  const [thinkingProgress, setThinkingProgress] = useState(0);
  const progressIntervalRef = useRef<number | null>(null);
  const progressHideTimeoutRef = useRef<number | null>(null);

  // The progress bar is bound to the actual agent-step API call lifecycle
  // (isAgentStepping). It animates asymptotically toward 90% so it never
  // visibly stalls, then jumps to 100% when the response arrives, holds for
  // 500ms, and finally resets/hides.
  const isAgentThinking = isAgentStepping || thinkingProgress > 0;

  useEffect(() => {
    if (isAgentStepping) {
      if (progressHideTimeoutRef.current) {
        window.clearTimeout(progressHideTimeoutRef.current);
        progressHideTimeoutRef.current = null;
      }
      if (progressIntervalRef.current) {
        window.clearInterval(progressIntervalRef.current);
      }
      setThinkingProgress(8);
      progressIntervalRef.current = window.setInterval(() => {
        setThinkingProgress((prev) => {
          // Asymptotic ease toward 90% — visually keeps moving, never stalls.
          const next = prev + (90 - prev) * 0.06;
          return next > 90 ? 90 : next;
        });
      }, 180);
      return;
    }

    if (progressIntervalRef.current) {
      window.clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }

    // Step finished — only flip to 100 if a step was actually in flight.
    setThinkingProgress((prev) => (prev > 0 ? 100 : 0));
    progressHideTimeoutRef.current = window.setTimeout(() => {
      setThinkingProgress(0);
      progressHideTimeoutRef.current = null;
    }, 500);
  }, [isAgentStepping]);

  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) window.clearInterval(progressIntervalRef.current);
      if (progressHideTimeoutRef.current) window.clearTimeout(progressHideTimeoutRef.current);
    };
  }, []);

  const graphQuery = useQuery({
    queryKey: ["environment-graph", sessionId],
    queryFn: () => apiClient.getEnvironmentGraph(sessionId ?? undefined),
    enabled: !state,
  });

  // Poll model status until it's ready so the Agent Step button can light up
  // automatically once the background loader thread finishes.
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: (query) => (query.state.data?.model_loaded ? false : 1500),
    refetchOnWindowFocus: false,
  });
  const modelLoaded = healthQuery.data?.model_loaded ?? false;
  const modelLoadState = healthQuery.data?.model_load_state ?? "idle";
  const modelLoading = !modelLoaded && (modelLoadState === "loading" || modelLoadState === "idle");

  const availableTargets = currentNodes;
  const manualActionSpec = findManualAction(manualAction);
  const targetKind = manualActionSpec.targetKind;
  const selectedManualNode =
    targetKind === "node" || targetKind === "two_nodes"
      ? availableTargets.find((node) => node.id === manualTarget) ?? null
      : null;
  const manualActionGuard = getActionGuard(manualActionSpec, manualTarget, manualTarget2, selectedManualNode);

  const handleManualActionChange = (next: string) => {
    setManualAction(next);
    setManualTarget(null);
    setManualTarget2(null);
  };

  const submitManualAction = () => {
    if (manualActionGuard.disabled) return;
    if (targetKind === "none") {
      void stepEpisode(manualActionSpec.id, null, null);
    } else if (targetKind === "sector") {
      void stepEpisode(manualActionSpec.id, manualTarget, null);
    } else if (targetKind === "two_sectors" || targetKind === "two_nodes") {
      void stepEpisode(manualActionSpec.id, manualTarget, manualTarget2);
    } else {
      void stepEpisode(manualActionSpec.id, manualTarget, null);
    }
  };

  // Agent + auto-run controls must be locked until the reset has produced
  // a session AND nodes are loaded. Pending here covers any in-flight call.
  // We additionally gate on modelLoaded so the user can't fire an agent step
  // while the local Qwen model is still being loaded in the background.
  const episodeReady = !!sessionId && !isResetting && currentNodes.length > 0;
  const agentDisabled = !episodeReady || pending || !!state?.done || !modelLoaded;
  const autoRunDisabled = !episodeReady || !!state?.done || !modelLoaded;
  const startDisabled = isResetting || pending;

  const handleResetClick = () => {
    void resetEpisode(config.task, config.seed);
  };

  const handleAgentStepClick = () => {
    void runAgentStep();
  };

  const handleDownloadReport = () => {
    if (!state || !sessionId) return;
    const report = {
      generated_at: new Date().toISOString(),
      session_id: sessionId,
      task: state.task,
      seed: config.seed,
      status: state.done ? "COMPLETED" : "IN_PROGRESS",
      summary: {
        episode_score: state.episode_score,
        steps_taken: state.step,
        max_steps: state.max_steps,
        budget_remaining: state.budget,
        cascade_depth: state.cascade_depth,
      },
      sector_health: state.sector_health,
      last_step_result: lastStepResult ?? null,
      grader_breakdown: lastStepResult?.grader_breakdown ?? null,
      agent_thinking: agentThinking || null,
      nodes: currentNodes.map((node) => ({
        id: node.id,
        label: node.label,
        sector: node.sector,
        status: node.status,
        health_pct: node.health_pct,
        is_operational: node.is_operational,
        is_hardened: node.is_hardened,
      })),
      edges: currentEdges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        kind: edge.kind,
        status: edge.status,
      })),
      event_log: logs.map((line) => ({
        timestamp: line.timestamp,
        level: line.level,
        message: line.message,
      })),
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `cascadeguard_episode_${state.task}_${sessionId.slice(0, 8)}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

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

            <div className="grid grid-cols-[1fr_auto] gap-2">
              <Button
                className="h-11 w-full bg-[hsl(var(--ink))] text-white hover:bg-[hsl(var(--ink))]/90"
                onClick={handleResetClick}
                disabled={startDisabled}
              >
                <Play className="h-4 w-4" />
                {isResetting ? "Starting..." : "Start Episode"}
              </Button>
              <Button
                variant="outline"
                size="icon"
                className="h-11 w-11 border-black/10 hover:bg-black/[0.04]"
                onClick={handleResetClick}
                disabled={startDisabled}
                aria-label="Reset"
                title="Reset episode"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
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
                    <div className="text-3xl font-bold tracking-tight">{state.episode_score.toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/40">Budget</div>
                    <div className="text-3xl font-bold tracking-tight">{state.budget.toFixed(1)}</div>
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
            <Button
              className="h-11 w-full bg-[hsl(var(--accent))] text-white hover:bg-[hsl(var(--accent))]/90 disabled:opacity-50"
              onClick={handleAgentStepClick}
              disabled={agentDisabled}
            >
              {modelLoading ? (
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Model Loading...
                </span>
              ) : modelLoadState === "error" ? (
                "Model Failed to Load"
              ) : isAgentStepping ? (
                "Agent thinking..."
              ) : (
                "Agent Step"
              )}
            </Button>
            {isAgentThinking ? (
              <div className="space-y-1.5 rounded-2xl border border-black/8 bg-black/[0.03] px-4 py-3">
                <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/55">
                  <span>Agent thinking...</span>
                  <span>{Math.round(thinkingProgress)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-black/8">
                  <div
                    className="h-full rounded-full bg-[hsl(var(--accent))] transition-[width] duration-300 ease-out"
                    style={{ width: `${thinkingProgress}%` }}
                  />
                </div>
              </div>
            ) : null}
            <div className="flex items-center justify-between rounded-2xl border border-black/8 bg-black/[0.03] px-4 py-3">
              <div>
                <div className="text-sm font-semibold tracking-tight">Auto-Run</div>
                <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/50">
                  Runs steps until episode ends
                </div>
              </div>
              <Switch checked={autoRun} onCheckedChange={setAutoRun} disabled={autoRunDisabled} />
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
              nodes={currentNodes}
              edges={currentEdges}
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
                <Select value={manualAction} onValueChange={handleManualActionChange}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select action" />
                  </SelectTrigger>
                  <SelectContent>
                    {manualActions.map((action) => (
                      <SelectItem key={action.id} value={action.id}>
                        <div className="flex items-baseline gap-2">
                          <span className="font-mono text-[12px]">{action.label}</span>
                          {action.cost ? (
                            <span className="font-mono text-[10px] text-foreground/50">{action.cost}</span>
                          ) : null}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="font-mono text-[10px] leading-relaxed text-foreground/55">
                  {manualActionSpec.description}
                </p>
              </div>

              {targetKind === "none" ? (
                <div className="rounded-2xl border border-black/8 bg-black/[0.03] px-3 py-2 font-mono text-[11px] text-foreground/55">
                  This action takes no target.
                </div>
              ) : null}

              {targetKind === "node" ? (
                <div className="space-y-2">
                  <Label>Target Node</Label>
                  <Select value={manualTarget ?? ""} onValueChange={setManualTarget}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select target node" />
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
              ) : null}

              {targetKind === "two_nodes" ? (
                <>
                  <div className="space-y-2">
                    <Label>Source Node (A)</Label>
                    <Select value={manualTarget ?? ""} onValueChange={setManualTarget}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select source node" />
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
                  <div className="space-y-2">
                    <Label>Target Node (B)</Label>
                    <Select value={manualTarget2 ?? ""} onValueChange={setManualTarget2}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select target node" />
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
                </>
              ) : null}

              {targetKind === "sector" ? (
                <div className="space-y-2">
                  <Label>Sector</Label>
                  <Select value={manualTarget ?? ""} onValueChange={setManualTarget}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select sector" />
                    </SelectTrigger>
                    <SelectContent>
                      {SECTOR_OPTIONS.map((sector) => (
                        <SelectItem key={sector} value={sector}>
                          {sector}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : null}

              {targetKind === "two_sectors" ? (
                <>
                  <div className="space-y-2">
                    <Label>Sector A</Label>
                    <Select value={manualTarget ?? ""} onValueChange={setManualTarget}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select sector A" />
                      </SelectTrigger>
                      <SelectContent>
                        {SECTOR_OPTIONS.map((sector) => (
                          <SelectItem key={sector} value={sector}>
                            {sector}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Sector B</Label>
                    <Select value={manualTarget2 ?? ""} onValueChange={setManualTarget2}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select sector B" />
                      </SelectTrigger>
                      <SelectContent>
                        {SECTOR_OPTIONS.map((sector) => (
                          <SelectItem key={sector} value={sector}>
                            {sector}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </>
              ) : null}

              {manualActionGuard.message ? (
                <div className="rounded-2xl border border-[hsl(var(--amber))]/25 bg-[hsl(var(--amber))]/10 px-3 py-2 text-sm text-[hsl(var(--amber))]">
                  {manualActionGuard.message}
                </div>
              ) : null}
              <Button
                className="h-11 w-full bg-[hsl(var(--accent))] text-white hover:bg-[hsl(var(--accent))]/90 disabled:opacity-50"
                onClick={submitManualAction}
                disabled={!episodeReady || pending || manualActionGuard.disabled}
              >
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

      <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Episode Report</div>
            <div className="mt-1 text-lg font-semibold tracking-tight">
              Download Full Episode Report
            </div>
            <div className="mt-1 text-sm text-foreground/60">
              Snapshots stats, sector health, grader breakdown, node graph, and the full event log
              into a single JSON report.
            </div>
          </div>
          <Button
            className="h-11 bg-[hsl(var(--accent))] px-5 text-white hover:bg-[hsl(var(--accent))]/90 disabled:opacity-50"
            onClick={handleDownloadReport}
            disabled={!state || !sessionId}
          >
            <Download className="h-4 w-4" />
            Download Report
          </Button>
        </div>
      </Card>
    </div>
  );
}

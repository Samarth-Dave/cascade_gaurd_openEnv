import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { apiClient } from "@/api/client";
import type {
  AgentAction,
  EpisodeResetResponse,
  EpisodeState,
  InfraNode,
  StepResult,
  Task,
  WsEvent,
} from "@/types";
import { useEpisodeStream } from "@/hooks/useEpisodeStream";
import type { LogLine } from "@/components/cascade/TerminalLog";

interface CascadeContextValue {
  config: { task: Task; seed: number };
  setConfig: (next: { task: Task; seed: number }) => void;
  sessionId: string | null;
  state: EpisodeState | null;
  currentNodes: InfraNode[];
  currentEdges: EpisodeState["edges"];
  lastStepResult: StepResult | null;
  logs: LogLine[];
  events: WsEvent[];
  agentThinking: string;
  agentAction: AgentAction | null;
  autoRun: boolean;
  setAutoRun: (value: boolean) => void;
  pending: boolean;
  error: string | null;
  isConnected: boolean;
  resetEpisode: (task: Task, seed: number) => Promise<void>;
  stepEpisode: (action: string, target: string | null) => Promise<void>;
  runAgentStep: () => Promise<void>;
}

const CascadeContext = createContext<CascadeContextValue | null>(null);

export function CascadeProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [config, setConfig] = useState<{ task: Task; seed: number }>({ task: "task_hard", seed: 42 });
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [state, setState] = useState<EpisodeState | null>(null);
  const [currentNodes, setCurrentNodes] = useState<InfraNode[]>([]);
  const [currentEdges, setCurrentEdges] = useState<EpisodeState["edges"]>([]);
  const [lastStepResult, setLastStepResult] = useState<StepResult | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [agentThinking, setAgentThinking] = useState("");
  const [agentAction, setAgentAction] = useState<AgentAction | null>(null);
  const [autoRun, setAutoRun] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stream = useEpisodeStream(sessionId, state?.done ?? false);

  useEffect(() => {
    if (!stream.lastEvent) return;
    appendLog(stream.lastEvent.level, stream.lastEvent.message, stream.lastEvent.timestamp);
    if (stream.lastEvent.type === "step_result") {
      const payload = stream.lastEvent.payload as { result?: StepResult; state?: EpisodeState };
      if (payload.result?.new_state) {
        const nextState = hydrateStepState(payload.result);
        setState(nextState);
        setCurrentNodes(nextState.nodes);
        setCurrentEdges(nextState.edges);
      } else if (payload.state) {
        setState(payload.state);
        setCurrentNodes(payload.state.nodes);
        setCurrentEdges(payload.state.edges);
      }
    }
  }, [stream.lastEvent]);

  useEffect(() => {
    if (!autoRun || !sessionId || !state || state.done || pending) return;
    const timer = window.setTimeout(() => {
      void runAgentStep();
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [autoRun, pending, sessionId, state]);

  function appendLog(level: LogLine["level"], message: string, timestamp = new Date().toISOString()) {
    const formattedTime = new Date(timestamp).toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    setLogs((current) => [
      ...current,
      {
        id: `${timestamp}-${current.length}`,
        timestamp: formattedTime,
        level,
        message,
      },
    ].slice(-200));
  }

  async function resetEpisode(task: Task, seed: number) {
    setPending(true);
    setError(null);
    setAutoRun(false);
    try {
      const response: EpisodeResetResponse = await apiClient.resetEpisode(task, seed);
      setConfig({ task, seed });
      setSessionId(response.session_id);
      setState(response.state);
      setCurrentNodes(response.nodes.length > 0 ? response.nodes : response.state.nodes);
      setCurrentEdges(response.edges.length > 0 ? response.edges : response.state.edges);
      setLastStepResult(null);
      setAgentThinking("");
      setAgentAction(null);
      setLogs([]);
      appendLog("INFO", `Episode started for ${task} with seed ${seed}.`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Failed to start episode.");
    } finally {
      setPending(false);
    }
  }

  async function stepEpisode(action: string, target: string | null) {
    if (!sessionId) return;
    setPending(true);
    setError(null);
    try {
      const result = await apiClient.stepEpisode(sessionId, action, target);
      handleStepResult(result, `Manual action ${action}${target ? ` -> ${target}` : ""}`, action, target);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Action failed.");
    } finally {
      setPending(false);
    }
  }

  async function runAgentStep() {
    if (!sessionId || pending || state?.done) return;
    setPending(true);
    setError(null);
    try {
      const result = await apiClient.agentStep(sessionId);
      handleStepResult(result, "Agent step executed.", result.agent_action?.action ?? null, result.agent_action?.target ?? null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Agent step failed.");
      setAutoRun(false);
    } finally {
      setPending(false);
    }
  }

  function handleStepResult(
    result: StepResult,
    message: string,
    actionLabel: string | null,
    actionTarget: string | null,
  ) {
    const nextState = hydrateStepState(result);
    const enrichedResult = {
      ...result,
      new_state: nextState,
      nodes: nextState.nodes,
      edges: nextState.edges,
    };
    const previousState = state;
    setLastStepResult(enrichedResult);
    setState(nextState);
    setCurrentNodes(nextState.nodes);
    setCurrentEdges(nextState.edges);
    setAgentThinking(result.agent_thinking ?? "");
    setAgentAction(result.agent_action ?? null);
    appendLog(result.done ? "OK" : "INFO", message);
    const feedback = buildActionFeedbackEntry(previousState, enrichedResult, actionLabel, actionTarget);
    appendLog(feedback.level, feedback.message);
    if (result.done) {
      setAutoRun(false);
      void queryClient.invalidateQueries({ queryKey: ["episode-history"] });
    }
  }

  const value = useMemo<CascadeContextValue>(
    () => ({
      config,
      setConfig,
      sessionId,
      state,
      currentNodes,
      currentEdges,
      lastStepResult,
      logs,
      events: stream.events,
      agentThinking,
      agentAction,
      autoRun,
      setAutoRun,
      pending,
      error,
      isConnected: stream.isConnected,
      resetEpisode,
      stepEpisode,
      runAgentStep,
    }),
    [agentAction, agentThinking, autoRun, config, currentEdges, currentNodes, error, lastStepResult, logs, pending, sessionId, state, stream.events, stream.isConnected],
  );

  return <CascadeContext.Provider value={value}>{children}</CascadeContext.Provider>;
}

function hydrateStepState(result: StepResult): EpisodeState {
  return {
    ...result.new_state,
    done: result.done || result.new_state.done,
    nodes: result.nodes.length > 0 ? result.nodes : result.new_state.nodes,
    edges: result.edges.length > 0 ? result.edges : result.new_state.edges,
  };
}

function findNode(nodes: InfraNode[], nodeId: string | null): InfraNode | null {
  if (!nodeId) {
    return null;
  }
  return nodes.find((node) => node.id === nodeId) ?? null;
}

function formatHealth(node: InfraNode | null): string {
  if (!node) {
    return "n/a";
  }
  return `${node.health_pct}%`;
}

function appendReasoningSuffix(
  beforeNode: InfraNode | null,
  afterNode: InfraNode | null,
  cascadeChanged: boolean,
  rewardStep: number,
): string {
  if (beforeNode && beforeNode.is_operational && beforeNode.health_pct > 70 && afterNode && beforeNode.health_pct === afterNode.health_pct && rewardStep <= 0.05) {
    return " Target was already healthy and operational, so the action had little effect.";
  }
  if (beforeNode && beforeNode.is_operational && afterNode && beforeNode.health_pct === afterNode.health_pct && !cascadeChanged && rewardStep <= 0.05) {
    return " Node state and cascade depth barely changed, so reward stayed low.";
  }
  if (!cascadeChanged && rewardStep <= 0.05) {
    return " Cascade depth did not improve, which limited the reward.";
  }
  return "";
}

function buildActionFeedbackMessage(
  previousState: EpisodeState | null,
  result: StepResult,
  actionLabel: string | null,
  actionTarget: string | null,
): string {
  const resolvedAction = actionLabel ?? result.agent_action?.action ?? "UNKNOWN";
  const resolvedTarget = actionTarget ?? result.agent_action?.target ?? null;
  const beforeNode = previousState ? findNode(previousState.nodes, resolvedTarget) : null;
  const afterNode = findNode(result.new_state.nodes, resolvedTarget);
  const previousCascade = previousState?.cascade_depth ?? result.new_state.cascade_depth;
  const nextCascade = result.new_state.cascade_depth;
  const cascadeChanged = previousCascade !== nextCascade;
  const cascadePart =
    previousState === null
      ? `cascade ${nextCascade}`
      : `cascade ${previousCascade} -> ${nextCascade}`;
  return (
    `${resolvedAction}${resolvedTarget ? ` ${resolvedTarget}` : ""} | ` +
    `health ${formatHealth(beforeNode)} -> ${formatHealth(afterNode)} | ` +
    `reward ${result.reward_step >= 0 ? "+" : ""}${result.reward_step.toFixed(3)} | ` +
    `${cascadePart}.` +
    appendReasoningSuffix(beforeNode, afterNode, cascadeChanged, result.reward_step)
  );
}

function buildActionFeedbackEntry(
  previousState: EpisodeState | null,
  result: StepResult,
  actionLabel: string | null,
  actionTarget: string | null,
) {
  const message = buildActionFeedbackMessage(previousState, result, actionLabel, actionTarget);
  const level = result.reward_step <= 0.05 ? "WARN" : result.reward_step > 0 ? "OK" : "CRIT";
  return { level, message };
}

export function useCascade() {
  const context = useContext(CascadeContext);
  if (!context) {
    throw new Error("useCascade must be used inside CascadeProvider.");
  }
  return context;
}

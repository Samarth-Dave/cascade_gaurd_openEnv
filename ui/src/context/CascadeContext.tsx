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
        setState(payload.result.new_state);
      } else if (payload.state) {
        setState(payload.state);
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
      handleStepResult(result, `Manual action ${action}${target ? ` -> ${target}` : ""}`);
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
      handleStepResult(result, "Agent step executed.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Agent step failed.");
      setAutoRun(false);
    } finally {
      setPending(false);
    }
  }

  function handleStepResult(result: StepResult, message: string) {
    setLastStepResult(result);
    setState(result.new_state);
    setAgentThinking(result.agent_thinking ?? "");
    setAgentAction(result.agent_action ?? null);
    appendLog(result.done ? "OK" : "INFO", message);
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
    [agentAction, agentThinking, autoRun, config, error, lastStepResult, logs, pending, sessionId, state, stream.events, stream.isConnected],
  );

  return <CascadeContext.Provider value={value}>{children}</CascadeContext.Provider>;
}

export function useCascade() {
  const context = useContext(CascadeContext);
  if (!context) {
    throw new Error("useCascade must be used inside CascadeProvider.");
  }
  return context;
}

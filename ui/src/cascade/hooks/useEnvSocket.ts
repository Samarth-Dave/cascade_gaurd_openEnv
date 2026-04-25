import { useCallback, useEffect, useRef, useState } from 'react';
import type { CGObservation } from '../data/types';
import { ENV_BASE, ENV_WS } from '../config/env';
import { normalizeObservation } from '../lib/liveObservation';
import { apiClient } from '@/api/client';

const SOCKET_DEBUG = import.meta.env.DEV;

const socketLog = (...args: unknown[]) => {
  if (!SOCKET_DEBUG) return;
  console.debug('[useEnvSocket]', ...args);
};

export type ConnState = 'connecting' | 'live' | 'sim' | 'error';

export interface UseEnvSocket {
  state: ConnState;
  lastObs: CGObservation | null;
  lastError: string | null;
  send: (
    action_type: string,
    target_node_id: string | null,
    parameters?: Record<string, unknown>,
  ) => void;
  reset: () => void;
  reconnect: (taskId: string) => void;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const toObservation = (message: Record<string, unknown>): CGObservation | null => {
  const data = isRecord(message.data) ? message.data : message;
  const rawObs = isRecord(data.observation) ? data.observation : data;
  if (!isRecord(rawObs)) return null;

  const obs: Record<string, unknown> = { ...rawObs };
  if (obs.reward == null && typeof data.reward === 'number') obs.reward = data.reward;
  if (obs.done == null && typeof data.done === 'boolean') obs.done = data.done;
  if (!Array.isArray(obs.active_failures)) obs.active_failures = [];
  if (!Array.isArray(obs.pending_recoveries)) obs.pending_recoveries = [];
  if (!Array.isArray(obs.edges)) obs.edges = [];
  if (!Array.isArray(obs.nodes)) obs.nodes = [];
  if (!isRecord(obs.sector_summary)) {
    obs.sector_summary = { power: 1, water: 1, hospital: 1, telecom: 1 };
  }
  if (!isRecord(obs.diagnostics)) {
    obs.diagnostics = {
      critical_failures: [],
      at_risk_nodes: [],
      dependency_alerts: [],
      recommended_recovery_order: [],
      system_pressure: 0,
    };
  }
  if (typeof obs.reward !== 'number') obs.reward = 0;
  if (typeof obs.done !== 'boolean') obs.done = false;
  if (typeof obs.step !== 'number') obs.step = 0;
  if (typeof obs.max_steps !== 'number') obs.max_steps = 10;
  if (typeof obs.budget_remaining !== 'number') obs.budget_remaining = 0;

  return normalizeObservation(obs as CGObservation);
};

export function useEnvSocket(
  taskId: string,
  enabled = true,
  onObservation?: (obs: CGObservation) => void,
  onError?: (message: string) => void,
): UseEnvSocket {
  const [state, setState] = useState<ConnState>(enabled ? 'connecting' : 'sim');
  const [lastObs, setLastObs] = useState<CGObservation | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const wsConnectingRef = useRef(false);
  const taskRef = useRef(taskId);
  const obsRef = useRef(onObservation);
  const errorRef = useRef(onError);
  obsRef.current = onObservation;
  errorRef.current = onError;

  const emitError = useCallback((message: string) => {
    socketLog('error', message);
    setLastError(message);
    errorRef.current?.(message);
  }, []);

  const cleanup = useCallback(() => {
    const ws = wsRef.current;
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.close();
    }
    wsRef.current = null;
    wsConnectingRef.current = false;
  }, []);

  const open = useCallback((task: string) => {
    if (wsConnectingRef.current) return;
    taskRef.current = task;
    cleanup();
    wsConnectingRef.current = true;
    setState('connecting');
    setLastError(null);
    let settled = false;

    const fallbackToSim = (message: string, ws: WebSocket) => {
      if (wsRef.current !== ws) return;
      settled = true;
      wsConnectingRef.current = false;
      wsRef.current = null;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.close();
      setLastObs(null);
      setState('sim');
      emitError(message);
    };

    let ws: WebSocket;
    try {
      ws = apiClient.createSocket(ENV_WS);
    } catch (error) {
      wsConnectingRef.current = false;
      setLastObs(null);
      setState('sim');
      emitError(`Could not connect to ${ENV_WS}: ${String(error)}`);
      return;
    }

    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current !== ws) return;
      settled = true;
      wsConnectingRef.current = false;
      setState('live');
      socketLog('connected', { ws: ENV_WS, task });
      try {
        ws.send(JSON.stringify({ type: 'reset', data: { task_id: task } }));
      } catch (error) {
        fallbackToSim(`Reset send failed: ${String(error)}`, ws);
      }
    };

    ws.onmessage = async (event) => {
      if (wsRef.current !== ws) return;

      const raw =
        typeof event.data === 'string'
          ? event.data
          : event.data instanceof Blob
            ? await event.data.text()
            : String(event.data);
      socketLog('raw message', raw.slice(0, 900));

      let msg: unknown;
      try {
        msg = JSON.parse(raw);
      } catch {
        emitError('WS parse error');
        return;
      }
      if (!isRecord(msg)) return;
      if (msg.type === 'error') {
        emitError(`[ENV ERROR] ${JSON.stringify(msg.data ?? {})}`);
        return;
      }

      const obs = toObservation(msg);
      if (!obs) return;
      socketLog('observation', {
        step: obs.step,
        max_steps: obs.max_steps,
        nodes: obs.nodes.length,
        nodes_with_coords: obs.nodes.filter((node) => Number.isFinite(node.lat) && Number.isFinite(node.lon)).length,
        active_failures: obs.active_failures.length,
        reward: obs.reward,
      });
      setLastObs(obs);
      obsRef.current?.(obs);
    };

    ws.onerror = () => {
      fallbackToSim('WebSocket error - falling back to simulation', ws);
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      wsConnectingRef.current = false;
      wsRef.current = null;
      if (!settled) {
        settled = true;
        setLastObs(null);
        setState('sim');
        emitError(`Disconnected while connecting to ${ENV_WS} - using simulation`);
        return;
      }
      setState('error');
      emitError(`Disconnected from OpenEnv at ${ENV_BASE}`);
    };
  }, [cleanup, emitError]);

  useEffect(() => {
    if (!enabled) {
      cleanup();
      setState('sim');
      setLastObs(null);
      setLastError(null);
      return;
    }

    taskRef.current = taskId;
    open(taskId);
    return () => cleanup();
  }, [cleanup, enabled, open, taskId]);

  const send = useCallback((
    action_type: string,
    target_node_id: string | null,
    parameters: Record<string, unknown> = {},
  ) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      emitError('Action not sent: OpenEnv socket is not connected');
      return;
    }
    try {
      ws.send(JSON.stringify({ type: 'step', data: { action_type, target_node_id, parameters } }));
      socketLog('sent action', { action_type, target_node_id, parameters });
    } catch (error) {
      emitError(`Action send failed: ${String(error)}`);
    }
  }, [emitError]);

  const reset = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      emitError('Reset not sent: OpenEnv socket is not connected');
      return;
    }
    try {
      ws.send(JSON.stringify({ type: 'reset', data: { task_id: taskRef.current } }));
      socketLog('sent reset', { task_id: taskRef.current });
    } catch (error) {
      emitError(`Reset send failed: ${String(error)}`);
    }
  }, [emitError]);

  const reconnect = useCallback((task: string) => {
    if (!enabled) return;
    open(task);
  }, [enabled, open]);

  return { state, lastObs, lastError, send, reset, reconnect };
}

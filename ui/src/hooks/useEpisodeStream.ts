import { useEffect, useRef, useState } from "react";

import { apiClient } from "@/api/client";
import type { EpisodeState, StepResult, WsEvent } from "@/types";

export interface EpisodeStreamStepPayload {
  result?: StepResult;
  state?: EpisodeState;
}

const RECONNECT_BASE_DELAY_MS = 750;
const RECONNECT_MAX_DELAY_MS = 10_000;
const MAX_EVENTS_KEPT = 100;
// 4404 is emitted by the backend when the session_id is unknown. Don't
// keep trying to reconnect in that case — the UI should reset the session.
const NON_RETRYABLE_CLOSE_CODES = new Set([4404, 4001, 4403]);

export function useEpisodeStream(
  sessionId?: string | null,
  done = false,
  onStepResult?: (payload: EpisodeStreamStepPayload) => void,
) {
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [lastEvent, setLastEvent] = useState<WsEvent | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Keep the latest callback in a ref so the effect below never restarts
  // just because the parent re-rendered with a new onStepResult closure.
  const onStepResultRef = useRef(onStepResult);
  useEffect(() => {
    onStepResultRef.current = onStepResult;
  }, [onStepResult]);

  useEffect(() => {
    if (!sessionId || done) {
      setIsConnected(false);
      return undefined;
    }

    let socket: WebSocket | null = null;
    let cancelled = false;
    let attempt = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const scheduleReconnect = () => {
      if (cancelled) return;
      attempt += 1;
      const delay = Math.min(
        RECONNECT_MAX_DELAY_MS,
        RECONNECT_BASE_DELAY_MS * 2 ** Math.max(0, attempt - 1),
      );
      clearReconnectTimer();
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        if (!cancelled) connect();
      }, delay);
    };

    const connect = () => {
      if (cancelled) return;
      try {
        socket = apiClient.createEpisodeSocket(sessionId);
      } catch {
        scheduleReconnect();
        return;
      }

      socket.onopen = () => {
        if (cancelled) return;
        attempt = 0;
        setIsConnected(true);
      };

      socket.onmessage = (event) => {
        if (cancelled) return;
        let parsed: WsEvent | null = null;
        try {
          parsed = JSON.parse(event.data) as WsEvent;
        } catch (err) {
          // Skip malformed frames — a single bad message must not kill the
          // stream. The backend only sends JSON, so this would be a server
          // or proxy bug.
          // eslint-disable-next-line no-console
          console.warn("[useEpisodeStream] failed to parse WS message", err);
          return;
        }
        if (!parsed) return;
        if (parsed.type === "step_result") {
          onStepResultRef.current?.(parsed.payload as EpisodeStreamStepPayload);
        }
        setLastEvent(parsed);
        setEvents((current) => [parsed as WsEvent, ...current].slice(0, MAX_EVENTS_KEPT));
      };

      socket.onerror = () => {
        if (cancelled) return;
        setIsConnected(false);
      };

      socket.onclose = (event) => {
        if (cancelled) return;
        setIsConnected(false);
        socket = null;
        if (done) return;
        if (NON_RETRYABLE_CLOSE_CODES.has(event.code)) return;
        scheduleReconnect();
      };
    };

    connect();

    return () => {
      cancelled = true;
      clearReconnectTimer();
      if (socket) {
        try {
          socket.close();
        } catch {
          // Ignore — socket already closing/closed.
        }
      }
      setIsConnected(false);
    };
  }, [done, sessionId]);

  return { events, lastEvent, isConnected };
}

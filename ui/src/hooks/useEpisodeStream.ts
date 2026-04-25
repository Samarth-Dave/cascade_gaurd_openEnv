import { useEffect, useState } from "react";

import { apiClient } from "@/api/client";
import type { EpisodeState, StepResult, WsEvent } from "@/types";

export interface EpisodeStreamStepPayload {
  result?: StepResult;
  state?: EpisodeState;
}

export function useEpisodeStream(
  sessionId?: string | null,
  done = false,
  onStepResult?: (payload: EpisodeStreamStepPayload) => void,
) {
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [lastEvent, setLastEvent] = useState<WsEvent | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!sessionId || done) {
      setIsConnected(false);
      return undefined;
    }

    const socket = apiClient.createEpisodeSocket(sessionId);

    socket.onopen = () => {
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      const parsed = JSON.parse(event.data) as WsEvent;
      if (parsed.type === "step_result") {
        onStepResult?.(parsed.payload as EpisodeStreamStepPayload);
      }
      setLastEvent(parsed);
      setEvents((current) => [parsed, ...current].slice(0, 100));
    };

    socket.onerror = () => {
      setIsConnected(false);
    };

    socket.onclose = () => {
      setIsConnected(false);
    };

    return () => {
      socket.close();
    };
  }, [done, onStepResult, sessionId]);

  return { events, lastEvent, isConnected };
}

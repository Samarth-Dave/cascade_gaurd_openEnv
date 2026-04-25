import type {
  BaselineComparison,
  DatasetRow,
  DatasetStats,
  EpisodeHistoryItem,
  EpisodeResetResponse,
  EpisodeState,
  GraphResponse,
  HealthResponse,
  RewardCurvePoint,
  StepResult,
  Task,
} from "@/types";

const trim = (value: string | undefined) => (value ?? "").trim().replace(/\/+$/, "");

// API_BASE_URL = "" in production (single HF Space) -> fetch uses relative URLs
// like "/api/health", which resolve to the same origin that served the UI.
// In local dev set VITE_API_URL=http://localhost:7860 to point at the backend.
const API_BASE_URL = trim(import.meta.env.VITE_API_URL);

// WS_BASE_URL: explicit override > derive from API base > derive from current origin.
function deriveWsBase(): string {
  const explicit = trim(import.meta.env.VITE_WS_URL);
  if (explicit) return explicit;
  if (API_BASE_URL) return API_BASE_URL.replace(/^http/i, "ws");
  if (typeof window !== "undefined" && window.location) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}`;
  }
  return "";
}
const WS_BASE_URL = deriveWsBase();

const pathForApi = (path: string) =>
  API_BASE_URL.endsWith("/api") && path.startsWith("/api/")
    ? path.slice(4)
    : path;

type RequestInitWithJson = RequestInit & {
  json?: unknown;
};

async function request<T>(path: string, init: RequestInitWithJson = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.json !== undefined) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
    body: init.json !== undefined ? JSON.stringify(init.json) : init.body,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }
  return (await response.text()) as T;
}

async function requestText(path: string): Promise<string> {
  return request<string>(path);
}

async function requestBlob(path: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.blob();
}

export const apiClient = {
  baseUrl: API_BASE_URL,
  wsUrl: WS_BASE_URL,
  createSocket: (url: string) => new WebSocket(url),

  getHealth: () => request<HealthResponse>("/api/health"),

  resetEpisode: (task: Task, seed: number) =>
    request<EpisodeResetResponse>("/api/episode/reset", {
      method: "POST",
      json: { task, seed },
    }),

  stepEpisode: (
    sessionId: string,
    action: string,
    target: string | null,
    target2: string | null = null,
  ) =>
    request<StepResult>(pathForApi("/api/episode/step"), {
      method: "POST",
      json: { session_id: sessionId, action, target, target2 },
    }),

  agentStep: (sessionId: string) =>
    request<StepResult>("/api/episode/agent-step", {
      method: "POST",
      json: { session_id: sessionId },
    }),

  getEpisodeState: (sessionId: string) =>
    request<EpisodeState>(`/api/episode/state/${sessionId}`),

  getEpisodeHistory: () => request<EpisodeHistoryItem[]>("/api/episodes/history"),

  getRewardCurve: () => request<RewardCurvePoint[]>("/api/training/reward-curve"),

  getBaselineComparison: () =>
    request<BaselineComparison>("/api/training/baseline-comparison"),

  getEnvironmentGraph: (sessionId?: string) =>
    request<GraphResponse>(
      `/api/environment/graph${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`,
    ),

  getDatasetStats: () => request<DatasetStats>("/api/dataset/sft/stats"),

  getDatasetRows: async () => parseCsv(await requestText("/api/dataset/sft?format=csv")),

  downloadDatasetCsv: async () => {
    const blob = await requestBlob("/api/dataset/sft?format=csv");
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "cascadeguard_sft_dataset.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  },

  createEpisodeSocket: (sessionId: string) =>
    new WebSocket(`${WS_BASE_URL}/ws/episode/${sessionId}`),
};

function parseCsv(text: string): DatasetRow[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        cell += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(cell);
      cell = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        index += 1;
      }
      row.push(cell);
      if (row.some((value) => value.length > 0)) {
        rows.push(row);
      }
      row = [];
      cell = "";
      continue;
    }

    cell += char;
  }

  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }

  const [headerRow, ...dataRows] = rows;
  if (!headerRow) return [];

  return dataRows.map((dataRow) => {
    const entry = {} as DatasetRow;
    headerRow.forEach((header, headerIndex) => {
      entry[header as keyof DatasetRow] = (dataRow[headerIndex] ?? "") as never;
    });
    return entry;
  });
}

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

// If VITE_API_URL already ends with /api (some deployments expose the backend
// at /api on a reverse proxy), strip the leading /api from our paths to avoid
// hitting /api/api/foo.
const pathForApi = (path: string) =>
  API_BASE_URL.endsWith("/api") && path.startsWith("/api/")
    ? path.slice(4)
    : path;

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

const DEFAULT_TIMEOUT_MS = 30_000;

type RequestInitWithJson = RequestInit & {
  json?: unknown;
  timeoutMs?: number;
};

async function fetchWithTimeout(input: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  // If the caller already supplied a signal, compose it with our timeout so
  // either source can abort the request.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException("Timeout", "TimeoutError")), timeoutMs);
  const upstream = init.signal;
  if (upstream) {
    if (upstream.aborted) controller.abort(upstream.reason);
    else upstream.addEventListener("abort", () => controller.abort(upstream.reason), { once: true });
  }
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function extractError(response: Response): Promise<ApiError> {
  const contentType = response.headers.get("content-type") ?? "";
  let detail: unknown = null;
  let message = "";
  try {
    if (contentType.includes("application/json")) {
      detail = await response.json();
      if (detail && typeof detail === "object" && "detail" in detail) {
        const maybeDetail = (detail as { detail?: unknown }).detail;
        message = typeof maybeDetail === "string" ? maybeDetail : JSON.stringify(maybeDetail);
      } else {
        message = JSON.stringify(detail);
      }
    } else {
      message = await response.text();
    }
  } catch {
    // Ignore — we'll fall back to a generic message.
  }
  if (!message) message = `Request failed with status ${response.status}`;
  return new ApiError(message, response.status, detail);
}

async function request<T>(path: string, init: RequestInitWithJson = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.json !== undefined) {
    headers.set("Content-Type", "application/json");
  }

  const resolvedPath = pathForApi(path);
  const timeoutMs = init.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  let response: Response;
  try {
    response = await fetchWithTimeout(
      `${API_BASE_URL}${resolvedPath}`,
      {
        ...init,
        headers,
        body: init.json !== undefined ? JSON.stringify(init.json) : init.body,
      },
      timeoutMs,
    );
  } catch (err) {
    if ((err as Error)?.name === "AbortError" || (err as Error)?.name === "TimeoutError") {
      throw new ApiError(`Request timed out after ${timeoutMs}ms`, 0, err);
    }
    throw new ApiError((err as Error)?.message ?? "Network error", 0, err);
  }

  if (!response.ok) throw await extractError(response);

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
  const resolvedPath = pathForApi(path);
  const response = await fetchWithTimeout(`${API_BASE_URL}${resolvedPath}`, {}, DEFAULT_TIMEOUT_MS);
  if (!response.ok) throw await extractError(response);
  return response.blob();
}

export const apiClient = {
  baseUrl: API_BASE_URL,
  wsUrl: WS_BASE_URL,
  createSocket: (url: string) => new WebSocket(url),

  getHealth: () => request<HealthResponse>("/api/health", { timeoutMs: 8_000 }),

  resetEpisode: (task: Task, seed: number) =>
    request<EpisodeResetResponse>("/api/episode/reset", {
      method: "POST",
      json: { task, seed },
      // Reset can take up to the env-server capacity wait (~12s) plus the
      // first WS handshake, so give it a little more headroom.
      timeoutMs: 45_000,
    }),

  stepEpisode: (
    sessionId: string,
    action: string,
    target: string | null,
    target2: string | null = null,
  ) =>
    request<StepResult>("/api/episode/step", {
      method: "POST",
      json: { session_id: sessionId, action, target, target2 },
    }),

  agentStep: (sessionId: string) =>
    request<StepResult>("/api/episode/agent-step", {
      method: "POST",
      json: { session_id: sessionId },
      timeoutMs: 60_000,
    }),

  getEpisodeState: (sessionId: string) =>
    request<EpisodeState>(`/api/episode/state/${sessionId}`),

  getEpisodeHistory: () => request<EpisodeHistoryItem[]>("/api/episodes/history"),

  getRewardCurve: () => request<RewardCurvePoint[]>("/api/training/reward-curve"),

  getBaselineComparison: () =>
    request<BaselineComparison>("/api/training/baseline-comparison"),

  getEnvironmentGraph: (sessionId?: string, city?: string) => {
    const params = new URLSearchParams();
    if (sessionId) params.set("session_id", sessionId);
    if (city) params.set("city", city);
    const qs = params.toString();
    return request<GraphResponse>(`/api/environment/graph${qs ? `?${qs}` : ""}`);
  },

  getDatasetStats: () => request<DatasetStats>("/api/dataset/sft/stats"),

  getDatasetRows: async () =>
    parseCsv(await requestText("/api/dataset/sft?format=csv")),

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

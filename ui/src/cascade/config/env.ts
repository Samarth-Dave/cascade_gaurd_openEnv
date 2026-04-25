const fromEnv = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
};

const apiBase = fromEnv(import.meta.env.VITE_API_URL) ?? fromEnv(import.meta.env.VITE_ENV_BASE);
const wsBase = fromEnv(import.meta.env.VITE_WS_URL) ?? fromEnv(import.meta.env.VITE_ENV_WS);

export const ENV_BASE = apiBase ?? window.location.origin;
export const ENV_WS = wsBase ?? ENV_BASE.replace(/^http/i, "ws");

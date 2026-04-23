const DEFAULT_ENV_BASE = 'http://localhost:8000';
const DEFAULT_ENV_WS = 'ws://localhost:8000/ws';

const fromEnv = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
};

export const ENV_BASE = fromEnv(import.meta.env.VITE_ENV_BASE) ?? DEFAULT_ENV_BASE;
export const ENV_WS = fromEnv(import.meta.env.VITE_ENV_WS) ?? DEFAULT_ENV_WS;

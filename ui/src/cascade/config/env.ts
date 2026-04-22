const fromEnv = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
};

const inferRuntimeEndpoints = () => {
  if (typeof window === 'undefined') {
    return {
      base: 'http://localhost:8000',
      ws: 'ws://localhost:8000/ws',
    };
  }

  const { hostname, origin, port, protocol, host } = window.location;
  const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';

  if (isLocal && port === '8080') {
    return {
      base: 'http://localhost:8000',
      ws: 'ws://localhost:8000/ws',
    };
  }

  const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
  return {
    base: origin,
    ws: `${wsProtocol}//${host}/ws`,
  };
};

const runtime = inferRuntimeEndpoints();

export const ENV_BASE = fromEnv(import.meta.env.VITE_ENV_BASE) ?? runtime.base;
export const ENV_WS = fromEnv(import.meta.env.VITE_ENV_WS) ?? runtime.ws;

import { useState, useEffect } from 'react';
import { CITIES, type City, type CityKey } from '../data/cities';
import { ENV_BASE, ENV_WS } from '../config/env';

interface TaskOption {
  id: string;
  label: string;
  city?: string;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const taskLabel = (task: Record<string, unknown>, id: string) => {
  const name = task.name ?? task.title ?? task.description;
  return typeof name === 'string' && name.trim().length > 0 ? `${id} - ${name.trim()}` : id;
};

const taskOptionsFromPayload = (payload: unknown): TaskOption[] => {
  const rawTasks = Array.isArray(payload)
    ? payload
    : isRecord(payload) && Array.isArray(payload.tasks)
      ? payload.tasks
      : isRecord(payload) && isRecord(payload.tasks)
        ? Object.entries(payload.tasks).map(([id, task]) => isRecord(task) ? { id, ...task } : id)
        : [];

  return rawTasks.reduce<TaskOption[]>((options, raw) => {
    if (typeof raw === 'string') {
      options.push({ id: raw, label: raw });
      return options;
    }
    if (!isRecord(raw)) return options;

    const id = raw.task_id ?? raw.taskId ?? raw.id ?? raw.key;
    if (typeof id !== 'string' || id.trim().length === 0) return options;
    const trimmedId = id.trim();
    const city = typeof raw.city === 'string' ? raw.city : undefined;
    options.push({ id: trimmedId, label: taskLabel(raw, trimmedId), city });
    return options;
  }, []);
};

interface Props {
  open: boolean;
  selected: CityKey;
  onClose: () => void;
  onSelect: (k: CityKey) => void;
  /** Called with the city key AND the chosen task id */
  onLaunch: (k: CityKey, taskId: string) => void;
}

export function CityModal({ open, selected, onClose, onSelect, onLaunch }: Props) {
  const sel: City = CITIES.find((city) => city.key === selected) ?? CITIES[0];
  const [tasks, setTasks] = useState<TaskOption[]>(() =>
    CITIES.map((city) => ({ id: city.taskId, label: city.taskId })),
  );

  // Local task state — resets to city default whenever selected city changes
  const [taskId, setTaskId] = useState<string>(sel.taskId);

  useEffect(() => {
    let cancelled = false;

    const loadTasks = async () => {
      try {
        const response = await fetch(`${ENV_BASE}/tasks`);
        if (!response.ok) return;

        const options = taskOptionsFromPayload(await response.json());
        // Filter to tasks that belong to the selected city, keeping the city's
        // default task even if its city field is absent (legacy task_* configs).
        const filtered = options.filter(t =>
          !t.city || t.city === sel.key || t.id === sel.taskId
        );
        if (!cancelled && filtered.length > 0) {
          setTasks(filtered);
        }
      } catch (error) {
        if (!cancelled && import.meta.env.DEV) {
          console.warn('[CityModal] task fetch failed', error);
        }
      }
    };

    void loadTasks();

    return () => {
      cancelled = true;
    };
  }, [selected, sel.key, sel.taskId]);

  useEffect(() => {
    // When a different city is picked, default its task to its own OSM task.
    if (open) setTaskId(sel.taskId);
  }, [open, selected, sel.taskId]);

  const taskOptions = tasks.some((task) => task.id === sel.taskId)
    ? tasks
    : [{ id: sel.taskId, label: sel.taskId }, ...tasks];

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-foreground/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-10 w-full max-w-[640px] cg-scale-in rounded-2xl border border-foreground/10 bg-card p-6 text-card-foreground cg-shadow-pop sm:p-7">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-foreground/60">
            <span className="relative inline-flex h-2 w-2 rounded-full bg-[hsl(var(--green))] cg-live-pulse" />
            CascadeGuard · Episode Selector
          </div>
          <button onClick={onClose} className="grid h-8 w-8 place-items-center rounded-md hover:bg-foreground/5">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M6 6l12 12M18 6 6 18" />
            </svg>
          </button>
        </div>
        <h2 className="mb-2 text-3xl font-bold tracking-tight">Choose a City</h2>
        <div className="mb-4 rounded-lg border border-foreground/10 bg-foreground/[0.04] p-3 font-mono text-[12px] leading-relaxed text-foreground/70">
          Launch the selected episode against the OpenEnv server at <code>{ENV_WS}</code>. If the backend is unavailable, the UI falls back to the local scripted simulation with the same controls.
        </div>
        <div className="mb-5 grid grid-cols-2 gap-2.5 sm:grid-cols-3">
          {CITIES.map((city) => {
            const active = city.key === selected;
            return (
              <button
                key={city.key}
                onClick={() => onSelect(city.key)}
                className={`rounded-xl border p-3 text-left transition-all ${active ? 'border-[hsl(var(--accent))] bg-[hsl(var(--accent))]/5 ring-2 ring-[hsl(var(--accent))]/15' : 'border-foreground/10 bg-background hover:border-foreground/25'}`}
              >
                <div className="mb-1.5 flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="text-base">{city.flag}</span>
                    <span className="text-sm font-semibold">{city.name}</span>
                  </div>
                  {city.live ? (
                    <span className="rounded bg-[hsl(var(--red))] px-1.5 py-0.5 font-mono text-[9px] font-bold text-white">
                      LIVE
                    </span>
                  ) : (
                    <span className="rounded bg-foreground/10 px-1.5 py-0.5 font-mono text-[9px] font-bold text-foreground/55">
                      SIM
                    </span>
                  )}
                </div>
                <div className="mb-1 font-mono text-[10px] text-foreground/45">{city.taskId}</div>
                <div className="text-[11px] leading-snug text-foreground/65">{city.description}</div>
              </button>
            );
          })}
        </div>

        {/* Task selector */}
        <div className="mb-4">
          <label className="block font-mono text-[10px] uppercase tracking-wider text-foreground/55 mb-1.5">
            Task
          </label>
          <select
            id="city-modal-task-select"
            value={taskId}
            onChange={(e) => setTaskId(e.target.value)}
            className="w-full h-9 px-3 rounded-lg border border-foreground/15 bg-background text-sm font-mono text-foreground focus:outline-none focus:ring-2 focus:ring-[hsl(var(--accent))]/40"
          >
            {taskOptions.map((task) => (
              <option key={task.id} value={task.id}>
                {task.label}{task.id === sel.taskId ? '  (default)' : ''}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={() => onLaunch(selected, taskId)}
          className="h-11 w-full rounded-xl bg-foreground text-sm font-semibold text-background transition-colors hover:bg-[hsl(var(--accent))]"
        >
          Launch Episode — {sel.name} / {taskId}
        </button>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";

import { CITIES, type City, type CityKey } from "../data/cities";
import { ENV_WS } from "../config/env";

interface TaskOption {
  id: string;
  label: string;
  city?: string;
}

interface Props {
  open: boolean;
  selected: CityKey;
  onClose: () => void;
  onSelect: (key: CityKey) => void;
  onLaunch: (key: CityKey, taskId: string) => void;
}

export function CityModal({ open, selected, onClose, onSelect, onLaunch }: Props) {
  const selectedCity: City = CITIES.find((city) => city.key === selected) ?? CITIES[0];
  const [taskId, setTaskId] = useState(selectedCity.taskId);

  useEffect(() => {
    if (open) {
      setTaskId(selectedCity.taskId);
    }
  }, [open, selectedCity.taskId]);

  const tasks: TaskOption[] = CITIES.map((city) => ({
    id: city.taskId,
    label: city.taskId,
    city: city.key,
  })).filter((task) => task.city === selectedCity.key);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-foreground/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-10 w-full max-w-[640px] rounded-2xl border border-foreground/10 bg-card p-6 text-card-foreground shadow-[0_30px_80px_-30px_rgba(0,0,0,0.45)] sm:p-7">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-foreground/60">
            <span className="inline-flex h-2 w-2 rounded-full bg-[hsl(var(--green))] cg-live-pulse" />
            CascadeGuard Episode Selector
          </div>
          <button onClick={onClose} className="grid h-8 w-8 place-items-center rounded-md hover:bg-foreground/5">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M6 6l12 12M18 6 6 18" />
            </svg>
          </button>
        </div>

        <h2 className="mb-2 text-3xl font-bold tracking-tight">Choose a City</h2>
        <div className="mb-4 rounded-lg border border-foreground/10 bg-foreground/[0.04] p-3 font-mono text-[12px] leading-relaxed text-foreground/70">
          Launch the selected episode against the OpenEnv server at <code>{ENV_WS}</code>.
        </div>

        <div className="mb-5 grid grid-cols-2 gap-2.5 sm:grid-cols-3">
          {CITIES.map((city) => {
            const active = city.key === selected;
            return (
              <button
                key={city.key}
                onClick={() => onSelect(city.key)}
                className={`rounded-xl border p-3 text-left transition-all ${
                  active
                    ? "border-[hsl(var(--accent))] bg-[hsl(var(--accent))]/5 ring-2 ring-[hsl(var(--accent))]/15"
                    : "border-foreground/10 bg-background hover:border-foreground/25"
                }`}
              >
                <div className="mb-1.5 flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="text-base">{city.flag}</span>
                    <span className="text-sm font-semibold">{city.name}</span>
                  </div>
                  <span
                    className={`rounded px-1.5 py-0.5 font-mono text-[9px] font-bold ${
                      city.live
                        ? "bg-[hsl(var(--red))] text-white"
                        : "bg-foreground/10 text-foreground/55"
                    }`}
                  >
                    {city.live ? "LIVE" : "SIM"}
                  </span>
                </div>
                <div className="mb-1 font-mono text-[10px] text-foreground/45">{city.taskId}</div>
                <div className="text-[11px] leading-snug text-foreground/65">{city.description}</div>
              </button>
            );
          })}
        </div>

        <div className="mb-4">
          <label className="mb-1.5 block font-mono text-[10px] uppercase tracking-wider text-foreground/55">
            Task
          </label>
          <select
            value={taskId}
            onChange={(event) => setTaskId(event.target.value)}
            className="h-9 w-full rounded-lg border border-foreground/15 bg-background px-3 text-sm font-mono text-foreground focus:outline-none focus:ring-2 focus:ring-[hsl(var(--accent))]/40"
          >
            {tasks.map((task) => (
              <option key={task.id} value={task.id}>
                {task.label}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={() => onLaunch(selected, taskId)}
          className="h-11 w-full rounded-xl bg-foreground text-sm font-semibold text-background transition-colors hover:bg-[hsl(var(--accent))]"
        >
          Launch Episode - {selectedCity.name} / {taskId}
        </button>
      </div>
    </div>
  );
}

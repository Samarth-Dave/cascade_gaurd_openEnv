import { NavLink, Outlet } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { apiClient } from "@/api/client";
import { useCascade } from "@/context/CascadeContext";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/simulate", label: "Simulate" },
  { to: "/analytics", label: "Analytics" },
  { to: "/history", label: "History" },
  { to: "/dataset", label: "Dataset" },
  { to: "/about", label: "About" },
];

export function AppShell() {
  const { sessionId, state, isConnected } = useCascade();
  const { data: health } = useQuery({
    queryKey: ["backend-health"],
    queryFn: apiClient.getHealth,
    refetchInterval: 10_000,
  });
  const modelLoaded = health?.model_loaded ?? false;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="fixed inset-0 -z-10 cg-dotgrid opacity-60" />
      <div className="grid min-h-screen md:grid-cols-[240px_minmax(0,1fr)]">
        <aside className="border-r border-black/10 bg-white/86 p-5 backdrop-blur md:sticky md:top-0 md:h-screen">
          <div className="rounded-[24px] border border-black/10 bg-[linear-gradient(160deg,rgba(20,70,245,0.08),rgba(255,255,255,0.92))] p-5 shadow-[0_26px_60px_-40px_rgba(0,0,0,0.45)]">
            <div className="flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-2xl bg-[hsl(var(--accent))] text-lg font-bold text-white">
                CG
              </div>
              <div>
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-foreground/45">CascadeGuard</div>
                <div className="text-lg font-semibold">Operations Deck</div>
              </div>
            </div>
            <nav className="mt-8 space-y-2">
              {navItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center justify-between rounded-2xl px-4 py-3 text-sm font-medium transition-all",
                      isActive
                        ? "bg-[hsl(var(--accent))] text-white shadow-[0_18px_40px_-28px_rgba(20,70,245,0.9)]"
                        : "bg-transparent text-foreground/70 hover:bg-black/5 hover:text-foreground",
                    )
                  }
                >
                  <span>{item.label}</span>
                  <span className="font-mono text-[10px] uppercase tracking-[0.16em]">
                    {item.label.slice(0, 2)}
                  </span>
                </NavLink>
              ))}
            </nav>
          </div>
        </aside>

        <div className="min-w-0">
          <header className="sticky top-0 z-20 border-b border-black/10 bg-background/88 backdrop-blur">
            <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4 md:px-8">
              <div className="flex items-center gap-3">
                <span className={cn("h-2.5 w-2.5 rounded-full", modelLoaded ? "bg-[hsl(var(--green))] cg-live-pulse" : "bg-[hsl(var(--amber))]")} />
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">
                    {modelLoaded ? "Model Ready" : "Model Loading"}
                  </div>
                  <div className="text-lg font-semibold">Cascade containment dashboard {isConnected ? "• Live" : "• Offline"}</div>
                </div>
              </div>
              <div className="rounded-full border border-black/10 bg-white/80 px-4 py-2 font-mono text-[11px] uppercase tracking-[0.16em] text-foreground/65">
                {sessionId && state ? `${state.task} • ${sessionId.slice(0, 8)}` : "No active episode"}
              </div>
            </div>
          </header>

          <main className="px-5 py-6 md:px-8">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
}

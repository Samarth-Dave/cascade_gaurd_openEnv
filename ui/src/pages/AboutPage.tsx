import { useMemo } from "react";

import { apiClient } from "@/api/client";
import { Card } from "@/components/ui/card";

type AboutLink = {
  label: string;
  href: string;
  display?: string;
};

function buildLinks(): AboutLink[] {
  const origin =
    apiClient.baseUrl ||
    (typeof window !== "undefined" ? window.location.origin : "");
  const docsHref = origin ? `${origin.replace(/\/+$/, "")}/docs` : "/docs";

  return [
    {
      label: "Backend API + OpenAPI Docs",
      href: docsHref,
      display: docsHref,
    },
    {
      label: "Open Simulation",
      href: "/simulate",
      display: "/simulate",
    },
    {
      label: "Training Evidence",
      href: "/analytics",
      display: "/analytics",
    },
    {
      label: "SFT Dataset Browser",
      href: "/dataset",
      display: "/dataset",
    },
  ];
}

export default function AboutPage() {
  const links = useMemo(buildLinks, []);

  return (
    <div className="space-y-6">
      <div>
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">About CascadeGuard</div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">Infrastructure resilience through live orchestration</h1>
      </div>

      <Card className="rounded-[28px] border-black/10 bg-white/88 p-6 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
        <p className="max-w-3xl text-sm leading-7 text-foreground/70">
          CascadeGuard demonstrates how an LLM-guided coordinator can contain cascading failures across power, water, hospital, and telecom systems without training during judge evaluation. The UI drives a session-aware FastAPI backend, the backend orchestrates the OpenEnv environment plus local model reasoning, and the analytics pages surface the training evidence and SFT dataset that produced the policy improvements.
        </p>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {links.map((link) => (
          <Card key={link.label} className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">{link.label}</div>
            <a
              href={link.href}
              className="mt-3 block break-all text-lg font-semibold text-[hsl(var(--accent))] underline-offset-4 hover:underline"
            >
              {link.display ?? link.href}
            </a>
          </Card>
        ))}
      </div>
    </div>
  );
}

import { Card } from "@/components/ui/card";

const links = [
  { label: "Interactive UI", href: "https://<ui-space>.hf.space" },
  { label: "Backend API + Docs", href: "https://<backend-space>.hf.space/docs" },
  { label: "Environment Server", href: "https://<env-space>.hf.space" },
  { label: "GRPO Colab Notebook", href: "training/CascadeGuard_GRPO_Colab.ipynb" },
];

export default function AboutPage() {
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
            <a href={link.href} className="mt-3 block text-lg font-semibold text-[hsl(var(--accent))] underline-offset-4 hover:underline">
              {link.href}
            </a>
          </Card>
        ))}
      </div>
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { apiClient } from "@/api/client";
import { MetricRing } from "@/components/cascade/MetricRing";
import { StateBadge } from "@/components/cascade/StateBadge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useCascade } from "@/context/CascadeContext";

export default function AnalyticsPage() {
  const { state } = useCascade();
  const baselineQuery = useQuery({
    queryKey: ["baseline-comparison"],
    queryFn: apiClient.getBaselineComparison,
  });
  const rewardCurveQuery = useQuery({
    queryKey: ["reward-curve"],
    queryFn: apiClient.getRewardCurve,
  });

  const sectorHealth = state?.sector_health ?? { power: 0.81, water: 0.84, hospital: 0.88, telecom: 0.79 };

  if (baselineQuery.isLoading || rewardCurveQuery.isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          <Skeleton className="h-36 rounded-[28px]" />
          <Skeleton className="h-36 rounded-[28px]" />
          <Skeleton className="h-36 rounded-[28px]" />
        </div>
        <Skeleton className="h-[420px] rounded-[28px]" />
      </div>
    );
  }

  if (baselineQuery.isError || rewardCurveQuery.isError) {
    return (
      <Alert className="border-[hsl(var(--red))]/30 bg-[hsl(var(--red))]/8">
        <AlertTitle>Could not load analytics</AlertTitle>
        <AlertDescription className="flex items-center gap-3">
          <span>Check the backend connection and try again.</span>
          <Button size="sm" variant="outline" onClick={() => {
            void baselineQuery.refetch();
            void rewardCurveQuery.refetch();
          }}>
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  const baseline = baselineQuery.data;
  const rewardCurve = rewardCurveQuery.data;

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Training Evidence</div>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">Policy improvement the judges can see</h1>
        </div>
        <StateBadge label="50 episodes" tone="info" />
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Random Policy</div>
          <div className="mt-4 text-4xl font-semibold">{baseline.random.toFixed(2)}</div>
          <p className="mt-2 text-sm text-foreground/60">No learning, pure chance</p>
        </Card>
        <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Zero-Shot LLM</div>
          <div className="mt-4 text-4xl font-semibold">{baseline.zero_shot.toFixed(2)}</div>
          <p className="mt-2 text-sm text-foreground/60">Instruction-tuned, no env training</p>
        </Card>
        <Card className="rounded-[28px] border-[hsl(var(--accent))]/40 bg-[linear-gradient(180deg,rgba(20,70,245,0.08),rgba(255,255,255,0.92))] p-5 shadow-[0_24px_60px_-36px_rgba(20,70,245,0.45)]">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-[hsl(var(--accent))]">GRPO Trained</div>
          <div className="mt-4 text-4xl font-semibold text-[hsl(var(--accent))]">{baseline.grpo_trained.toFixed(2)}</div>
          <p className="mt-2 text-sm text-foreground/60">50 episodes, Curriculum+CoT</p>
        </Card>
      </div>

      <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">Reward Curve</div>
            <div className="mt-1 text-xl font-semibold">Mean episode score over training</div>
          </div>
          <StateBadge label="curriculum shift @ ep25" tone="warn" />
        </div>

        <div className="h-[360px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rewardCurve} margin={{ top: 20, right: 24, bottom: 24, left: 12 }}>
              <CartesianGrid stroke="rgba(0,0,0,0.08)" strokeDasharray="4 4" />
              <XAxis dataKey="episode" label={{ value: "Training Episode", position: "insideBottom", offset: -10 }} />
              <YAxis domain={[0, 1]} label={{ value: "Mean Episode Score", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              <ReferenceLine
                x={25}
                stroke="rgba(0,0,0,0.3)"
                strokeDasharray="5 5"
                label={{ value: "Curriculum: easy → hard", position: "top", fill: "#555", fontSize: 12 }}
              />
              <Line type="monotone" dataKey="grpo_score" name="GRPO" stroke="#1446f5" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="zero_shot_baseline" name="Zero-shot" stroke="#d97706" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="random_baseline" name="Random" stroke="#9ca3af" strokeWidth={2} strokeDasharray="7 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricRing label="Power" value={sectorHealth.power} sector="power" />
        <MetricRing label="Water" value={sectorHealth.water} sector="water" />
        <MetricRing label="Hospital" value={sectorHealth.hospital} sector="hospital" />
        <MetricRing label="Telecom" value={sectorHealth.telecom} sector="telecom" />
      </div>
    </div>
  );
}

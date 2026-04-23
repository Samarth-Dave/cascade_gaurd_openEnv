// Static GRPO training curve points ep1..ep50 — climbs from ~0.18 to ~0.81
export const TRAINING_CURVE: number[] = (() => {
  const pts: number[] = [];
  for (let i=0;i<50;i++) {
    const t = i/49;
    const base = 0.18 + (0.81-0.18) * (1 - Math.exp(-3.2*t));
    const noise = (Math.sin(i*1.7)+Math.cos(i*0.9))*0.018;
    pts.push(Math.max(0.05, Math.min(0.93, base + noise)));
  }
  return pts;
})();

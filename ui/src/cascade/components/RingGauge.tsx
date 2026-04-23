interface Props {
  value: number; // 0..1
  size?: number;
  label?: string;
  showPct?: boolean;
}
export function RingGauge({ value, size = 28, label, showPct = true }: Props) {
  const stroke = Math.max(3, size/8);
  const r = (size - stroke) / 2;
  const C = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value));
  const off = C * (1 - v);
  const color = v >= 0.7 ? 'hsl(var(--green))' : v >= 0.4 ? 'hsl(var(--amber))' : 'hsl(var(--red))';
  return (
    <div className="flex items-center gap-2">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
        <circle cx={size/2} cy={size/2} r={r} stroke="hsl(var(--border) / 0.18)" strokeWidth={stroke} fill="none" />
        <circle cx={size/2} cy={size/2} r={r} stroke={color} strokeWidth={stroke} strokeLinecap="round"
                fill="none" strokeDasharray={C} strokeDashoffset={off} style={{ transition: 'stroke-dashoffset .6s ease' }} />
      </svg>
      {showPct && <span className="font-mono text-[11px] text-foreground/80">{Math.round(v*100)}%</span>}
      {label && <span className="text-xs text-foreground/60">{label}</span>}
    </div>
  );
}

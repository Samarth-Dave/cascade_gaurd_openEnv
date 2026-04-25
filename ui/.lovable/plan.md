
# CascadeGuard — Infrastructure Resilience Intelligence

A pixel-faithful single-page dashboard built on **Vite + React 18 + TypeScript + Tailwind** (Next.js isn't supported here, but every feature in your spec maps cleanly to this stack). Tries to connect to `ws://localhost:8000/ws` on load and automatically falls back to a scripted 11-step London episode if the env isn't reachable.

## Design system
- DM Sans + DM Mono via Google Fonts
- HSL CSS variables matching your spec: `--bg #fafafa`, `--ink #0d0d0d`, `--accent #1446f5`, `--accent2 #00a878`, `--accent3 #e8192c`, plus amber/purple/green/red semantics, borders, shadow scale
- Fixed dot-grid background (28px radial, z-0, pointer-events none)
- Custom keyframes: `fadeUp`, `livePulse`, `ringOut`, `slideIn`, `cotIn`, `dashFlow`, `lineFlow`, `blink`, `scaleIn`
- IntersectionObserver scroll-reveal for sections, stat counters, and progress bars

## Sections (single scrolling page)

1. **Navbar** — fixed 60px, blur backdrop, hex logo + "CascadeGuard", center nav (Home, Live Map, AI Duel, Neural Graph, Threat Intel, Analytics, Logs), city badge button, black "⬇ Report" CTA
2. **City modal** — fullscreen blur overlay, opens on load and via city badge. 6 city cards (London = LIVE, others = SIMULATION), London preselected, "Launch Episode — [City]" button
3. **Hero** — viewport-height, live-episode pill, giant "CASCADE / GUARD" (GUARD in accent blue, clamp typography, -6px tracking), subtitle, three CTAs (Live Map / AI Duel / Neural Graph)
4. **Live Map** — Leaflet (dark CartoDB tiles, dynamic client-side mount), 540px height, custom div markers per node with pulsing rings on critical, color-coded polylines (cascade/reroute/isolate). Overlays: Dispatch Action panel (target dropdown + 5 action buttons with cost labels), Sector Health rings (Power/Water/Hospital/Telecom), terminal-style Live Event Feed (cap 20). Top control bar (Reset/Play/Step/Speed), step progress bar, connection banner, legend, "What's Happening" strip
5. **AI Duel** — split black/blue card, VS badge, Attacker (red ⟨think⟩/⟨action⟩ blocks + attack-pressure bar) vs Defender (blue/green blocks + defense-score bar)
6. **Neural Graph** — 960×500 SVG with 6 sector zones, 13 fixed-position nodes, 11 edges. requestAnimationFrame loop animates cascade (red, glow, arrowhead) and reroute (teal) edges. Stats strip below
7. **Threat Intel** — Attacker Terminal (macOS dots, blinking cursor, scrollable colored lines) + Reward Canvas (HTML5 Canvas reward curve, dashed 0.23 baseline, gradient fill, live updates)
8. **Analytics** — 4 stat cards with count-up (Episode Score / Budget / Nodes at Risk / Steps), static GRPO training curve SVG (ep1→ep50), 4 sector ring gauges
9. **Step-by-Step Trace** — vertical timeline with gradient left border, dot + step label + title + description + tags, scroll-reveal
10. **How the Agent Improved** — 2×2 grid (Reasoning 88%, Reward 94%, Curriculum 72%, Efficiency 81%) with progress bars that fill on scroll, plus Before/After 3-col (Random 0.23 / Zero-Shot 0.49 / GRPO 0.81 highlighted)
11. **Logs** — dark terminal card, timestamped colored level badges, scrollable max-380px, downloads: full report (.txt), raw episode (.json), reward curve (.csv)
12. **Footer** — "CascadeGuard" + "Meta × HuggingFace Hackathon · OpenEnv · 2025"

## State & data flow
- `useEnvSocket` hook: connects to `ws://localhost:8000/ws`, sends `{type:'reset', data:{task_id:'task_hard'}}`, parses `observation` messages, exposes connection state (connecting/live/sim/error), cleans up on unmount
- `applyState()` — single source of truth: every observation (live or sim) is normalized into one shape that updates map markers, edges, sector rings, header stats, agent CoT, neural graph, terminal, reward canvas, event feed, and timeline simultaneously
- Node status derivation rules exactly per spec (in_recovery → repair, !operational → critical, health<0.35 → degraded, hardened → operational, else healthy)
- Coordinate fallback: if server lat/lon are zero, distribute nodes around city center on a circle (`angle = i/n*2π`, `radius = 0.015 + (i%3)*0.008`)
- Play speeds: 0.5×/1×/2×/4× → 2400/1400/700/300ms tick
- Action dispatch: target node + action type → step message → buttons disabled until next observation
- City switch: re-centers map, updates labels, reconnects WS with new task_id, resets sim
- **Scripted 11-step London episode** (used on WS failure): all healthy → PWR_1 critical w/ cascade → 3 recovery steps → fully restored → 5 monitoring steps. Includes per-step node states, cascade edges, sector health, attacker/defender CoT, events, timeline entries — content matched to your screenshots ("PWR_1 critical failure", "Recover PWR_1 initiated", "HOSP_1 recovery started", "PWR_1 fully repaired", "All sectors restored", etc.)

## Responsive (<960px)
Stats 2-col, sectors 2-col, hide center nav links, hide map sidebar + event feed, map height 380px, city grid 2-col, agent panels stack, action panel hidden

## Out of scope for this build
- Real OpenEnv backend (we connect-and-fallback; you run the WS server locally to see live mode)
- Actual GRPO model inference (CoT content is scripted to match the demo episode)

import { HexLogo } from './icons';
import type { City } from '../data/cities';

interface Props {
  city: City;
  onOpenCity: () => void;
}
export function Navbar({ city, onOpenCity }: Props) {
  const links = ['Home','Live Map','AI Duel','Neural Graph','Threat Intel','Analytics','Logs'];
  const ids   = ['#hero','#map','#duel','#graph','#intel','#analytics','#logs'];
  return (
    <header className="fixed inset-x-0 top-0 z-[120] h-[60px] cg-glass border-b border-foreground/10">
      <div className="h-full max-w-[1400px] mx-auto px-5 flex items-center justify-between gap-4">
        <a href="#hero" className="flex items-center gap-2.5">
          <HexLogo />
          <span className="font-bold tracking-tight text-[17px]">CascadeGuard</span>
        </a>
        <nav className="hidden lg:flex items-center gap-1">
          {links.map((l, i) => (
            <a key={l} href={ids[i]} className="px-3 py-2 text-[13px] text-foreground/70 hover:text-foreground rounded-md transition-colors">
              {l}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <button onClick={onOpenCity}
            className="hidden sm:flex items-center gap-2 px-3 h-9 rounded-full border border-foreground/10 bg-background hover:bg-foreground/5 text-[13px] font-medium transition-colors">
            <span>{city.flag}</span><span>{city.name}</span>
          </button>
          <button className="flex items-center gap-1.5 px-3.5 h-9 rounded-full bg-foreground text-background hover:bg-[hsl(var(--accent))] text-[13px] font-semibold transition-colors">
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v12m0 0 4-4m-4 4-4-4M5 21h14"/></svg>
            Report
          </button>
        </div>
      </div>
    </header>
  );
}

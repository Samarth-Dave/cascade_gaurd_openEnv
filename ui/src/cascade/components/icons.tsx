import type { Sector } from '../data/types';

export const SectorIcon = ({ sector, className = 'h-4 w-4' }: { sector: Sector; className?: string }) => {
  const common = { className, fill:'none', stroke:'currentColor', strokeWidth:1.8, strokeLinecap:'round' as const, strokeLinejoin:'round' as const, viewBox:'0 0 24 24' };
  switch (sector) {
    case 'power':
      return <svg {...common}><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>;
    case 'water':
      return <svg {...common}><path d="M12 3s6 7 6 12a6 6 0 1 1-12 0c0-5 6-12 6-12z"/></svg>;
    case 'hospital':
      return <svg {...common}><rect x="4" y="5" width="16" height="15" rx="2"/><path d="M12 9v7M9 12h6"/></svg>;
    case 'telecom':
      return <svg {...common}><path d="M5 12a7 7 0 0 1 14 0"/><path d="M8 12a4 4 0 0 1 8 0"/><circle cx="12" cy="12" r="1.5"/><path d="M12 14v6"/></svg>;
    case 'storage':
      return <svg {...common}><rect x="3" y="8" width="16" height="10" rx="2"/><path d="M19 11h2v4h-2"/><path d="M7 13h6"/></svg>;
    case 'grid':
    default:
      return <svg {...common}><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z"/></svg>;
  }
};

// Raw SVG path data (for embedding inside string-templated HTML, e.g. Leaflet divIcons)
export const SECTOR_PATHS: Record<Sector, string> = {
  power:    '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
  water:    '<path d="M12 3s6 7 6 12a6 6 0 1 1-12 0c0-5 6-12 6-12z"/>',
  hospital: '<rect x="4" y="5" width="16" height="15" rx="2"/><path d="M12 9v7M9 12h6"/>',
  telecom:  '<path d="M5 12a7 7 0 0 1 14 0"/><path d="M8 12a4 4 0 0 1 8 0"/><circle cx="12" cy="12" r="1.5"/><path d="M12 14v6"/>',
  storage:  '<rect x="3" y="8" width="16" height="10" rx="2"/><path d="M19 11h2v4h-2"/><path d="M7 13h6"/>',
  grid:     '<path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z"/>',
};

export const sectorSvg = (sector: Sector, size = 18) =>
  `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${SECTOR_PATHS[sector] ?? SECTOR_PATHS.grid}</svg>`;

export const HexLogo = ({ className = 'h-7 w-7' }: { className?: string }) => (
  <div className={`${className} rounded-lg bg-foreground flex items-center justify-center`}>
    <svg viewBox="0 0 24 24" className="h-4 w-4 text-background" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round">
      <path d="M12 2 21 7v10l-9 5-9-5V7z"/>
      <path d="M12 8v8M8 12h8" strokeLinecap="round" />
    </svg>
  </div>
);

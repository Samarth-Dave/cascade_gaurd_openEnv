export type CityKey = 'london' | 'mumbai' | 'bangalore' | 'delhi' | 'nyc' | 'tokyo';

export interface City {
  key: CityKey;
  name: string;
  flag: string;
  taskId: string;
  description: string;
  live: boolean;
  center: [number, number];
  zoom: number;
}

export const CITIES: City[] = [
  { key:'london',   name:'London',    flag:'🇬🇧', taskId:'task_osm_london',    description:'Dense critical-infrastructure mesh. NHS hospitals, Thames Water, National Grid.', live:true,  center:[51.506, -0.109], zoom:13 },
  { key:'mumbai',   name:'Mumbai',    flag:'🇮🇳', taskId:'task_osm_mumbai',    description:'Coastal megacity, monsoon-stressed grid, BEST power network.', live:false, center:[19.076, 72.877], zoom:12 },
  { key:'bangalore',name:'Bangalore', flag:'🇮🇳', taskId:'task_osm_bangalore', description:'Tech-corridor load spikes, BESCOM substations, BWSSB pumping.', live:false, center:[12.972, 77.594], zoom:12 },
  { key:'delhi',    name:'Delhi',     flag:'🇮🇳', taskId:'task_osm_delhi',     description:'Capital region, AIIMS hospitals, DTL transmission backbone.', live:false, center:[28.644, 77.216], zoom:12 },
  { key:'nyc',      name:'New York',  flag:'🇺🇸', taskId:'task_osm_nyc',       description:'Con Edison grid, Bellevue, MTA telecom interdependencies.', live:false, center:[40.712, -74.006], zoom:12 },
  { key:'tokyo',    name:'Tokyo',     flag:'🇯🇵', taskId:'task_osm_tokyo',     description:'TEPCO substations, seismic resilience, Shinjuku core demand.', live:false, center:[35.682, 139.769], zoom:12 },
];

export const cityByKey = (k: CityKey) => CITIES.find(c => c.key === k)!;

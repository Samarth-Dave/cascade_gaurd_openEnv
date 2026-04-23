"""
fetch_real_nodes.py
Queries OpenStreetMap Overpass API for real infrastructure nodes
for each CascadeGuard city and outputs data/real_nodes.json.
Run once before starting the server:  python scripts/fetch_real_nodes.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    print("WARNING: 'requests' not installed. Using fallback coords only.", file=sys.stderr)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

CITY_BBOXES: Dict[str, str] = {
    "london":    "51.48,-0.15,51.54,-0.08",
    "mumbai":    "18.90,72.80,19.15,72.98",
    "bangalore": "12.90,77.52,13.05,77.68",
    "delhi":     "28.58,77.17,28.72,77.28",
    "nyc":       "40.68,-74.05,40.78,-73.93",
    "tokyo":     "35.65,139.72,35.72,139.82",
}

CITY_CENTERS: Dict[str, Tuple[float, float]] = {
    "london":    (51.506, -0.109),
    "mumbai":    (19.076, 72.877),
    "bangalore": (12.972, 77.594),
    "delhi":     (28.644, 77.216),
    "nyc":       (40.712, -74.006),
    "tokyo":     (35.682, 139.769),
}

# Queries: (osm_tag_kv, limit, sector, node_id_prefix)
QUERY_SPECS = [
    ("power=substation",        4, "power",    "SUB"),
    ("power=generator",         2, "power",    "GEN"),
    ("man_made=pumping_station", 3, "water",   "PUMP"),
    ("man_made=water_tower",    2, "water",    "WTWR"),
    ("amenity=hospital",        4, "hospital", "HOSP"),
    ("man_made=mast",           2, "telecom",  "MAST"),
    ("man_made=tower",          1, "telecom",  "TWR"),
]

# ── Fallback data if Overpass is unreachable ──────────────────────────────────
FALLBACK_NODES: Dict[str, List[Dict]] = {
    "london": [
        {"lat": 51.5074, "lng": -0.1278, "sector": "power",    "id": "SUB_CITY",     "name": "City of London Substation"},
        {"lat": 51.5200, "lng": -0.0900, "sector": "power",    "id": "SUB_STEPNEY",  "name": "Stepney Substation"},
        {"lat": 51.4900, "lng": -0.1400, "sector": "power",    "id": "GEN_BATTERSEA","name": "Battersea Power Station"},
        {"lat": 51.5010, "lng": -0.1195, "sector": "water",    "id": "PUMP_THAMES",  "name": "Thames Water Pump"},
        {"lat": 51.5150, "lng": -0.0800, "sector": "water",    "id": "PUMP_LEA",     "name": "Lea Valley Pump"},
        {"lat": 51.4836, "lng": -0.1276, "sector": "water",    "id": "WTWR_STREATHAM","name": "Streatham Water Tower"},
        {"lat": 51.4877, "lng": -0.1174, "sector": "hospital", "id": "HOSP_KINGS",   "name": "King's College Hospital"},
        {"lat": 51.4989, "lng": -0.1187, "sector": "hospital", "id": "HOSP_ST_THOMAS","name": "St Thomas' Hospital"},
        {"lat": 51.5229, "lng": -0.0754, "sector": "hospital", "id": "HOSP_ROYAL_LONDON","name": "Royal London Hospital"},
        {"lat": 51.5246, "lng": -0.0834, "sector": "hospital", "id": "HOSP_WHITECHAPEL","name": "Whitechapel Medical"},
        {"lat": 51.5130, "lng": -0.1100, "sector": "telecom",  "id": "MAST_BT_TOWER","name": "BT Tower"},
        {"lat": 51.5000, "lng": -0.0850, "sector": "telecom",  "id": "TWR_CANARY",   "name": "Canary Wharf Tower"},
    ],
    "mumbai": [
        {"lat": 19.0760, "lng": 72.8777, "sector": "power",    "id": "SUB_DHARAVI",  "name": "Dharavi Substation"},
        {"lat": 19.1200, "lng": 72.8400, "sector": "power",    "id": "SUB_BANDRA",   "name": "Bandra Substation"},
        {"lat": 19.0550, "lng": 72.8350, "sector": "power",    "id": "GEN_TROMBAY",  "name": "Trombay Power Plant"},
        {"lat": 19.0650, "lng": 72.8350, "sector": "water",    "id": "PUMP_WORLI",   "name": "Worli Pumping Station"},
        {"lat": 19.1050, "lng": 72.8700, "sector": "water",    "id": "PUMP_MATUNGA", "name": "Matunga Pump"},
        {"lat": 19.0800, "lng": 72.8550, "sector": "water",    "id": "WTWR_SION",    "name": "Sion Water Tower"},
        {"lat": 19.0550, "lng": 72.8350, "sector": "hospital", "id": "HOSP_KEM",     "name": "KEM Hospital"},
        {"lat": 19.1150, "lng": 72.8750, "sector": "hospital", "id": "HOSP_SION",    "name": "Sion Hospital"},
        {"lat": 19.0760, "lng": 72.8600, "sector": "hospital", "id": "HOSP_NAIR",    "name": "Nair Hospital"},
        {"lat": 19.1800, "lng": 72.9600, "sector": "hospital", "id": "HOSP_THANE",   "name": "Thane Civil Hospital"},
        {"lat": 19.1000, "lng": 72.8200, "sector": "telecom",  "id": "MAST_BANDRA",  "name": "Bandra Telecom Mast"},
        {"lat": 19.0500, "lng": 72.9000, "sector": "telecom",  "id": "TWR_SOUTH",    "name": "South Mumbai Tower"},
    ],
    "bangalore": [
        {"lat": 12.9716, "lng": 77.5946, "sector": "power",    "id": "SUB_CENTRAL",  "name": "Central Substation"},
        {"lat": 13.0200, "lng": 77.5900, "sector": "power",    "id": "SUB_NORTH",    "name": "North Bangalore Sub"},
        {"lat": 12.9500, "lng": 77.5800, "sector": "power",    "id": "GEN_SOUTH",    "name": "South Generator"},
        {"lat": 12.9600, "lng": 77.5700, "sector": "water",    "id": "PUMP_CAUVERY", "name": "Cauvery Pump Station"},
        {"lat": 12.9900, "lng": 77.6100, "sector": "water",    "id": "PUMP_EAST",    "name": "East Pump Station"},
        {"lat": 12.9800, "lng": 77.5800, "sector": "water",    "id": "WTWR_MG",      "name": "MG Road Water Tower"},
        {"lat": 12.9698, "lng": 77.6064, "sector": "hospital", "id": "HOSP_VICTORIA","name": "Victoria Hospital"},
        {"lat": 12.9950, "lng": 77.5710, "sector": "hospital", "id": "HOSP_NIMHANS", "name": "NIMHANS Hospital"},
        {"lat": 13.0100, "lng": 77.5500, "sector": "hospital", "id": "HOSP_NORTH",   "name": "North Bangalore Hospital"},
        {"lat": 12.9300, "lng": 77.6200, "sector": "hospital", "id": "HOSP_SOUTH",   "name": "South Bangalore Hospital"},
        {"lat": 12.9800, "lng": 77.5500, "sector": "telecom",  "id": "MAST_INDIRANAGAR","name": "Indiranagar Mast"},
        {"lat": 12.9600, "lng": 77.6400, "sector": "telecom",  "id": "TWR_KORAMANGALA","name": "Koramangala Tower"},
    ],
    "delhi": [
        {"lat": 28.6139, "lng": 77.2090, "sector": "power",    "id": "SUB_CONNAUGHT","name": "Connaught Place Sub"},
        {"lat": 28.6800, "lng": 77.2200, "sector": "power",    "id": "SUB_NORTH",    "name": "North Delhi Substation"},
        {"lat": 28.5500, "lng": 77.2000, "sector": "power",    "id": "GEN_INDRAPRASTHA","name": "Indraprastha Plant"},
        {"lat": 28.6000, "lng": 77.1800, "sector": "water",    "id": "PUMP_YAMUNA",  "name": "Yamuna Pump Station"},
        {"lat": 28.6500, "lng": 77.2500, "sector": "water",    "id": "PUMP_EAST",    "name": "East Delhi Pump"},
        {"lat": 28.6200, "lng": 77.2000, "sector": "water",    "id": "WTWR_CENTRAL", "name": "Central Water Tower"},
        {"lat": 28.6400, "lng": 77.2000, "sector": "hospital", "id": "HOSP_AIIMS",   "name": "AIIMS Hospital"},
        {"lat": 28.6517, "lng": 77.2219, "sector": "hospital", "id": "HOSP_SAFDAR",  "name": "Safdarjung Hospital"},
        {"lat": 28.6750, "lng": 77.2200, "sector": "hospital", "id": "HOSP_GTB",     "name": "GTB Hospital"},
        {"lat": 28.5900, "lng": 77.2100, "sector": "hospital", "id": "HOSP_SOUTH",   "name": "South Delhi Hospital"},
        {"lat": 28.6300, "lng": 77.2100, "sector": "telecom",  "id": "MAST_CP",      "name": "Connaught Place Mast"},
        {"lat": 28.6600, "lng": 77.2300, "sector": "telecom",  "id": "TWR_NORTH",    "name": "North Delhi Tower"},
    ],
    "nyc": [
        {"lat": 40.7128, "lng": -74.0060, "sector": "power",    "id": "SUB_MANHATTAN","name": "Manhattan Substation"},
        {"lat": 40.7500, "lng": -73.9500, "sector": "power",    "id": "SUB_QUEENS",   "name": "Queens Substation"},
        {"lat": 40.6800, "lng": -73.9500, "sector": "power",    "id": "GEN_BROOKLYN", "name": "Brooklyn Generator"},
        {"lat": 40.7200, "lng": -73.9900, "sector": "water",    "id": "PUMP_HUDSON",  "name": "Hudson River Pump"},
        {"lat": 40.7400, "lng": -73.9300, "sector": "water",    "id": "PUMP_BRONX",   "name": "Bronx Pump Station"},
        {"lat": 40.7050, "lng": -74.0150, "sector": "water",    "id": "WTWR_LOWER",   "name": "Lower Manhattan Tower"},
        {"lat": 40.7635, "lng": -73.9540, "sector": "hospital", "id": "HOSP_BELLEVUE","name": "Bellevue Hospital"},
        {"lat": 40.7900, "lng": -73.9500, "sector": "hospital", "id": "HOSP_COLUMBIA","name": "Columbia Presbyterian"},
        {"lat": 40.7000, "lng": -73.9800, "sector": "hospital", "id": "HOSP_BROOKLYN","name": "Brooklyn Medical"},
        {"lat": 40.7300, "lng": -73.9400, "sector": "hospital", "id": "HOSP_QUEENS",  "name": "Queens Hospital"},
        {"lat": 40.7484, "lng": -73.9967, "sector": "telecom",  "id": "MAST_EMPIRE",  "name": "Empire State Antenna"},
        {"lat": 40.6892, "lng": -74.0445, "sector": "telecom",  "id": "TWR_LIBERTY",  "name": "Liberty Tower"},
    ],
    "tokyo": [
        {"lat": 35.6762, "lng": 139.6503, "sector": "power",    "id": "SUB_SHIBUYA",  "name": "Shibuya Substation"},
        {"lat": 35.7090, "lng": 139.7320, "sector": "power",    "id": "SUB_UENO",     "name": "Ueno Substation"},
        {"lat": 35.6500, "lng": 139.7500, "sector": "power",    "id": "GEN_SHINAGAWA","name": "Shinagawa Generator"},
        {"lat": 35.6700, "lng": 139.7100, "sector": "water",    "id": "PUMP_SUMIDA",  "name": "Sumida River Pump"},
        {"lat": 35.7000, "lng": 139.7700, "sector": "water",    "id": "PUMP_ARAKAWA", "name": "Arakawa Pump Station"},
        {"lat": 35.6800, "lng": 139.7400, "sector": "water",    "id": "WTWR_SHINJUKU","name": "Shinjuku Water Tower"},
        {"lat": 35.7023, "lng": 139.7731, "sector": "hospital", "id": "HOSP_TOKYO_U", "name": "Tokyo University Hospital"},
        {"lat": 35.6894, "lng": 139.6917, "sector": "hospital", "id": "HOSP_KEIO",    "name": "Keio University Hospital"},
        {"lat": 35.7200, "lng": 139.7500, "sector": "hospital", "id": "HOSP_JIKEI",   "name": "Jikei University Hospital"},
        {"lat": 35.6600, "lng": 139.7300, "sector": "hospital", "id": "HOSP_SHOWA",   "name": "Showa University Hospital"},
        {"lat": 35.6586, "lng": 139.7454, "sector": "telecom",  "id": "MAST_TOKYO",   "name": "Tokyo Tower"},
        {"lat": 35.7101, "lng": 139.8107, "sector": "telecom",  "id": "TWR_SKYTREE",  "name": "Tokyo Skytree"},
    ],
}


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def overpass_fetch(tag_kv: str, bbox: str, limit: int) -> List[Dict]:
    """Fetch nodes from Overpass API. Returns list of {lat, lng, osm_id, name}."""
    if not REQUESTS_OK:
        return []
    key, val = tag_kv.split("=", 1)
    query = (
        f'[out:json][timeout:30];'
        f'(node["{key}"="{val}"]({bbox});'
        f'way["{key}"="{val}"]({bbox}););'
        f'out center {limit};'
    )
    headers = {
        "User-Agent": "CascadeGuard/4.0 (hackathon; contact@cascadeguard.ai)",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    try:
        # Try GET first (most Overpass mirrors prefer it)
        resp = requests.get(
            OVERPASS_URL,
            params={"data": query},
            headers=headers,
            timeout=35,
        )
        if resp.status_code == 406:
            # Fall back to POST with form-encoded body
            resp = requests.post(
                OVERPASS_URL,
                data=f"data={requests.utils.quote(query)}",
                headers=headers,
                timeout=35,
            )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        nodes = []
        for el in elements:
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lng = el.get("lon") or el.get("center", {}).get("lon")
            if lat and lng:
                name = el.get("tags", {}).get("name", "")
                nodes.append({"lat": float(lat), "lng": float(lng),
                               "osm_id": el["id"], "name": name})
        return nodes[:limit]
    except Exception as exc:
        print(f"    Overpass error ({tag_kv}): {exc}", file=sys.stderr)
        return []


def _make_nid(prefix: str, idx: int, name: str) -> str:
    if name:
        slug = name.upper().replace(" ", "_").replace("'", "").replace("-", "_")
        # Keep only alphanumeric + underscore, max 14 chars
        slug = "".join(c for c in slug if c.isalnum() or c == "_")[:14]
        return f"{prefix}_{slug}"
    return f"{prefix}_{idx + 1}"


def _service_radius(sector: str) -> float:
    return {"power": 35.0, "water": 20.0, "hospital": 8.0, "telecom": 100.0, "ai": 5.0}.get(sector, 15.0)


def fetch_city(city: str, bbox: str) -> Optional[Dict]:
    """Fetch real nodes for one city from Overpass. Returns structured dict or None on total failure."""
    nodes_out: List[Dict] = []
    sector_counts: Dict[str, int] = {}

    for tag_kv, limit, sector, prefix in QUERY_SPECS:
        raw = overpass_fetch(tag_kv, bbox, limit)
        time.sleep(0.8)  # be polite to Overpass
        for item in raw:
            idx = sector_counts.get(sector, 0)
            nid = _make_nid(prefix, idx, item["name"])
            # Ensure unique
            existing_ids = {n["id"] for n in nodes_out}
            while nid in existing_ids:
                nid = nid + "_X"
            nodes_out.append({
                "id":       nid,
                "lat":      round(item["lat"], 6),
                "lng":      round(item["lng"], 6),
                "sector":   sector,
                "name":     item["name"] or nid,
                "osm_id":   item.get("osm_id"),
                "service_radius_km": _service_radius(sector),
            })
            sector_counts[sector] = idx + 1

    return nodes_out if nodes_out else None


def build_edges(nodes: List[Dict]) -> List[List[str]]:
    """Build sector-topology edges matching CascadeGuard cascade logic."""
    edges: set = set()
    by_sector: Dict[str, List[str]] = {}
    for n in nodes:
        by_sector.setdefault(n["sector"], []).append(n["id"])

    power    = by_sector.get("power", [])
    water    = by_sector.get("water", [])
    hospital = by_sector.get("hospital", [])
    telecom  = by_sector.get("telecom", [])

    # Power backbone chain
    for i in range(len(power) - 1):
        edges.add((power[i], power[i + 1]))
    if len(power) >= 3:
        edges.add((power[0], power[2]))

    primary = power[0] if power else None

    # Power → other sectors
    if primary:
        for nid in water[:1]:   edges.add((primary, nid))
        for nid in hospital[:1]: edges.add((primary, nid))
        for nid in telecom[:1]:  edges.add((primary, nid))
        edges.add((primary, "AI_CORE"))

    # Water chain
    for i in range(len(water) - 1):
        edges.add((water[i], water[i + 1]))

    # Hospital chain
    for i in range(len(hospital) - 1):
        edges.add((hospital[i], hospital[i + 1]))

    # Telecom chain
    for i in range(len(telecom) - 1):
        edges.add((telecom[i], telecom[i + 1]))

    # AI_CORE hub
    for lst in [water[:1], hospital[:1], telecom[:1]]:
        for nid in lst:
            edges.add(("AI_CORE", nid))

    return [list(e) for e in edges]


def ensure_connected(nodes: List[Dict], edges: List[List[str]]) -> List[List[str]]:
    """If graph is disconnected, bridge closest pair of nodes across components."""
    node_ids = {n["id"] for n in nodes} | {"AI_CORE"}
    coord = {n["id"]: (n["lat"], n["lng"]) for n in nodes}

    # BFS
    adj: Dict[str, set] = {nid: set() for nid in node_ids}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    visited: set = set()
    components: List[set] = []
    for start in node_ids:
        if start not in visited:
            comp: set = set()
            stack = [start]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                comp.add(v)
                stack.extend(adj[v] - visited)
            components.append(comp)

    extra: List[List[str]] = []
    while len(components) > 1:
        # Find closest pair across first two components
        best_dist = float("inf")
        best_pair = ("", "")
        for a in components[0]:
            if a not in coord:
                continue
            for b in components[1]:
                if b not in coord:
                    continue
                d = haversine_km(coord[a][0], coord[a][1], coord[b][0], coord[b][1])
                if d < best_dist:
                    best_dist = d
                    best_pair = (a, b)
        if best_pair[0]:
            extra.append(list(best_pair))
            components[0] |= components[1]
        components.pop(1)

    return edges + extra


def process_city(city: str, bbox: str) -> Dict:
    center = CITY_CENTERS[city]
    print(f"  Querying Overpass for {city}...", file=sys.stderr)
    nodes = fetch_city(city, bbox)

    if not nodes:
        print(f"  Overpass returned nothing for {city}. Using fallback.", file=sys.stderr)
        raw_fallback = FALLBACK_NODES.get(city, [])
        nodes = [{
            "id":   n["id"],
            "lat":  n["lat"],
            "lng":  n["lng"],
            "sector": n["sector"],
            "name": n["name"],
            "osm_id": None,
            "service_radius_km": _service_radius(n["sector"]),
        } for n in raw_fallback]

    # AI_CORE at centroid
    lats = [n["lat"] for n in nodes]
    lngs = [n["lng"] for n in nodes]
    ai_lat = sum(lats) / len(lats) if lats else center[0]
    ai_lng = sum(lngs) / len(lngs) if lngs else center[1]
    nodes.append({
        "id": "AI_CORE", "lat": round(ai_lat, 6), "lng": round(ai_lng, 6),
        "sector": "ai", "name": "CascadeGuard AI Core",
        "osm_id": None, "service_radius_km": 5.0,
    })

    edges = build_edges(nodes)
    edges = ensure_connected(nodes, edges)

    return {
        "nodes": nodes,
        "edges": edges,
        "center": list(center),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def main():
    out_path = Path(__file__).parent.parent / "data" / "real_nodes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output: Dict[str, Dict] = {}
    for city, bbox in CITY_BBOXES.items():
        try:
            result = process_city(city, bbox)
            output[city] = result
            print(
                f"  {city}: {result['node_count']} nodes, {result['edge_count']} edges",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"  ERROR processing {city}: {exc}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

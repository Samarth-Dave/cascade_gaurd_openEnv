"""generate_fallback_nodes.py — Write complete real_nodes.json from curated fallback data."""
import json
from pathlib import Path

RADIUS = {"power": 35.0, "water": 20.0, "hospital": 8.0, "telecom": 100.0, "ai": 5.0}

CENTERS = {
    "london":    [51.506, -0.109],
    "mumbai":    [19.076, 72.877],
    "bangalore": [12.972, 77.594],
    "delhi":     [28.644, 77.216],
    "nyc":       [40.712, -74.006],
    "tokyo":     [35.682, 139.769],
}

FALLBACK = {
    "london": [
        {"lat": 51.5074, "lng": -0.1278, "sector": "power",    "id": "SUB_CITY",       "name": "City of London Substation"},
        {"lat": 51.5200, "lng": -0.0900, "sector": "power",    "id": "SUB_STEPNEY",    "name": "Stepney Substation"},
        {"lat": 51.4900, "lng": -0.1400, "sector": "power",    "id": "GEN_BATTERSEA",  "name": "Battersea Power Station"},
        {"lat": 51.5010, "lng": -0.1195, "sector": "water",    "id": "PUMP_THAMES",    "name": "Thames Water Pump Station"},
        {"lat": 51.5150, "lng": -0.0800, "sector": "water",    "id": "PUMP_LEA",       "name": "Lea Valley Pump Station"},
        {"lat": 51.4836, "lng": -0.1276, "sector": "water",    "id": "WTWR_STREATHAM", "name": "Streatham Water Tower"},
        {"lat": 51.4877, "lng": -0.1174, "sector": "hospital", "id": "HOSP_KINGS",     "name": "Kings College Hospital"},
        {"lat": 51.4989, "lng": -0.1187, "sector": "hospital", "id": "HOSP_ST_THOMAS", "name": "St Thomas Hospital"},
        {"lat": 51.5229, "lng": -0.0754, "sector": "hospital", "id": "HOSP_ROYAL_LONDON", "name": "Royal London Hospital"},
        {"lat": 51.5246, "lng": -0.0834, "sector": "hospital", "id": "HOSP_WHITECHAPEL",  "name": "Whitechapel Medical Centre"},
        {"lat": 51.5130, "lng": -0.1100, "sector": "telecom",  "id": "MAST_BT_TOWER",  "name": "BT Tower"},
        {"lat": 51.5000, "lng": -0.0850, "sector": "telecom",  "id": "TWR_CANARY",     "name": "Canary Wharf Tower"},
    ],
    "mumbai": [
        {"lat": 19.0760, "lng": 72.8777, "sector": "power",    "id": "SUB_DHARAVI",  "name": "Dharavi Substation"},
        {"lat": 19.1200, "lng": 72.8400, "sector": "power",    "id": "SUB_BANDRA",   "name": "Bandra Substation"},
        {"lat": 19.0553, "lng": 72.9036, "sector": "power",    "id": "GEN_TROMBAY",  "name": "Tata Power Trombay Plant"},
        {"lat": 19.0650, "lng": 72.8350, "sector": "water",    "id": "PUMP_WORLI",   "name": "Worli Pumping Station"},
        {"lat": 19.1050, "lng": 72.8700, "sector": "water",    "id": "PUMP_MATUNGA", "name": "Matunga Pump Station"},
        {"lat": 19.0800, "lng": 72.8550, "sector": "water",    "id": "WTWR_SION",    "name": "Sion Water Tower"},
        {"lat": 19.0012, "lng": 72.8402, "sector": "hospital", "id": "HOSP_KEM",     "name": "KEM Hospital Parel"},
        {"lat": 19.0406, "lng": 72.8696, "sector": "hospital", "id": "HOSP_SION",    "name": "Sion Municipal Hospital"},
        {"lat": 19.0019, "lng": 72.8337, "sector": "hospital", "id": "HOSP_NAIR",    "name": "Nair Hospital"},
        {"lat": 19.1800, "lng": 72.9600, "sector": "hospital", "id": "HOSP_THANE",   "name": "Thane Civil Hospital"},
        {"lat": 19.1000, "lng": 72.8200, "sector": "telecom",  "id": "MAST_BANDRA",  "name": "Bandra Telecom Mast"},
        {"lat": 19.0500, "lng": 72.9000, "sector": "telecom",  "id": "TWR_SOUTH",    "name": "South Mumbai Relay Tower"},
    ],
    "bangalore": [
        {"lat": 12.9716, "lng": 77.5946, "sector": "power",    "id": "SUB_CENTRAL",   "name": "Central Substation"},
        {"lat": 13.0200, "lng": 77.5900, "sector": "power",    "id": "SUB_NORTH",     "name": "North Bangalore Substation"},
        {"lat": 12.9500, "lng": 77.5800, "sector": "power",    "id": "GEN_SOUTH",     "name": "Southern Power Generator"},
        {"lat": 12.9600, "lng": 77.5700, "sector": "water",    "id": "PUMP_CAUVERY",  "name": "Cauvery Pump Station"},
        {"lat": 12.9900, "lng": 77.6100, "sector": "water",    "id": "PUMP_EAST",     "name": "East Pump Station"},
        {"lat": 12.9800, "lng": 77.5800, "sector": "water",    "id": "WTWR_MG",       "name": "MG Road Water Tower"},
        {"lat": 12.9698, "lng": 77.6064, "sector": "hospital", "id": "HOSP_VICTORIA", "name": "Victoria Hospital"},
        {"lat": 12.9950, "lng": 77.5710, "sector": "hospital", "id": "HOSP_NIMHANS",  "name": "NIMHANS Hospital"},
        {"lat": 13.0100, "lng": 77.5500, "sector": "hospital", "id": "HOSP_NORTH",    "name": "North Bangalore Hospital"},
        {"lat": 12.9300, "lng": 77.6200, "sector": "hospital", "id": "HOSP_SOUTH",    "name": "South Bangalore Hospital"},
        {"lat": 12.9800, "lng": 77.5500, "sector": "telecom",  "id": "MAST_INDIRA",   "name": "Indiranagar Telecom Mast"},
        {"lat": 12.9600, "lng": 77.6400, "sector": "telecom",  "id": "TWR_KORAMANGALA","name": "Koramangala Tower"},
    ],
    "delhi": [
        {"lat": 28.6139, "lng": 77.2090, "sector": "power",    "id": "SUB_CONNAUGHT", "name": "Connaught Place Substation"},
        {"lat": 28.6800, "lng": 77.2200, "sector": "power",    "id": "SUB_NORTH",     "name": "North Delhi Substation"},
        {"lat": 28.5500, "lng": 77.2000, "sector": "power",    "id": "GEN_INDRA",     "name": "Indraprastha Power Plant"},
        {"lat": 28.6000, "lng": 77.1800, "sector": "water",    "id": "PUMP_YAMUNA",   "name": "Yamuna Pump Station"},
        {"lat": 28.6500, "lng": 77.2500, "sector": "water",    "id": "PUMP_EAST",     "name": "East Delhi Pump Station"},
        {"lat": 28.6200, "lng": 77.2000, "sector": "water",    "id": "WTWR_CENTRAL",  "name": "Central Water Tower"},
        {"lat": 28.6400, "lng": 77.2000, "sector": "hospital", "id": "HOSP_AIIMS",    "name": "AIIMS New Delhi"},
        {"lat": 28.6517, "lng": 77.2219, "sector": "hospital", "id": "HOSP_SAFDAR",   "name": "Safdarjung Hospital"},
        {"lat": 28.6750, "lng": 77.2200, "sector": "hospital", "id": "HOSP_GTB",      "name": "GTB Hospital"},
        {"lat": 28.5900, "lng": 77.2100, "sector": "hospital", "id": "HOSP_SOUTH",    "name": "South Delhi Hospital"},
        {"lat": 28.6300, "lng": 77.2100, "sector": "telecom",  "id": "MAST_CP",       "name": "Connaught Place Mast"},
        {"lat": 28.6600, "lng": 77.2300, "sector": "telecom",  "id": "TWR_NORTH",     "name": "North Delhi Tower"},
    ],
    "nyc": [
        {"lat": 40.7128, "lng": -74.0060, "sector": "power",    "id": "SUB_MANHATTAN", "name": "Manhattan Substation"},
        {"lat": 40.7500, "lng": -73.9500, "sector": "power",    "id": "SUB_QUEENS",    "name": "Queens Substation"},
        {"lat": 40.6800, "lng": -73.9500, "sector": "power",    "id": "GEN_BROOKLYN",  "name": "Brooklyn Generator"},
        {"lat": 40.7200, "lng": -73.9900, "sector": "water",    "id": "PUMP_HUDSON",   "name": "Hudson River Pump Station"},
        {"lat": 40.7400, "lng": -73.9300, "sector": "water",    "id": "PUMP_BRONX",    "name": "Bronx Pump Station"},
        {"lat": 40.7050, "lng": -74.0150, "sector": "water",    "id": "WTWR_LOWER",    "name": "Lower Manhattan Tower"},
        {"lat": 40.7635, "lng": -73.9540, "sector": "hospital", "id": "HOSP_BELLEVUE", "name": "Bellevue Hospital"},
        {"lat": 40.7900, "lng": -73.9500, "sector": "hospital", "id": "HOSP_COLUMBIA", "name": "Columbia Presbyterian"},
        {"lat": 40.7000, "lng": -73.9800, "sector": "hospital", "id": "HOSP_BROOKLYN", "name": "Brooklyn Medical Center"},
        {"lat": 40.7300, "lng": -73.9400, "sector": "hospital", "id": "HOSP_QUEENS",   "name": "Queens Hospital Center"},
        {"lat": 40.7484, "lng": -73.9967, "sector": "telecom",  "id": "MAST_EMPIRE",   "name": "Empire State Antenna"},
        {"lat": 40.6892, "lng": -74.0445, "sector": "telecom",  "id": "TWR_LIBERTY",   "name": "Liberty Tower"},
    ],
    "tokyo": [
        {"lat": 35.6762, "lng": 139.6503, "sector": "power",    "id": "SUB_SHIBUYA",   "name": "Shibuya Substation"},
        {"lat": 35.7090, "lng": 139.7320, "sector": "power",    "id": "SUB_UENO",      "name": "Ueno Substation"},
        {"lat": 35.6500, "lng": 139.7500, "sector": "power",    "id": "GEN_SHINAGAWA", "name": "Shinagawa Power Station"},
        {"lat": 35.6700, "lng": 139.7100, "sector": "water",    "id": "PUMP_SUMIDA",   "name": "Sumida River Pump Station"},
        {"lat": 35.7000, "lng": 139.7700, "sector": "water",    "id": "PUMP_ARAKAWA",  "name": "Arakawa Pump Station"},
        {"lat": 35.6800, "lng": 139.7400, "sector": "water",    "id": "WTWR_SHINJUKU", "name": "Shinjuku Water Tower"},
        {"lat": 35.7023, "lng": 139.7731, "sector": "hospital", "id": "HOSP_TOKYO_U",  "name": "Tokyo University Hospital"},
        {"lat": 35.6894, "lng": 139.6917, "sector": "hospital", "id": "HOSP_KEIO",     "name": "Keio University Hospital"},
        {"lat": 35.7200, "lng": 139.7500, "sector": "hospital", "id": "HOSP_JIKEI",    "name": "Jikei University Hospital"},
        {"lat": 35.6600, "lng": 139.7300, "sector": "hospital", "id": "HOSP_SHOWA",    "name": "Showa University Hospital"},
        {"lat": 35.6586, "lng": 139.7454, "sector": "telecom",  "id": "MAST_TOKYO",    "name": "Tokyo Tower"},
        {"lat": 35.7101, "lng": 139.8107, "sector": "telecom",  "id": "TWR_SKYTREE",   "name": "Tokyo Skytree"},
    ],
}


def build_edges(nodes):
    by_sector = {}
    for n in nodes:
        by_sector.setdefault(n["sector"], []).append(n["id"])
    edges = set()
    pw = by_sector.get("power", [])
    wa = by_sector.get("water", [])
    ho = by_sector.get("hospital", [])
    te = by_sector.get("telecom", [])
    for i in range(len(pw) - 1):
        edges.add((pw[i], pw[i + 1]))
    if len(pw) >= 3:
        edges.add((pw[0], pw[2]))
    if pw:
        for x in wa[:1]:  edges.add((pw[0], x))
        for x in ho[:1]:  edges.add((pw[0], x))
        for x in te[:1]:  edges.add((pw[0], x))
        edges.add((pw[0], "AI_CORE"))
    for i in range(len(wa) - 1): edges.add((wa[i], wa[i + 1]))
    for i in range(len(ho) - 1): edges.add((ho[i], ho[i + 1]))
    for i in range(len(te) - 1): edges.add((te[i], te[i + 1]))
    for lst in [wa[:1], ho[:1], te[:1]]:
        for x in lst: edges.add(("AI_CORE", x))
    return [list(e) for e in edges]


def main():
    output = {}
    for city, raw in FALLBACK.items():
        lats = [n["lat"] for n in raw]
        lngs = [n["lng"] for n in raw]
        ai_lat = round(sum(lats) / len(lats), 6)
        ai_lng = round(sum(lngs) / len(lngs), 6)
        nodes = [{
            "id":               n["id"],
            "lat":              n["lat"],
            "lng":              n["lng"],
            "sector":           n["sector"],
            "name":             n["name"],
            "osm_id":           None,
            "service_radius_km": RADIUS.get(n["sector"], 15.0),
        } for n in raw]
        nodes.append({
            "id": "AI_CORE", "lat": ai_lat, "lng": ai_lng,
            "sector": "ai", "name": "CascadeGuard AI Core",
            "osm_id": None, "service_radius_km": 5.0,
        })
        edges = build_edges(nodes)
        output[city] = {
            "nodes":      nodes,
            "edges":      edges,
            "center":     CENTERS[city],
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
        print(f"  {city}: {len(nodes)} nodes, {len(edges)} edges")

    out_path = Path(__file__).parent.parent / "data" / "real_nodes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()

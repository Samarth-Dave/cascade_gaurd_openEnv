import asyncio, json, websockets

async def test():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "task_osm_london"}}))
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        msg = json.loads(raw)
        wrapper = msg.get("data", {})
        obs = wrapper.get("observation", wrapper)
        nodes = obs.get("nodes", [])
        ss = obs.get("sector_summary", {})
        print("Task: task_osm_london")
        print("Step:", obs.get("step"), " Nodes:", len(nodes))
        print("Budget:", obs.get("budget_remaining"))
        print("Sector summary:", json.dumps(ss, indent=2))
        if nodes:
            n0 = nodes[0]
            print("First node:", n0.get("node_id"), "|", n0.get("real_name"), "| lat=", n0.get("lat"), "lon=", n0.get("lon"))
            lats = [n.get("lat", 0) for n in nodes]
            print("Lat range:", round(min(lats), 3), "to", round(max(lats), 3), "(London ~51.5, Mumbai ~19.0)")

asyncio.run(test())

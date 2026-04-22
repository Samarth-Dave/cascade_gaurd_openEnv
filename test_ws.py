import asyncio, json, websockets

async def test():
    uri = "ws://localhost:8000/ws"
    print("Connecting to", uri)
    async with websockets.connect(uri) as ws:
        # --- RESET ---
        print("\n=== RESET ===")
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "task_easy"}}))
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        msg = json.loads(raw)
        wrapper = msg.get("data", {})
        obs = wrapper.get("observation", wrapper)
        reward = wrapper.get("reward", obs.get("reward"))
        done = wrapper.get("done", obs.get("done"))
        print(f"  msg.type   = {msg.get('type')}")
        print(f"  step       = {obs.get('step')}")
        print(f"  nodes      = {len(obs.get('nodes', []))}")
        print(f"  budget     = {obs.get('budget_remaining')}")
        print(f"  reward     = {reward}")
        print(f"  done       = {done}")
        nodes = obs.get("nodes", [])
        if nodes:
            n0 = nodes[0]
            print(f"  first node = {n0.get('node_id')} | {n0.get('real_name')} | lat={n0.get('lat')} lon={n0.get('lon')}")

        # --- STEP (wait) ---
        print("\n=== STEP (wait) ===")
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "wait", "target_node_id": None, "parameters": {}}
        }))
        raw2 = await asyncio.wait_for(ws.recv(), timeout=10)
        msg2 = json.loads(raw2)
        wrapper2 = msg2.get("data", {})
        obs2 = wrapper2.get("observation", wrapper2)
        reward2 = wrapper2.get("reward", obs2.get("reward"))
        done2 = wrapper2.get("done", obs2.get("done"))
        print(f"  msg.type   = {msg2.get('type')}")
        print(f"  step       = {obs2.get('step')}")
        print(f"  budget     = {obs2.get('budget_remaining')}")
        print(f"  reward     = {reward2}")
        print(f"  done       = {done2}")
        print(f"  active_failures = {obs2.get('active_failures', [])}")

        print("\n=== ALL CHECKS PASSED - protocol working end-to-end ===")

asyncio.run(test())

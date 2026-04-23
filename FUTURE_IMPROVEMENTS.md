# CascadeGuard: Future System Improvements

While the current architecture robustly addresses the Meta OpenEnv Hackathon requirements, building a truly state-of-the-art infrastructure resilience system requires looking beyond immediate features. Here are systemic, architectural, and research-level improvements that can drastically enhance CascadeGuard's scope and realism.

## 1. Multi-Agent Coordination (Decentralized Defender)
**Current:** A single central agent has absolute control over all four sectors (Power, Water, Hospital, Telecom).
**Improvement:** Split the defender into **four distinct agents**, one for each sector. Give them private observations (e.g., the water agent doesn't see exact power loads, only incoming voltages) and require them to communicate via a bandwidth-limited `send_message` action. This perfectly maps to real-world siloed infrastructure operators and enables Multi-Agent RL (MARL) research.

## 2. Continuous Action formulation
**Current:** Actions are discrete (e.g., `shed_load(POWER_GEN_1)`, `wait()`).
**Improvement:** Transition to a hybrid or fully continuous action space. For example, `shed_load` should take a percentage `(0.0 to 1.0)`, and `harden` should take a continuous budget allocation. This allows the model to learn precise resource management using Proximal Policy Optimization (PPO) algorithms tailored for continuous domains.

## 3. Real-World Geospatial & Semantic Topologies
**Current:** Topologies in `tasks.py` are hard-coded networks mapped to abstract coordinates.
**Improvement:** Integrate with OpenStreetMap (OSM) or real SCADA datasets to procedurally generate city-scale infrastructure graphs. Adding semantic physical constraints (e.g. water pressure physics via EPANET or power flow via GridLAB-D) would elevate the environment into a true digital twin.

## 4. Unsupervised Representation Learning for States
**Current:** `get_state_features()` manually vectorizes the state for baseline purposes.
**Improvement:** Instead of relying purely on engineered observation arrays, train a Graph Neural Network (GNN) encoder on historical cascading events so the RL agent receives a dense, latent representation of graph stress. This unlocks zero-shot transferability to entirely novel power grid topologies unseen during training.

## 5. Visualizer Dashboard / 3D Post-Mortem
**Current:** A basic ASCII / ANSI terminal printer (`cli_display.py`).
**Improvement:** Build an interactive React + WebGL/Three.js frontend dashboard. The evaluator can upload the trajectory JSON, and the dashboard simulates the cascade spreading geographically across a glowing 3D city map, allowing judges to pause, inspect node states, and view the exact thought-process of the agent at each step.

## 6. Dynamic Threat Modeling (Learned Attacker)
**Current:** A rule-based `AdversarialAttacker` applying static heuristic logic.
**Improvement:** Co-evolve the attacker using self-play RL. Train an Attacker agent whose reward function is the inverse of the Defender's score. This would organically discover zero-day exploits in infrastructure setups and force the Defender to learn highly robust, minimax-optimal defense policies.

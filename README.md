# Angels for Charlie — Agent 7

Lux AI Season 3 agent that combines a **rule-based policy** (relic discovery, exploration, collection, sap logic) with **reinforcement learning** via a hybrid DQN. Development was iterative over several agent versions; **agent_7 is the final version** and the only one published here. Professor-provided base code (everything outside the agent folder) is **not** included in this repo and is not presented as part of this work.

---

## What This Repo Contains (Our Work - Jerome and Kira )

- **Hybrid agent** (`agent_afc_dqn.py`): Rule-based behavior with a DQN that can take over per unit with probability `epsilon_rl`. Actions are movement (stay/up/right/down/left) and optional sap; the DQN outputs a 5-D action choice. Uses experience replay, target network, and reward shaping (e.g. productive-tile detection, distance-to-relic, energy).
- **Design choices**: (1) **Hybrid** so rules handle exploration and relic coverage while the DQN learns where to stand for points. (2) **17-D per-unit state** (normalized position, energy, step fraction, relic visibility and distance, map edges, nearest ally/enemy) so the net sees a fixed-size vector. (3) **Statistical productive-tile tracking** in `update()` to credit positions near relics when team points increase and to shape rewards for staying/moving to those tiles. (4) **Sap logic** with scoring over the sap AoE, friendly-fire avoidance, and a score threshold before firing.
- **Training**: Self-play loop in `adaptive_training.py` (cycles of games, benchmark vs a baseline, early stop on degradation). Checkpoints save Q-net, target net, optimizer, and training counters; replay buffer is not saved.
- **Results/evidence**: Result tables and checkpoint notes live in **`agent_7/readme.md`** in this repo. Those tables report runs (e.g. 20 games) for Agent7 variants vs a baseline: win rates and average points. For example, Agent7-DQN80-Rules20 is reported at 90% win rate and 75.2 avg pts; Agent7-DQN100-v3_1 at 85% and 79.2 avg pts. No automated benchmark script is bundled that reproduces these without the course baseline; to reproduce, you would need the Lux AI S3 environment and a baseline opponent (see External dependencies below).

---

## What You Need to Publish (Agent-7-Only Repo)

### Files and folders that are **definitely needed**

| Item | Role |
|------|------|
| `agent_7/agent_afc_dqn.py` | Main agent (`AfcAgent`). |
| `agent_7/main.py` | Entry point to run games (train or eval). |
| `agent_7/adaptive_training.py` | Self-play training with benchmarking and early stopping. |
| `agent_7/feat_generator.py` | **Not in agent_7 today** — must be added (see Dependencies below). Provides `get_ally_arrays`, `state_for_unit` (17-D vector). |
| `agent_7/__init__.py` | So the folder can be used as a package. |
| `agent_7/agents.yaml` | Competition-style config (IMPORT_PATH / LOAD_PATH); adapt paths to your repo layout. |
| `agent_7/checkpoints/` | At least one `.pth` (e.g. `afc_dqn_v3_1.pth`) if you want “run with pretrained” without training first. |
| `requirements.txt` | At repo root; see below. |

### Optional but present in agent_7

- `improved_agent.py` — alternate agent variant (same interface, different heuristics).
- `test_agents.py`, `test_agents_improved.py`, `test_epsilon_rl.py` — evaluation scripts (depend on a baseline opponent).
- `debug_dqn.py`, `inspect_checkpoints.py`, `training_monitor.py` — inspection and monitoring.
- `readme.md` — local notes and result tables (evidence for reported metrics).

### Minimal repo tree

```
<repo_root>/
├── README.md
├── requirements.txt
└── agent_7/
    ├── __init__.py
    ├── feat_generator.py      # Local implementation (see Dependencies)
    ├── agent_afc_dqn.py
    ├── main.py
    ├── adaptive_training.py
    ├── agents.yaml
    ├── checkpoints/
    │   └── afc_dqn_v3_1.pth   # (or another .pth)
    ├── improved_agent.py      # optional
    ├── test_agents.py         # optional
    ├── test_agents_improved.py
    ├── test_epsilon_rl.py
    ├── debug_dqn.py
    ├── inspect_checkpoints.py
    ├── training_monitor.py
    └── readme.md              # Result tables / notes
```

---

## Dependencies Outside agent_7 (How to Handle Them)

Anything the agent or scripts use that lives **outside** the agent_7 folder is classified below. **Do not copy professor base code into the repo.** Use one of: **replace with a small local implementation**, or **treat as external**.

| Dependency | Where it lives in course repo | Classification | What to do |
|------------|------------------------------|----------------|------------|
| **Feature generator** (`get_ally_arrays`, `state_for_unit` 17-D) | `agents/hochstein/agent_1/feat_generator.py` (professor/course code) | **Replace with a small local implementation** | Add `agent_7/feat_generator.py` that implements the same API (numpy only; uses agent attributes `team_id`, `opp_team_id`, `max_units`, `W`, `H`, `max_energy`, `env_cfg`, and `state_dim`). In agent code, change imports from `agents.hochstein.agent_1.feat_generator` to the local module. If the invalid-unit branch returns `np.zeros(agent.state_dim, ...)`, use `(agent.state_dim or 17)` so it works before first init. |
| **DefaultAgent (baseline opponent)** | `agents/default/default.py`; uses `lux/utils.direction_to` | **External (bring-your-own base)** or **Replace with minimal local implementation** | To run without the course repo: add a small local module (e.g. `minimal_baseline.py`) with a class that implements `set_env_cfg(env_cfg)` and `act(step, obs)` returning shape `(max_units, 3)`, plus a `direction_to` helper. Then point `main.py` and `adaptive_training.py` at this module instead of `agents.default.default`. |
| **Lux AI S3 environment** (`LuxAIS3GymEnv`, `RecordEpisode`, obs/action format) | Official Lux-Design-S3 kit | **External dependency** | Install via the [Lux AI Season 3](https://github.com/Lux-AI-Challenge/Lux-Design-S3) Python kit (or pip if available). Required to run or train. |
| **luxai_s3** (Python package) | — | **External dependency (pip)** | Add to `requirements.txt`; install with the Lux S3 kit or as instructed by the competition. |
| **numpy, torch, python-dotenv** | — | **External dependency (pip)** | In `requirements.txt`. |

### requirements.txt

```
numpy
torch
luxai_s3
python-dotenv
```

`python-dotenv` is optional; `main.py` uses it for `REPLAY_SAVE_DIR` (default `./replays` otherwise).

---

## How to Run (Minimal)

Running or training **requires** the Lux AI S3 environment and a way to load the agent (and, for training/eval, an opponent). This repo does not bundle the professor base, so the following is high-level.

1. **Environment**: Install the Lux AI Season 3 Python environment and ensure `luxai_s3` (and its Gym API) is available.
2. **Local replacements**: Add `agent_7/feat_generator.py` as above and fix imports in `agent_afc_dqn.py` (and `improved_agent.py` if you ship it). For a baseline opponent, use the course repo on PYTHONPATH or a minimal local baseline.
3. **Imports in scripts**: In `main.py`, the AfcAgent import currently points at `agent_5`; for an agent_7-only repo, change it to the local agent_7 module (e.g. `from agent_7.agent_afc_dqn import AfcAgent` or equivalent given your layout).
4. **Run**: From the repo root (with PYTHONPATH set so `agent_7` and any baseline are importable), run e.g. `python agent_7/main.py`. Checkpoint paths are set at the top of `main.py`; use an existing `.pth` in `checkpoints/` for evaluation or leave load path empty to train from scratch.

Specific commands are not verifiable here without the full course environment; the above describes the external requirements and the minimal code changes needed so only agent_7 (and a local baseline, if you add one) is used.

---

## Reproducibility and Metrics

- **Seeds**: The code does not set global RNG seeds. For reproducible runs, set `random`, `numpy`, and `torch` seeds at the start of the run script.
- **Checkpoints**: They store Q-network, target network, optimizer, and training metadata (e.g. `train_steps`, `games_played`, `epsilon_dqn`, `epsilon_rl`). Replay buffer is not saved.
- **Reported metrics**: All cited result numbers come from the tables and notes in **`agent_7/readme.md`**. There is no in-repo script that recomputes those numbers without the course baseline; adding a small benchmark script and documenting how to run it would strengthen reproducibility.

---

## Licensing and Credits

- **Professor-provided / course base** (Lux framework, `lux/`, `agents/default/`, other agents, competition harness) is **not** included and is **not** presented as part of this work. It is either used as an external dependency or replaced by minimal local code (e.g. local feat_generator, minimal baseline).
- **Lux AI Season 3**: [Lux-Design-S3](https://github.com/Lux-AI-Challenge/Lux-Design-S3). Attribution and license follow the official project and your course rules.

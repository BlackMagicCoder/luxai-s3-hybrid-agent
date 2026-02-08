from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
import glob, json, os
import numpy as np  # ← ADD THIS IMPORT!

# --- Checkpoint directory fix (needed to avoid WinError 3) ---
base_dir = os.path.dirname(__file__)
ckpt_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
save_path = os.path.join(ckpt_dir, "afc_dqn_v3_1.pth")
load_path = save_path if (os.path.exists(save_path) and os.path.getsize(save_path) > 0) else None


# Function to inject player names into replay JSON metadata
def inject_names(path, name_p0, name_p1):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    md = j.setdefault("metadata", {})
    players = md.setdefault("players", {})
    players["player_0"] = name_p0
    players["player_1"] = name_p1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(j, f)


# Main function to run two agents against each other
def run_agents(player_0, player_1, replay_save_dir, games_to_play=10, train=True, save_frequency=10):

    # If replay_save_dir does not exist, create it
    if not os.path.exists(replay_save_dir):
        os.makedirs(replay_save_dir)

    # Set up Lux AI environment with recording wrapper
    env = RecordEpisode(LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_save_dir)
    # env = LuxAIS3GymEnv(numpy_output=True)

    # Set agents to eval mode if not training (e.g. set epsilon to 0 for epsilon-greedy policies, turn off dropout, etc.).
    # Requires agent to implement set_to_eval_mode() method.
    if not train:
        for agent in [player_0, player_1]:
            if hasattr(agent, "set_to_eval_mode"):
                agent.set_to_eval_mode()

    # Main loop to play multiple games
    for i in range(games_to_play):

        # Reset environment and get initial observations
        obs, info = env.reset()

        # Provide environment configuration to agents.
        # Only contains observable game parameters such as map size, max number of steps, etc.
        env_cfg = info["params"]
        player_0.set_env_cfg(env_cfg)
        player_1.set_env_cfg(env_cfg)

        # Save trained agent every n-th game.
        # Requires agent to implement save_agent() method.
        if train and i % save_frequency == 0:
            for agent in [player_0, player_1]:
                if hasattr(agent, "save_agent"):
                    agent.save_agent()

        # Main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        while not game_done:

            # Each agent takes an action
            # Requires agent to implement act() method.
            actions = dict()
            for agent in [player_0, player_1]:
                if hasattr(agent, "act"):
                    actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)
            # info["state"] contains unobservable game data that agents can't see

            # Check for end of game
            if isinstance(terminated, dict) and isinstance(truncated, dict):
                dones = {k: bool(terminated.get(k, False) or truncated.get(k, False)) for k in set(terminated) | set(truncated)}
            else:
                done_bool = bool(terminated) or bool(truncated)
                dones = {"player_0": done_bool, "player_1": done_bool}

            if dones.get("player_0", False) or dones.get("player_1", False):
                game_done = True

            # Let agents learn from this experience
            # Requires agent to implement update() method.
            if train:
                for agent in [player_0, player_1]:
                    if hasattr(agent, "update"):
                        agent.update(
                            step=step,
                            obs=obs[agent.player],
                            reward=reward[agent.player],
                            action=actions[agent.player],
                            terminated=dones.get(agent.player, False),
                            next_obs=next_obs[agent.player]
                        )

            # Move to next step
            step += 1
            obs = next_obs

        env.close()

        # **Call on_episode_end() after each game completes**
        if train:
            for agent in [player_0, player_1]:
                if hasattr(agent, "on_episode_end"):
                    agent.on_episode_end()

        # Stamp names into the replay JSON so the viewer shows them
        try:
            latest = max(glob.glob(os.path.join(replay_save_dir, "*.json")), key=os.path.getmtime)
            inject_names(latest, player_0.name, player_1.name)
        except ValueError:
            pass


if __name__ == "__main__":
    from dotenv import load_dotenv
    from agents.default.default import DefaultAgent
    from agents.TwoAngelsForCharlie.agent_5.agent_afc_dqn import AfcAgent

    load_dotenv()
    replay_save_dir = os.getenv("REPLAY_SAVE_DIR", "./replays")

    # --- Initialize agents ---
    agent_0 = DefaultAgent("player_0")

    # --- Your agent with checkpoint ---
    agent_1 = AfcAgent(
        player="player_1",
        save_path=save_path,
        load_path=load_path,  # ← load only if file exists and is non-empty
        save_every_n_games=20,
        epsilon_rl=0.8,  # 20% DQN (Hybrid) # 0.2 vorher
        epsilon_dqn=0.1, # 10% random exploration
    )

    # FORCE training mode after init
    agent_1.set_to_train_mode()  # ← Add this line!

    try:
        # --- Run games ---
        run_agents(
            agent_0,
            agent_1,
            replay_save_dir=replay_save_dir,
            train=False,  # Training mode
            games_to_play=3,  # Run more games for better learning statistics
        )

    except KeyboardInterrupt:
        print("\n\n=== Interrupted by user ===")
    finally:
        # Always save at the end
        if hasattr(agent_1, "save_checkpoint"):
            print("\n=== Saving final checkpoint ===")
            agent_1.save_checkpoint()
        print("\nDone!")
    # Visualize game replays at https://s3vis.lux-ai.org/

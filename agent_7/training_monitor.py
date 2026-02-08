"""
🔍 Training Monitor - Detect Overtraining Early
Evaluates checkpoints and finds the best performing one
"""

import os
import glob
import torch
from datetime import datetime


def evaluate_checkpoint(checkpoint_path, agent_class, n_games=10):
    """
    Quick evaluation of a checkpoint against DefaultAgent

    Returns average score
    """
    from luxai_s3.wrappers import LuxAIS3GymEnv
    from agents.default.default import DefaultAgent

    # Load agent
    agent = agent_class(
        player="player_0",
        load_path=checkpoint_path,
        save_path=None,
        epsilon_rl=1.0,  # Use DQN for evaluation
        epsilon_dqn=0.0,  # No exploration during eval
    )
    agent.set_to_eval_mode()

    baseline = DefaultAgent("player_1")

    total_score = 0
    wins = 0

    for game in range(n_games):
        env = LuxAIS3GymEnv(numpy_output=True)
        obs, info = env.reset()

        agent.set_env_cfg(info["params"])
        baseline.set_env_cfg(info["params"])

        done = False
        step = 0
        game_score = 0

        while not done:
            actions = {
                "player_0": agent.act(step=step, obs=obs["player_0"]),
                "player_1": baseline.act(step=step, obs=obs["player_1"])
            }

            next_obs, reward, terminated, truncated, info = env.step(actions)

            game_score += reward["player_0"]

            if isinstance(terminated, dict):
                done = terminated.get("player_0", False) or truncated.get("player_0", False)
            else:
                done = bool(terminated) or bool(truncated)

            obs = next_obs
            step += 1

        total_score += game_score
        if game_score > 0:
            wins += 1

        env.close()

    avg_score = total_score / n_games
    win_rate = wins / n_games

    return avg_score, win_rate


def analyze_training_progression(checkpoint_dir, agent_class):
    """
    Analyze all checkpoints to find when performance degraded
    """

    print(f"\n{'=' * 80}")
    print(f"{'TRAINING PROGRESSION ANALYSIS':^80}")
    print(f"{'=' * 80}\n")

    # Find all checkpoints
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))

    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints\n")

    # Load and sort by date/name
    checkpoint_info = []

    for cp_path in checkpoints:
        try:
            # Load checkpoint to get metadata
            cp = torch.load(cp_path, map_location='cpu')

            name = os.path.basename(cp_path)
            file_time = os.path.getmtime(cp_path)
            date = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M')

            games = cp.get('games_played', cp.get('episode', cp.get('episodes', '?')))
            steps = cp.get('train_steps', cp.get('steps', cp.get('total_steps', '?')))

            checkpoint_info.append({
                'path': cp_path,
                'name': name,
                'date': date,
                'games': games,
                'steps': steps,
                'file_time': file_time
            })

        except Exception as e:
            print(f"⚠️  Could not load {os.path.basename(cp_path)}: {e}")

    # Sort by file time (oldest to newest)
    checkpoint_info.sort(key=lambda x: x['file_time'])

    print(f"{'Checkpoint':<30} {'Games':<10} {'Steps':<12} {'Date':<20}")
    print(f"{'─' * 80}")
    for cp in checkpoint_info:
        print(f"{cp['name']:<30} {str(cp['games']):<10} {str(cp['steps']):<12} {cp['date']:<20}")

    print(f"\n{'=' * 80}")
    print(f"EVALUATING CHECKPOINTS (this may take a while...)")
    print(f"{'=' * 80}\n")

    results = []

    for i, cp in enumerate(checkpoint_info, 1):
        print(f"[{i}/{len(checkpoint_info)}] Evaluating {cp['name']}...")

        try:
            avg_score, win_rate = evaluate_checkpoint(cp['path'], agent_class, n_games=5)

            results.append({
                'name': cp['name'],
                'games': cp['games'],
                'steps': cp['steps'],
                'score': avg_score,
                'win_rate': win_rate,
                'date': cp['date']
            })

            print(f"    Score: {avg_score:.1f}, Win Rate: {win_rate * 100:.0f}%")

        except Exception as e:
            print(f"    ❌ Error: {e}")

    # Find best checkpoint
    print(f"\n{'=' * 80}")
    print(f"{'RESULTS':^80}")
    print(f"{'=' * 80}\n")

    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

    print(f"{'Rank':<6} {'Checkpoint':<30} {'Games':<10} {'Score':<10} {'Win%':<10}")
    print(f"{'─' * 80}")

    for rank, r in enumerate(results_sorted, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(
            f"{medal} {rank:<3} {r['name']:<30} {str(r['games']):<10} {r['score']:<10.1f} {r['win_rate'] * 100:<10.0f}%")

    # Detect overtraining
    print(f"\n{'=' * 80}")
    print(f"{'OVERTRAINING ANALYSIS':^80}")
    print(f"{'=' * 80}\n")

    best = results_sorted[0]
    print(f"🏆 Best checkpoint: {best['name']}")
    print(f"   Score: {best['score']:.1f}")
    print(f"   Games: {best['games']}")
    print(f"   Win Rate: {best['win_rate'] * 100:.0f}%\n")

    # Find when performance started degrading
    best_idx = results.index(best)

    if best_idx < len(results) - 1:
        print(f"⚠️  OVERTRAINING DETECTED!")
        print(f"   Performance peaked at game {best['games']}")
        print(f"   Training beyond this point degraded performance:\n")

        print(f"   {'Checkpoint':<30} {'Games':<10} {'Score':<10} {'Change':<10}")
        print(f"   {'─' * 70}")

        for i in range(best_idx, len(results)):
            r = results[i]
            change = r['score'] - best['score']
            change_str = f"{change:+.1f}" if i > best_idx else "PEAK"
            symbol = "📉" if change < -5 else "📊" if change < 0 else ""
            print(f"   {symbol} {r['name']:<30} {str(r['games']):<10} {r['score']:<10.1f} {change_str:<10}")

        print(f"\n   💡 Recommendation: Use {best['name']} for competition")
        print(f"      Stop training or implement early stopping!")
    else:
        print(f"✅ No overtraining detected")
        print(f"   Latest checkpoint is the best one")
        print(f"   You can continue training safely")

    print(f"\n{'=' * 80}\n")


def main():
    """Main function"""

    import sys

    print(f"\n{'🔍' * 40}")
    print(f"{'TRAINING MONITOR':^80}")
    print(f"{'🔍' * 40}\n")

    if len(sys.argv) < 2:
        print("Usage: python training_monitor.py <checkpoint_directory>")
        print("\nExample:")
        print("  python training_monitor.py ./agents/TwoAngelsForCharlie/agent_7/checkpoints/")
        return

    checkpoint_dir = sys.argv[1]

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Directory not found: {checkpoint_dir}")
        return

    # Determine agent class
    if 'agent_7' in checkpoint_dir:
        from agents.TwoAngelsForCharlie.agent_7.agent_afc_dqn import AfcAgent
        print(f"📦 Using agent_7.AfcAgent")
    elif 'agent_6' in checkpoint_dir:
        from agents.TwoAngelsForCharlie.agent_6.agent_afc_dqn import AfcAgent
        print(f"📦 Using agent_6.AfcAgent")
    else:
        print(f"⚠️  Could not determine agent version from path")
        print(f"   Defaulting to agent_7")
        from agents.TwoAngelsForCharlie.agent_7.agent_afc_dqn import AfcAgent

    analyze_training_progression(checkpoint_dir, AfcAgent)


if __name__ == "__main__":
    main()
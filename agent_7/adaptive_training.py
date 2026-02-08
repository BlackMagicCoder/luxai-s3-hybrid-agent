"""
🚀 Adaptive Training Loop with Early Stopping
Trains agent incrementally and automatically stops when performance degrades
- Self-play training (50 games per cycle)
- Automatic benchmarking after each cycle
- Stops after 2 consecutive degradations
- Saves best checkpoint separately
"""

import os
import sys
from datetime import datetime
from luxai_s3.wrappers import LuxAIS3GymEnv
import numpy as np


class TrainingCycleManager:
    """Manages training cycles with automatic performance tracking"""

    def __init__(
        self,
        checkpoint_dir="./checkpoints/",
        base_checkpoint="afc_dqn_selfplay.pth",
        games_per_cycle=50,
        benchmark_games=10,
        patience=2,
        min_improvement=2.0,
        max_total_games=500
    ):
        """
        Initialize training cycle manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            base_checkpoint: Starting checkpoint name
            games_per_cycle: Games to train per cycle
            benchmark_games: Games to benchmark performance
            patience: Number of consecutive degradations before stopping
            min_improvement: Minimum improvement to consider "better"
            max_total_games: Safety limit for total training games
        """
        self.checkpoint_dir = checkpoint_dir
        self.base_checkpoint = base_checkpoint
        self.games_per_cycle = games_per_cycle
        self.benchmark_games = benchmark_games
        self.patience = patience
        self.min_improvement = min_improvement
        self.max_total_games = max_total_games

        # Tracking
        self.current_cycle = 0
        self.total_games_trained = 0
        self.best_score = -float('inf')
        self.best_checkpoint_path = None
        self.consecutive_degradations = 0
        self.history = []

        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

    def get_checkpoint_path(self, cycle_num=None):
        """Get checkpoint path for given cycle"""
        if cycle_num is None:
            cycle_num = self.current_cycle

        if cycle_num == 0:
            return os.path.join(self.checkpoint_dir, self.base_checkpoint)
        else:
            base_name = self.base_checkpoint.replace('.pth', '')
            return os.path.join(self.checkpoint_dir, f"{base_name}_{cycle_num}.pth")

    def get_best_checkpoint_path(self):
        """Get path for best checkpoint"""
        return os.path.join(self.checkpoint_dir, "best_checkpoint.pth")

    def update_performance(self, score):
        """
        Update performance tracking and determine if training should continue

        Args:
            score: Average points from benchmark

        Returns:
            (should_continue, is_new_best)
        """
        self.history.append({
            'cycle': self.current_cycle,
            'score': score,
            'total_games': self.total_games_trained
        })

        # Check if this is a new best
        is_new_best = False
        if score > self.best_score + self.min_improvement:
            self.best_score = score
            self.best_checkpoint_path = self.get_checkpoint_path()
            self.consecutive_degradations = 0
            is_new_best = True
            print(f"  ✅ NEW BEST! Score improved to {score:.1f} pts (+{score - self.best_score + self.min_improvement:.1f})")
        elif score >= self.best_score - self.min_improvement:
            # Within tolerance, not degradation
            self.consecutive_degradations = 0
            print(f"  ➡️  Score stable at {score:.1f} pts (best: {self.best_score:.1f})")
        else:
            # Degradation detected
            self.consecutive_degradations += 1
            degradation = self.best_score - score
            print(f"  ⚠️  Performance degraded: {score:.1f} pts (-{degradation:.1f}) [Degradation #{self.consecutive_degradations}/{self.patience}]")

        # Determine if should continue
        should_continue = True

        if self.consecutive_degradations >= self.patience:
            print(f"\n  🛑 STOPPING: {self.patience} consecutive degradations detected")
            should_continue = False
        elif self.total_games_trained >= self.max_total_games:
            print(f"\n  🛑 STOPPING: Reached max training games ({self.max_total_games})")
            should_continue = False

        return should_continue, is_new_best

    def print_summary(self):
        """Print final training summary"""
        print(f"\n{'=' * 80}")
        print(f"{'TRAINING SUMMARY':^80}")
        print(f"{'=' * 80}\n")

        print(f"Total Cycles: {self.current_cycle}")
        print(f"Total Games Trained: {self.total_games_trained}")
        print(f"Best Score: {self.best_score:.1f} pts")
        print(f"Best Checkpoint: {os.path.basename(self.best_checkpoint_path)}")

        print(f"\n{'Cycle History':^80}")
        print(f"{'─' * 80}")
        print(f"{'Cycle':<8} {'Games':<10} {'Score':<12} {'Status':<20}")
        print(f"{'─' * 80}")

        for entry in self.history:
            cycle = entry['cycle']
            games = entry['total_games']
            score = entry['score']

            # Determine status
            if cycle == 0 or score > self.history[cycle-1]['score'] + self.min_improvement:
                status = "✅ Improved"
            elif score >= self.history[cycle-1]['score'] - self.min_improvement if cycle > 0 else False:
                status = "➡️  Stable"
            else:
                status = "⚠️  Degraded"

            is_best = (self.best_checkpoint_path and
                      os.path.basename(self.best_checkpoint_path) == os.path.basename(self.get_checkpoint_path(cycle)))
            marker = "🏆" if is_best else "  "

            print(f"{marker} {cycle:<6} {games:<10} {score:<12.1f} {status:<20}")

        print(f"{'─' * 80}\n")


def train_self_play(agent, opponent_agent, n_games=50):
    """
    Train agent through self-play

    Args:
        agent: Agent to train (will be updated)
        opponent_agent: Opponent agent (frozen, no updates)
        n_games: Number of games to play

    Returns:
        Training statistics
    """
    print(f"\n{'─' * 80}")
    print(f"🎯 TRAINING: {n_games} games of self-play")
    print(f"{'─' * 80}\n")

    wins = 0
    losses = 0
    draws = 0

    for game_num in range(n_games):
        env = LuxAIS3GymEnv(numpy_output=True)
        obs, info = env.reset()
        env_cfg = info["params"]

        agent.set_env_cfg(env_cfg)
        opponent_agent.set_env_cfg(env_cfg)

        # Progress indicator
        if (game_num + 1) % 10 == 0:
            print(f"  Training game {game_num + 1}/{n_games}... (W:{wins} L:{losses} D:{draws})")

        step = 0
        game_done = False

        while not game_done:
            # Get actions
            actions = {
                "player_0": agent.act(step=step, obs=obs["player_0"]),
                "player_1": opponent_agent.act(step=step, obs=obs["player_1"])
            }

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)

            # Update only the training agent
            if hasattr(agent, 'update'):
                agent.update(
                    step=step,
                    obs=obs["player_0"],
                    reward=reward["player_0"],
                    action=actions["player_0"],
                    terminated=bool(terminated.get("player_0", False) if isinstance(terminated, dict) else terminated),
                    next_obs=next_obs["player_0"]
                )

            # Check if done
            if isinstance(terminated, dict) and isinstance(truncated, dict):
                dones = {
                    k: bool(terminated.get(k, False) or truncated.get(k, False))
                    for k in set(terminated) | set(truncated)
                }
            else:
                done_bool = bool(terminated) or bool(truncated)
                dones = {"player_0": done_bool, "player_1": done_bool}

            if dones.get("player_0", False) or dones.get("player_1", False):
                game_done = True

            step += 1
            obs = next_obs

        # Record result
        try:
            final_score_0 = next_obs["player_0"]["team_points"][0]
            final_score_1 = next_obs["player_1"]["team_points"][1]

            if final_score_0 > final_score_1:
                wins += 1
            elif final_score_0 < final_score_1:
                losses += 1
            else:
                draws += 1
        except:
            pass

        # Episode end callback
        if hasattr(agent, 'on_episode_end'):
            agent.on_episode_end()
        if hasattr(opponent_agent, 'on_episode_end'):
            opponent_agent.on_episode_end()

        env.close()

    print(f"\n  ✓ Training complete: W:{wins} L:{losses} D:{draws}")
    print(f"{'─' * 80}\n")

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / n_games if n_games > 0 else 0
    }


def benchmark_agent(agent, n_games=10):
    """
    Benchmark agent against DefaultAgent

    Args:
        agent: Agent to benchmark (in eval mode)
        n_games: Number of benchmark games

    Returns:
        Average points scored
    """
    print(f"\n{'─' * 80}")
    print(f"📊 BENCHMARKING: {n_games} games vs DefaultAgent")
    print(f"{'─' * 80}\n")

    from agents.default.default import DefaultAgent

    total_points = 0
    wins = 0

    for game_num in range(n_games):
        # Create fresh baseline opponent
        baseline = DefaultAgent("player_1")

        env = LuxAIS3GymEnv(numpy_output=True)
        obs, info = env.reset()
        env_cfg = info["params"]

        agent.set_env_cfg(env_cfg)
        baseline.set_env_cfg(env_cfg)

        if (game_num + 1) % 5 == 0:
            print(f"  Benchmark game {game_num + 1}/{n_games}...")

        step = 0
        game_done = False
        game_score = 0

        while not game_done:
            actions = {
                "player_0": agent.act(step=step, obs=obs["player_0"]),
                "player_1": baseline.act(step=step, obs=obs["player_1"])
            }

            next_obs, reward, terminated, truncated, info = env.step(actions)

            # Track score
            game_score += reward.get("player_0", 0) if isinstance(reward, dict) else 0

            # Check if done
            if isinstance(terminated, dict) and isinstance(truncated, dict):
                dones = {
                    k: bool(terminated.get(k, False) or truncated.get(k, False))
                    for k in set(terminated) | set(truncated)
                }
            else:
                done_bool = bool(terminated) or bool(truncated)
                dones = {"player_0": done_bool, "player_1": done_bool}

            if dones.get("player_0", False) or dones.get("player_1", False):
                game_done = True

            step += 1
            obs = next_obs

        # Record final score
        try:
            final_score_0 = next_obs["player_0"]["team_points"][0]
            final_score_1 = next_obs["player_1"]["team_points"][1]

            total_points += final_score_0
            if final_score_0 > final_score_1:
                wins += 1
        except:
            total_points += game_score

        # Episode end
        if hasattr(agent, 'on_episode_end'):
            agent.on_episode_end()

        env.close()

    avg_points = total_points / n_games if n_games > 0 else 0
    win_rate = wins / n_games if n_games > 0 else 0

    print(f"\n  ✓ Benchmark complete:")
    print(f"    Average Points: {avg_points:.1f}")
    print(f"    Win Rate: {win_rate*100:.0f}%")
    print(f"{'─' * 80}\n")

    return avg_points


def main():
    """Main adaptive training loop"""

    print(f"\n{'🚀' * 40}")
    print(f"{'=' * 80}")
    print(f"{'ADAPTIVE TRAINING LOOP WITH EARLY STOPPING':^80}")
    print(f"{'=' * 80}")
    print(f"{'🚀' * 40}\n")

    # ============================================================================
    # CONFIGURATION - EDIT THESE PATHS TO MATCH YOUR SETUP
    # ============================================================================

    # Option 1: Use absolute path (recommended for Windows)
    # checkpoint_dir = r"C:\Users\kira\PycharmProjects\03-luxai-final-2-angels-for-charlie\agents\TwoAngelsForCharlie\agent_7\checkpoints"

    # Option 2: Use relative path (if running from project root)
    checkpoint_dir = "./checkpoints/"

    # Base checkpoint to start from
    base_checkpoint = "afc_dqn_selfplay.pth"

    # Which agent to use
    agent_version = "agent_7"  # or "agent_6"

    # ============================================================================
    # END CONFIGURATION
    # ============================================================================

    print(f"Configuration:")
    print(f"  • Training: Self-play (50 games per cycle)")
    print(f"  • Benchmarking: 10 games vs DefaultAgent")
    print(f"  • Stopping: After 2 consecutive degradations")
    print(f"  • Agent: {agent_version}")
    print(f"  • Training epsilon_rl: 0.2 (20% DQN, 80% rules)")
    print(f"  • Eval epsilon_rl: 1.0 (100% DQN)")
    print(f"  • Checkpoint dir: {checkpoint_dir}")
    print(f"  • Starting checkpoint: {base_checkpoint}")
    print(f"\n{'=' * 80}\n")

    # Initialize manager
    manager = TrainingCycleManager(
        checkpoint_dir=checkpoint_dir,
        base_checkpoint=base_checkpoint,
        games_per_cycle=50,
        benchmark_games=10,
        patience=2,
        min_improvement=2.0,
        max_total_games=500
    )

    # Check if base checkpoint exists
    base_path = manager.get_checkpoint_path(0)
    abs_base_path = os.path.abspath(base_path)

    print(f"🔍 Looking for base checkpoint...")
    print(f"   Relative path: {base_path}")
    print(f"   Absolute path: {abs_base_path}")
    print(f"   Current directory: {os.getcwd()}\n")

    if not os.path.exists(base_path):
        print(f"❌ ERROR: Base checkpoint not found!")
        print(f"\nTried:")
        print(f"  {abs_base_path}")
        print(f"\nPossible solutions:")
        print(f"  1. Edit checkpoint_dir in the script to use absolute path:")
        print(f"     checkpoint_dir = r\"C:\\Users\\kira\\...\\agent_7\\checkpoints\"")
        print(f"  2. Run script from project root directory")
        print(f"  3. Copy {base_checkpoint} to: {checkpoint_dir}")
        print(f"\nExample absolute path (edit line 574 in script):")
        print(f"  checkpoint_dir = r\"C:\\Users\\kira\\PycharmProjects\\03-luxai-final-2-angels-for-charlie\\agents\\TwoAngelsForCharlie\\agent_7\\checkpoints\"")
        return

    print(f"✓ Found base checkpoint: {base_path}\n")

    input("Press ENTER to start training, or Ctrl+C to abort...")

    # Import agent based on version
    if agent_version == "agent_7":
        from agents.TwoAngelsForCharlie.agent_7.agent_afc_dqn import AfcAgent
    else:
        from agents.TwoAngelsForCharlie.agent_6.agent_afc_dqn import AfcAgent

    print(f"✓ Found base checkpoint: {base_path}\n")

    # Main training loop
    continue_training = True

    while continue_training:
        manager.current_cycle += 1

        print(f"\n{'🔥' * 40}")
        print(f"{'=' * 80}")
        print(f"{'TRAINING CYCLE ' + str(manager.current_cycle):^80}")
        print(f"{'=' * 80}")
        print(f"{'🔥' * 40}\n")

        # Load checkpoint for training
        load_path = manager.get_checkpoint_path(manager.current_cycle - 1)
        print(f"📂 Loading checkpoint: {os.path.basename(load_path)}")

        training_agent = AfcAgent(
            player="player_0",
            load_path=load_path,
            save_path=None,  # We'll save manually
            epsilon_rl=0.2,  # 20% DQN, 80% rules for training
            epsilon_dqn=0.05,
            dqn_lr=1e-4,
            train_freq=4,
            batch_size=128,
        )
        training_agent.set_to_train_mode()
        print(f"  ✓ Training agent loaded (epsilon_rl=0.2)\n")

        # Create frozen opponent (same checkpoint, no updates)
        opponent_agent = AfcAgent(
            player="player_1",
            load_path=load_path,
            save_path=None,
            epsilon_rl=0.2,
            epsilon_dqn=0.05,
            dqn_lr=1e-4,
        )
        opponent_agent.set_to_eval_mode()
        print(f"  ✓ Opponent agent loaded (frozen copy)\n")

        # STEP 1: Train
        print(f"{'=' * 80}")
        print(f"STEP 1: TRAINING")
        print(f"{'=' * 80}")
        train_stats = train_self_play(training_agent, opponent_agent, n_games=manager.games_per_cycle)
        manager.total_games_trained += manager.games_per_cycle

        # STEP 2: Save trained checkpoint
        save_path = manager.get_checkpoint_path(manager.current_cycle)
        print(f"{'=' * 80}")
        print(f"STEP 2: SAVING CHECKPOINT")
        print(f"{'=' * 80}\n")
        print(f"  💾 Saving as: {os.path.basename(save_path)}")
        training_agent.save_checkpoint(save_path)
        print(f"  ✓ Checkpoint saved\n")

        # STEP 3: Benchmark
        print(f"{'=' * 80}")
        print(f"STEP 3: BENCHMARKING PERFORMANCE")
        print(f"{'=' * 80}")

        # Create evaluation agent (100% DQN, no exploration)
        eval_agent = AfcAgent(
            player="player_0",
            load_path=save_path,
            save_path=None,
            epsilon_rl=1.0,  # 100% DQN for evaluation
            epsilon_dqn=0.0,  # No exploration
        )
        eval_agent.set_to_eval_mode()
        print(f"  ✓ Evaluation agent loaded (epsilon_rl=1.0, epsilon_dqn=0)\n")

        avg_score = benchmark_agent(eval_agent, n_games=manager.benchmark_games)

        # STEP 4: Update performance and decide
        print(f"{'=' * 80}")
        print(f"STEP 4: PERFORMANCE ANALYSIS")
        print(f"{'=' * 80}\n")

        continue_training, is_new_best = manager.update_performance(avg_score)

        # Save best checkpoint if new best found
        if is_new_best:
            best_path = manager.get_best_checkpoint_path()
            print(f"\n  💎 Saving as best checkpoint: {os.path.basename(best_path)}")
            import shutil
            shutil.copy2(save_path, best_path)
            print(f"  ✓ Best checkpoint updated\n")

        print(f"\n{'─' * 80}")
        if continue_training:
            print(f"  ➡️  CONTINUING: Training next cycle...")
        else:
            print(f"  🛑 STOPPING: Training complete")
        print(f"{'─' * 80}\n")

    # Print final summary
    manager.print_summary()

    print(f"\n{'=' * 80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"  Best Checkpoint: {os.path.basename(manager.best_checkpoint_path)}")
    print(f"  Best Score: {manager.best_score:.1f} pts")
    print(f"  Total Games: {manager.total_games_trained}")
    print(f"  Checkpoints saved in: {manager.checkpoint_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
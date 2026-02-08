"""
🎮 IMPROVED AGENT BENCHMARK
Tests the optimized agent against baseline and previous versions
- Tests improved agent with systematic tile testing
- 20 games per agent vs DefaultAgent
- Comparison with agent_6 and agent_7 previous versions
"""

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
import glob
import json
import os
from datetime import datetime


def inject_names(path, name_p0, name_p1):
    """Inject player names into replay JSON metadata"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        md = j.setdefault("metadata", {})
        players = md.setdefault("players", {})
        players["player_0"] = name_p0
        players["player_1"] = name_p1
        with open(path, "w", encoding="utf-8") as f:
            json.dump(j, f)
    except Exception as e:
        print(f"⚠️ Warning: Could not inject names: {e}")


class GameStats:
    """Track statistics for each agent"""

    def __init__(self, name):
        self.name = name
        self.games_played = 0
        self.games_won = 0
        self.total_points = 0
        self.match_wins = 0
        self.match_losses = 0
        self.match_draws = 0
        self.total_matches = 0
        self.game_scores = []
        self.match_scores = []  # Track individual match scores

    def add_game(self, my_score, opp_score, game_num):
        """Record a game result"""
        self.games_played += 1
        self.total_points += my_score
        won = my_score > opp_score
        if won:
            self.games_won += 1
        self.game_scores.append((game_num, my_score, opp_score, won))

    def add_match(self, my_score, opp_score):
        """Record a match result"""
        self.total_matches += 1
        self.match_scores.append(my_score)  # Track individual match scores
        if my_score > opp_score:
            self.match_wins += 1
        elif my_score < opp_score:
            self.match_losses += 1
        else:
            self.match_draws += 1

    @property
    def avg_points(self):
        return self.total_points / self.games_played if self.games_played > 0 else 0

    @property
    def avg_match_score(self):
        """Average points per individual match"""
        return sum(self.match_scores) / len(self.match_scores) if self.match_scores else 0

    @property
    def win_rate(self):
        return (self.games_won / self.games_played * 100) if self.games_played > 0 else 0

    @property
    def match_win_rate(self):
        return (self.match_wins / self.total_matches * 100) if self.total_matches > 0 else 0


def verify_agent_functionality(agent, agent_name):
    """
    Quick test to verify an agent can:
    1. Be initialized
    2. Receive environment config
    3. Call act() method without errors
    """

    print(f"\n  🔍 Verifying: {agent_name}")
    print(f"     " + "─" * 60)

    try:
        # Create a simple test environment
        env = LuxAIS3GymEnv(numpy_output=True)
        obs, info = env.reset()
        env_cfg = info["params"]

        # Set env config
        agent.set_env_cfg(env_cfg)
        print(f"     ✓ Environment config set")

        # Try to call act() for first 5 steps
        actions_returned = 0
        for step in range(5):
            try:
                action = agent.act(step=step, obs=obs[agent.player])
                if action is not None or action == {}:
                    actions_returned += 1
            except Exception as e:
                print(f"     ❌ Error calling act() at step {step}: {e}")
                env.close()
                return False

        env.close()

        print(f"     ✓ act() called successfully for 5 steps")

        # Check if agent has expected attributes
        if hasattr(agent, 'epsilon_rl'):
            print(
                f"     ✓ epsilon_rl = {agent.epsilon_rl} ({agent.epsilon_rl * 100:.0f}% DQN, {(1 - agent.epsilon_rl) * 100:.0f}% rules)")

        # Check for qnet
        if hasattr(agent, 'qnet') and agent.qnet is not None:
            print(f"     ✓ Q-network loaded")
        elif hasattr(agent, 'epsilon_rl') and agent.epsilon_rl > 0.0:
            print(f"     ⚠️  Note: epsilon_rl > 0.0 but qnet verification not possible")

        print(f"     ✅ {agent_name} is working correctly!")
        return True

    except Exception as e:
        print(f"     ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_match(player_0, player_1, replay_save_dir, games_to_play=20):
    """Run multiple games between two agents"""

    stats_0 = GameStats(player_0.name)
    stats_1 = GameStats(player_1.name)

    if not os.path.exists(replay_save_dir):
        os.makedirs(replay_save_dir)

    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True),
        save_on_close=True,
        save_on_reset=True,
        save_dir=replay_save_dir
    )

    for game_num in range(games_to_play):
        print(f"\n{'=' * 70}")
        print(f"🎮 GAME {game_num + 1}/{games_to_play}")
        print(f"{'=' * 70}")
        print(f"  {player_0.name} vs {player_1.name}")
        print(f"{'=' * 70}\n")

        obs, info = env.reset()
        env_cfg = info["params"]

        player_0.set_env_cfg(env_cfg)
        player_1.set_env_cfg(env_cfg)

        match_scores_0 = []
        match_scores_1 = []

        game_done = False
        step = 0
        current_match = 0
        match_step = 0
        last_points_0 = 0
        last_points_1 = 0

        while not game_done:
            actions = {}
            actions["player_0"] = player_0.act(step=step, obs=obs["player_0"])
            actions["player_1"] = player_1.act(step=step, obs=obs["player_1"])

            next_obs, reward, terminated, truncated, info = env.step(actions)

            match_step += 1
            if match_step >= env_cfg.get("max_steps_in_match", 100):
                try:
                    current_points_0 = next_obs["player_0"]["team_points"][0]
                    current_points_1 = next_obs["player_1"]["team_points"][1]

                    match_score_0 = current_points_0 - last_points_0
                    match_score_1 = current_points_1 - last_points_1

                    match_scores_0.append(match_score_0)
                    match_scores_1.append(match_score_1)

                    stats_0.add_match(match_score_0, match_score_1)
                    stats_1.add_match(match_score_1, match_score_0)

                    current_match += 1

                    winner = "Draw" if match_score_0 == match_score_1 else (
                        player_0.name if match_score_0 > match_score_1 else player_1.name
                    )
                    print(
                        f"  📊 Match {current_match}/3: {player_0.name}={match_score_0}, {player_1.name}={match_score_1} | Winner: {winner}")

                    last_points_0 = current_points_0
                    last_points_1 = current_points_1
                    match_step = 0

                except Exception as e:
                    print(f"  ⚠️ Could not read match scores: {e}")

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

            for agent, player_key in [(player_0, "player_0"), (player_1, "player_1")]:
                if hasattr(agent, "update"):
                    agent.update(
                        step=step,
                        obs=obs[player_key],
                        reward=reward[player_key],
                        action=actions[player_key],
                        terminated=dones.get(player_key, False),
                        next_obs=next_obs[player_key]
                    )

            step += 1
            obs = next_obs

        total_score_0 = sum(match_scores_0)
        total_score_1 = sum(match_scores_1)

        stats_0.add_game(total_score_0, total_score_1, game_num + 1)
        stats_1.add_game(total_score_1, total_score_0, game_num + 1)

        print(f"\n🏁 GAME {game_num + 1} COMPLETE:")
        print(f"  {player_0.name}: {total_score_0:4d} points {'🏆' if total_score_0 > total_score_1 else ''}")
        print(f"  {player_1.name}: {total_score_1:4d} points {'🏆' if total_score_1 > total_score_0 else ''}")

        for agent in [player_0, player_1]:
            if hasattr(agent, "on_episode_end"):
                agent.on_episode_end()

        try:
            latest = max(
                glob.glob(os.path.join(replay_save_dir, "*.json")),
                key=os.path.getmtime
            )
            inject_names(latest, player_0.name, player_1.name)
        except ValueError:
            pass

    env.close()

    return stats_0, stats_1


def print_results(all_stats):
    """Print final results table with improved agent analysis"""

    print(f"\n\n{'🏆' * 35}")
    print(f"{'=' * 70}")
    print(f"{'FINAL RESULTS - IMPROVED AGENT BENCHMARK':^70}")
    print(f"{'=' * 70}")
    print(f"{'🏆' * 35}\n")

    sorted_stats = sorted(
        all_stats,
        key=lambda s: (s.avg_match_score, s.win_rate),  # Sort by avg match score first
        reverse=True
    )

    print(f"{'Rank':<6} {'Agent':<35} {'Games':<8} {'Wins':<6} {'Win%':<8} {'Avg Pts/Match':<15} {'Total Pts':<12}")
    print(f"{'─' * 95}")

    for rank, stats in enumerate(sorted_stats, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "

        print(f"{medal} {rank:<3} {stats.name:<35} {stats.games_played:<8} "
              f"{stats.games_won:<6} {stats.win_rate:>6.1f}%  "
              f"{stats.avg_match_score:>13.1f}  {stats.avg_points:>10.1f}")

    print(f"{'─' * 95}\n")

    # Analysis section
    print(f"\n{'📊' * 35}")
    print(f"{'=' * 70}")
    print(f"{'PERFORMANCE ANALYSIS':^70}")
    print(f"{'=' * 70}\n")

    # Find improved agent
    improved = next((s for s in sorted_stats if "IMPROVED" in s.name.upper()), None)
    baseline = next((s for s in sorted_stats if "Default" in s.name), None)
    agent7 = next((s for s in sorted_stats if "Agent7" in s.name and "IMPROVED" not in s.name.upper()), None)
    agent6 = next((s for s in sorted_stats if "Agent6" in s.name), None)

    if improved:
        print(f"⭐ IMPROVED AGENT RESULTS:")
        print(f"   Name: {improved.name}")
        print(f"   Avg Points/Match: {improved.avg_match_score:.1f}")
        print(f"   Win Rate: {improved.win_rate:.1f}%")
        print(f"   Total Games: {improved.games_played}")
        print(f"   Total Matches: {improved.total_matches}\n")

        # Compare to baseline
        if baseline:
            improvement_vs_baseline = improved.avg_match_score - baseline.avg_match_score
            improvement_pct = (
                        improvement_vs_baseline / baseline.avg_match_score * 100) if baseline.avg_match_score > 0 else 0

            print(f"📈 VS BASELINE (DefaultAgent):")
            print(f"   Baseline: {baseline.avg_match_score:.1f} pts/match")
            print(f"   Improved: {improved.avg_match_score:.1f} pts/match")
            print(f"   Difference: {improvement_vs_baseline:+.1f} pts ({improvement_pct:+.1f}%)")

            if improvement_pct > 50:
                print(f"   Status: 🚀 MAJOR IMPROVEMENT!")
            elif improvement_pct > 20:
                print(f"   Status: ✅ SIGNIFICANT IMPROVEMENT")
            elif improvement_pct > 0:
                print(f"   Status: ⬆️ Modest improvement")
            else:
                print(f"   Status: ⚠️ Needs work")
            print()

        # Compare to agent7
        if agent7:
            improvement_vs_agent7 = improved.avg_match_score - agent7.avg_match_score
            improvement_pct_7 = (
                        improvement_vs_agent7 / agent7.avg_match_score * 100) if agent7.avg_match_score > 0 else 0

            print(f"📊 VS PREVIOUS VERSION (Agent7):")
            print(f"   Agent7: {agent7.avg_match_score:.1f} pts/match")
            print(f"   Improved: {improved.avg_match_score:.1f} pts/match")
            print(f"   Difference: {improvement_vs_agent7:+.1f} pts ({improvement_pct_7:+.1f}%)")

            if improvement_vs_agent7 > 0:
                print(f"   Status: ✅ Better than previous version!")
            else:
                print(f"   Status: ⚠️ Previous version was better")
            print()

        # Goal tracking
        print(f"🎯 GOAL TRACKING:")
        print(f"   Current: {improved.avg_match_score:.1f} pts/match")
        print(f"   Target: 200.0 pts/match")
        remaining = 200.0 - improved.avg_match_score
        progress = (improved.avg_match_score / 200.0 * 100) if improved.avg_match_score > 0 else 0
        print(f"   Progress: {progress:.1f}%")
        print(f"   Remaining: {remaining:.1f} pts")

        if improved.avg_match_score >= 200:
            print(f"   Status: 🎉 GOAL ACHIEVED!")
        elif improved.avg_match_score >= 150:
            print(f"   Status: 🔥 Getting close!")
        elif improved.avg_match_score >= 100:
            print(f"   Status: 📈 Good progress")
        elif improved.avg_match_score >= 70:
            print(f"   Status: ⬆️ Improving")
        else:
            print(f"   Status: 🔧 Needs optimization")
        print()

        # Match score distribution
        if improved.match_scores:
            best_match = max(improved.match_scores)
            worst_match = min(improved.match_scores)
            median_match = sorted(improved.match_scores)[len(improved.match_scores) // 2]

            print(f"📊 MATCH SCORE DISTRIBUTION:")
            print(f"   Best Match: {best_match:.1f} pts")
            print(f"   Median Match: {median_match:.1f} pts")
            print(f"   Average Match: {improved.avg_match_score:.1f} pts")
            print(f"   Worst Match: {worst_match:.1f} pts")
            print(f"   Range: {best_match - worst_match:.1f} pts")

            # Count matches above 100, 150, 200
            above_100 = sum(1 for s in improved.match_scores if s >= 100)
            above_150 = sum(1 for s in improved.match_scores if s >= 150)
            above_200 = sum(1 for s in improved.match_scores if s >= 200)

            total_matches = len(improved.match_scores)
            print(f"\n   Matches ≥ 100 pts: {above_100}/{total_matches} ({above_100 / total_matches * 100:.1f}%)")
            print(f"   Matches ≥ 150 pts: {above_150}/{total_matches} ({above_150 / total_matches * 100:.1f}%)")
            print(f"   Matches ≥ 200 pts: {above_200}/{total_matches} ({above_200 / total_matches * 100:.1f}%)")

    print(f"\n{'=' * 70}\n")


def create_agent(agent_name, checkpoint_path=None, epsilon_rl=0.0, agent_type="improved"):
    """
    Create an agent based on name and configuration

    Args:
        agent_name: Name for the agent
        checkpoint_path: Path to checkpoint file (for DQN agents)
        epsilon_rl: Probability of using DQN (1.0 = 100% DQN, 0.0 = 100% rules)
        agent_type: "improved", "agent_7", "agent_6", or "default"
    """

    if agent_type == "default" or "DefaultAgent" in agent_name:
        from agents.default.default import DefaultAgent
        agent = DefaultAgent("player_0")
        agent.name = agent_name
        return agent

    # Import the correct agent version
    if agent_type == "improved":
        # Import from improved_agent.py
        from agents.TwoAngelsForCharlie.agent_7.improved_agent import AfcAgent
    elif agent_type == "agent_7":
        from agents.TwoAngelsForCharlie.agent_7.agent_afc_dqn import AfcAgent
    elif agent_type == "agent_6":
        from agents.TwoAngelsForCharlie.agent_6.agent_afc_dqn import AfcAgent
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    load_path = checkpoint_path if (checkpoint_path and os.path.exists(checkpoint_path)) else None

    agent = AfcAgent(
        player="player_0",
        load_path=load_path,
        save_path=None,
        save_every_n_games=999999,
        dqn_lr=1e-4,
        epsilon_rl=epsilon_rl,
        epsilon_dqn=0.05,
        train_freq=4,
        batch_size=128,
    )

    agent.set_to_eval_mode()
    agent.name = agent_name

    return agent


def main():
    """Main benchmark function"""

    print(f"\n{'=' * 70}")
    print(f"{'⭐ IMPROVED AGENT BENCHMARK ⭐':^70}")
    print(f"{'=' * 70}")
    print(f"  Testing optimized agent with:")
    print(f"  • Systematic 5x5 grid testing")
    print(f"  • Lower confidence threshold (0.3 vs 0.5)")
    print(f"  • Simplified collector/explorer ratio")
    print(f"  • Extended search range (20 tiles)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    # ============================================================================
    # CONFIGURATION - Testing improved agent vs baselines
    # ============================================================================

    test_agents = [
        # ⭐ NEW IMPROVED AGENT ⭐
        {
            'name': '⭐ IMPROVED Agent7 (Optimized)',
            'checkpoint': './agents/TwoAngelsForCharlie/agent_7/checkpoints/afc_dqn_selfplay_2.pth',
            'epsilon_rl': 0.0,  # 100% rules for now (focus on rule improvements)
            'agent_type': 'improved'
        },

        # Original Agent7 for comparison
        {
            'name': 'Agent7 (Original)',
            'checkpoint': './agents/TwoAngelsForCharlie/agent_7/checkpoints/afc_dqn_selfplay_2.pth',
            'epsilon_rl': 0.0,
            'agent_type': 'agent_7'
        },

        # Agent6 for reference
        {
            'name': 'Agent6 (Previous)',
            'checkpoint': './agents/TwoAngelsForCharlie/agent_6/checkpoints/afc_dqn_selfplay_2.pth',
            'epsilon_rl': 0.0,
            'agent_type': 'agent_6'
        },
    ]

    games_per_matchup = 20
    replay_base = "./test_replays_improved"

    # ============================================================================
    # END CONFIGURATION
    # ============================================================================

    os.makedirs(replay_base, exist_ok=True)

    print(f"📋 AGENTS TO TEST:")
    for i, config in enumerate(test_agents, 1):
        print(f"  {i}. {config['name']:<40} (Type: {config['agent_type']})")
    print()

    # ============================================================================
    # PRE-FLIGHT CHECK
    # ============================================================================

    print(f"\n{'🔧' * 35}")
    print(f"{'=' * 70}")
    print(f"{'PRE-FLIGHT CHECK: Verifying All Agents':^70}")
    print(f"{'=' * 70}")
    print(f"{'🔧' * 35}")

    all_agents_ok = True

    # Test DefaultAgent first
    print(f"\n{'─' * 70}")
    print(f"  Testing DefaultAgent (Baseline)")
    print(f"{'─' * 70}")

    try:
        baseline_test = create_agent("DefaultAgent (Baseline)", agent_type="default")
        baseline_test.player = "player_0"
        if not verify_agent_functionality(baseline_test, "DefaultAgent"):
            print(f"\n  ❌ DefaultAgent verification FAILED!")
            all_agents_ok = False
        else:
            print(f"  ✅ DefaultAgent verified successfully!")
    except Exception as e:
        print(f"\n  ❌ Error creating DefaultAgent: {e}")
        import traceback
        traceback.print_exc()
        all_agents_ok = False

    # Test each configured agent
    for i, agent_config in enumerate(test_agents, 1):
        print(f"\n{'─' * 70}")
        print(f"  Testing Agent {i}/{len(test_agents)}: {agent_config['name']}")
        print(f"{'─' * 70}")

        try:
            test_agent = create_agent(
                agent_config['name'],
                checkpoint_path=agent_config.get('checkpoint'),
                epsilon_rl=agent_config['epsilon_rl'],
                agent_type=agent_config['agent_type']
            )
            test_agent.player = "player_0"

            if not verify_agent_functionality(test_agent, agent_config['name']):
                print(f"\n  ❌ {agent_config['name']} verification FAILED!")
                all_agents_ok = False
            else:
                print(f"  ✅ {agent_config['name']} verified successfully!")

        except Exception as e:
            print(f"\n  ❌ Error creating {agent_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_agents_ok = False

    print(f"\n{'=' * 70}")
    if all_agents_ok:
        print(f"  ✅ PRE-FLIGHT CHECK PASSED!")
        print(f"  All agents are working correctly. Starting benchmark...")
    else:
        print(f"  ❌ PRE-FLIGHT CHECK FAILED!")
        print(f"  Please fix the issues above before running the benchmark.")
        print(f"  Aborting benchmark.")
        return
    print(f"{'=' * 70}\n")

    # Pause to let user review
    input("\nPress ENTER to continue with the full benchmark, or Ctrl+C to abort...")

    # ============================================================================
    # RUN FULL BENCHMARK
    # ============================================================================

    all_stats = []

    # Create baseline agent once for stats tracking
    baseline_stats = GameStats("DefaultAgent (Baseline)")

    # Test each agent against baseline
    for i, agent_config in enumerate(test_agents, 1):
        print(f"\n{'🔥' * 35}")
        print(f"{'=' * 70}")
        print(f"  MATCHUP {i}/{len(test_agents)}")
        print(f"{'=' * 70}")
        print(f"  {agent_config['name']}")
        print(f"    vs")
        print(f"  DefaultAgent (Baseline)")
        print(f"{'=' * 70}")
        print(f"{'🔥' * 35}\n")

        # Create test agent
        print(f"🔧 Initializing agents...")
        test_agent = create_agent(
            agent_config['name'],
            checkpoint_path=agent_config.get('checkpoint'),
            epsilon_rl=agent_config['epsilon_rl'],
            agent_type=agent_config['agent_type']
        )
        test_agent.player = "player_0"

        # Create fresh baseline for this matchup
        baseline_opponent = create_agent("DefaultAgent (Baseline)", agent_type="default")
        baseline_opponent.player = "player_1"

        print(f"  ✓ {test_agent.name} initialized")
        print(f"  ✓ {baseline_opponent.name} initialized\n")

        # Create replay directory
        matchup_dir = os.path.join(
            replay_base,
            f"matchup_{i}_{agent_config['agent_type']}"
        )
        os.makedirs(matchup_dir, exist_ok=True)

        # Run games
        stats_test, stats_baseline = run_match(
            test_agent,
            baseline_opponent,
            matchup_dir,
            games_to_play=games_per_matchup
        )

        # Accumulate stats
        all_stats.append(stats_test)

        # Merge baseline stats
        baseline_stats.games_played += stats_baseline.games_played
        baseline_stats.games_won += stats_baseline.games_won
        baseline_stats.total_points += stats_baseline.total_points
        baseline_stats.match_wins += stats_baseline.match_wins
        baseline_stats.match_losses += stats_baseline.match_losses
        baseline_stats.match_draws += stats_baseline.match_draws
        baseline_stats.total_matches += stats_baseline.total_matches
        baseline_stats.game_scores.extend(stats_baseline.game_scores)
        baseline_stats.match_scores.extend(stats_baseline.match_scores)

        print(f"\n✓ Matchup {i}/{len(test_agents)} complete!\n")

    # Add baseline to stats
    all_stats.append(baseline_stats)

    # Print final results with analysis
    print_results(all_stats)

    print(f"\n{'=' * 70}")
    print(f"  ✓ Benchmark complete!")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Replays saved to: {replay_base}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error during benchmark: {e}")
        import traceback

        traceback.print_exc()
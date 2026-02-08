"""
🧪 Epsilon_RL Verification Test
Verifies that epsilon_rl is correctly set and used in your agents
"""

import os
import numpy as np


def test_epsilon_rl():
    """Test that epsilon_rl is correctly set for different agents"""

    print(f"\n{'=' * 70}")
    print(f"{'🧪 EPSILON_RL VERIFICATION TEST':^70}")
    print(f"{'=' * 70}\n")

    # Test configurations
    test_configs = [
        {
            'name': 'Agent7-Rules100',
            'checkpoint': None,
            'epsilon_rl': 1.0,
            'agent_module': 'agents.TwoAngelsForCharlie.agent_7.agent_afc_dqn',
        },
        {
            'name': 'Agent7-DQN100-v3',
            'checkpoint': './agents/TwoAngelsForCharlie/agent_7/checkpoints/afc_dqn_v3.pth',
            'epsilon_rl': 0.0,
            'agent_module': 'agents.TwoAngelsForCharlie.agent_7.agent_afc_dqn',
        },
        {
            'name': 'Agent6-Rules100',
            'checkpoint': None,
            'epsilon_rl': 1.0,
            'agent_module': 'agents.TwoAngelsForCharlie.agent_6.agent_afc_dqn',
        },
    ]

    all_passed = True

    for i, config in enumerate(test_configs, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}/{len(test_configs)}: {config['name']}")
        print(f"{'─' * 70}")
        print(f"  Expected epsilon_rl: {config['epsilon_rl']}")
        print(f"  Checkpoint: {config['checkpoint'] if config['checkpoint'] else 'None (rules only)'}")

        try:
            # Import the correct agent module
            if 'agent_7' in config['agent_module']:
                from agents.TwoAngelsForCharlie.agent_7.agent_afc_dqn import AfcAgent
            else:
                from agents.TwoAngelsForCharlie.agent_6.agent_afc_dqn import AfcAgent

            # Create agent
            checkpoint = config['checkpoint']
            load_path = checkpoint if (checkpoint and os.path.exists(checkpoint)) else None

            agent = AfcAgent(
                player="player_0",
                load_path=load_path,
                save_path=None,
                save_every_n_games=999999,
                dqn_lr=1e-4,
                epsilon_rl=config['epsilon_rl'],
                epsilon_dqn=0.05,
                train_freq=4,
                batch_size=128,
            )

            agent.set_to_eval_mode()

            # Check 1: Verify epsilon_rl attribute
            print(f"\n  ✓ Agent created successfully")

            if hasattr(agent, 'epsilon_rl'):
                actual_epsilon = agent.epsilon_rl
                print(f"  ✓ agent.epsilon_rl exists: {actual_epsilon}")

                if abs(actual_epsilon - config['epsilon_rl']) < 0.001:
                    print(f"  ✅ PASS: epsilon_rl is correctly set to {actual_epsilon}")
                else:
                    print(f"  ❌ FAIL: epsilon_rl is {actual_epsilon}, expected {config['epsilon_rl']}")
                    all_passed = False
            else:
                print(f"  ❌ FAIL: agent.epsilon_rl attribute not found!")
                all_passed = False
                continue

            # Check 2: Verify checkpoint loading (if applicable)
            if load_path and os.path.exists(load_path):
                print(f"  ✓ Checkpoint loaded from: {load_path}")
                if hasattr(agent, 'dqn') and agent.dqn is not None:
                    print(f"  ✓ DQN model exists and is loaded")
                else:
                    print(f"  ⚠️  Warning: DQN model not found or not loaded")
            elif config['epsilon_rl'] < 1.0:
                print(f"  ⚠️  Warning: Agent uses DQN but no checkpoint loaded!")

            # Check 3: Verify behavior interpretation
            print(f"\n  Expected behavior:")
            if config['epsilon_rl'] == 1.0:
                print(f"    • 100% rules-based actions")
                print(f"    • DQN should NOT be used for action selection")
            elif config['epsilon_rl'] == 0.0:
                print(f"    • 100% DQN-based actions (with 5% random exploration)")
                print(f"    • Rules should NOT be used for action selection")
            else:
                print(f"    • {config['epsilon_rl'] * 100:.0f}% rules, {(1 - config['epsilon_rl']) * 100:.0f}% DQN")

            # Check 4: Look for the action selection logic
            print(f"\n  Checking action selection method...")
            if hasattr(agent, 'act'):
                print(f"  ✓ agent.act() method exists")

                # Try to inspect the code to see if epsilon_rl is used
                import inspect
                source = inspect.getsource(agent.act)

                if 'epsilon_rl' in source:
                    print(f"  ✓ epsilon_rl is referenced in act() method")
                else:
                    print(f"  ⚠️  Warning: epsilon_rl not found in act() method source")
                    print(f"     This might mean it's not being used correctly!")

                # Check if there's a random choice based on epsilon_rl
                if 'random' in source.lower() and 'epsilon_rl' in source:
                    print(f"  ✓ Random selection logic found (good for epsilon-greedy)")

            else:
                print(f"  ❌ FAIL: agent.act() method not found!")
                all_passed = False

            print(f"\n  {'✅ Agent configuration looks correct!' if all_passed else '⚠️  Please review warnings above'}")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"{'TEST SUMMARY':^70}")
    print(f"{'=' * 70}")

    if all_passed:
        print(f"\n  ✅ ALL TESTS PASSED!")
        print(f"  Your epsilon_rl configuration appears to be correct.")
        print(f"  You can proceed with confidence to run the benchmark.\n")
    else:
        print(f"\n  ⚠️  SOME TESTS FAILED OR HAD WARNINGS")
        print(f"  Please review the output above and verify:")
        print(f"    1. epsilon_rl is correctly set in each agent")
        print(f"    2. Checkpoints are loading correctly")
        print(f"    3. The act() method uses epsilon_rl for action selection\n")

    print(f"{'=' * 70}\n")

    return all_passed


def test_action_selection_behavior():
    """
    More detailed test: Run a few actions and track whether rules or DQN is used
    This requires instrumenting the agent code or checking internal state
    """

    print(f"\n{'=' * 70}")
    print(f"{'🔬 DETAILED ACTION SELECTION TEST':^70}")
    print(f"{'=' * 70}\n")
    print(f"This test would require modifying your agent code to add logging.")
    print(f"Suggested additions to your AfcAgent.act() method:\n")

    print("""
    # Add this in your act() method where you choose between rules and DQN:

    if np.random.random() < self.epsilon_rl:
        # Using rules
        print(f"[DEBUG] Step {step}: Using RULES (epsilon_rl={self.epsilon_rl})")
        action = self._get_rules_action(...)
    else:
        # Using DQN
        print(f"[DEBUG] Step {step}: Using DQN (epsilon_rl={self.epsilon_rl})")
        action = self._get_dqn_action(...)
    """)

    print(f"\nWith this logging, you can run a single game and verify:")
    print(f"  • epsilon_rl=1.0 → Should see only 'Using RULES' messages")
    print(f"  • epsilon_rl=0.0 → Should see only 'Using DQN' messages")
    print(f"  • epsilon_rl=0.5 → Should see roughly 50/50 split\n")


if __name__ == "__main__":
    try:
        passed = test_epsilon_rl()

        print(f"\n" + "─" * 70)
        test_action_selection_behavior()

        if not passed:
            print(f"\n⚠️  RECOMMENDATION:")
            print(f"  Before running the full benchmark, fix any issues above.")
            print(f"  You can also add debug logging to verify action selection.\n")

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
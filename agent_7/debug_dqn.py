"""
🔍 Agent DQN Debugger
Inspects agent objects to find where DQN models are stored and verify they're loaded
"""

import sys
import os
import torch
import inspect


def inspect_agent_object(agent, agent_name="Agent"):
    """Deep inspection of an agent object to find DQN models"""

    print(f"\n{'=' * 80}")
    print(f"🔍 INSPECTING: {agent_name}")
    print(f"{'=' * 80}\n")

    # Basic info
    print(f"📊 BASIC INFORMATION")
    print(f"{'─' * 80}")
    print(f"  Agent class: {type(agent).__name__}")
    print(f"  Module: {type(agent).__module__}")

    if hasattr(agent, 'player'):
        print(f"  Player: {agent.player}")
    if hasattr(agent, 'epsilon_rl'):
        print(f"  epsilon_rl: {agent.epsilon_rl}")
    if hasattr(agent, 'epsilon_dqn'):
        print(f"  epsilon_dqn: {agent.epsilon_dqn}")

    # Find all attributes
    print(f"\n📋 ALL ATTRIBUTES")
    print(f"{'─' * 80}")

    all_attrs = dir(agent)

    # Categorize attributes
    public_attrs = [a for a in all_attrs if not a.startswith('_')]
    private_attrs = [a for a in all_attrs if a.startswith('_') and not a.startswith('__')]

    print(f"  Total attributes: {len(all_attrs)}")
    print(f"  Public: {len(public_attrs)}, Private: {len(private_attrs)}")

    # Look for potential DQN/model attributes
    print(f"\n🧠 POTENTIAL MODEL ATTRIBUTES")
    print(f"{'─' * 80}")

    model_keywords = ['dqn', 'model', 'net', 'network', 'q_', 'policy', 'actor', 'critic']
    found_models = []

    for attr_name in all_attrs:
        # Check if attribute name suggests it's a model
        if any(keyword in attr_name.lower() for keyword in model_keywords):
            try:
                attr_value = getattr(agent, attr_name)
                attr_type = type(attr_value).__name__

                # Check if it's a neural network
                is_nn = isinstance(attr_value, torch.nn.Module)

                if is_nn:
                    num_params = sum(p.numel() for p in attr_value.parameters())
                    found_models.append(attr_name)
                    print(f"  ✓ {attr_name:<25} {attr_type:<20} [NN with {num_params:,} params]")
                else:
                    print(f"    {attr_name:<25} {attr_type:<20}")

            except Exception as e:
                print(f"    {attr_name:<25} (error accessing: {e})")

    if not found_models:
        print(f"  ⚠️  No neural network models found!")
    else:
        print(f"\n  ✅ Found {len(found_models)} neural network model(s): {', '.join(found_models)}")

    # Detailed inspection of found models
    if found_models:
        print(f"\n🔬 DETAILED MODEL INSPECTION")
        print(f"{'─' * 80}")

        for model_name in found_models:
            model = getattr(agent, model_name)
            print(f"\n  Model: {model_name}")
            print(f"  {'─' * 76}")

            # Model architecture
            print(f"  Architecture:")
            for name, module in model.named_children():
                print(f"    • {name}: {module}")

            # Parameter details
            print(f"\n  Parameters:")
            total_params = 0
            trainable_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                print(f"    • {name:<30} shape={tuple(param.shape):<20} trainable={param.requires_grad}")

            print(f"\n  Summary:")
            print(f"    Total parameters: {total_params:,}")
            print(f"    Trainable: {trainable_params:,}")
            print(f"    Non-trainable: {(total_params - trainable_params):,}")

            # Check if model is in eval/train mode
            print(f"    Training mode: {model.training}")

    # Check for act() method and inspect it
    print(f"\n🎯 ACTION SELECTION METHOD")
    print(f"{'─' * 80}")

    if hasattr(agent, 'act'):
        print(f"  ✓ act() method exists")

        # Try to get source code
        try:
            source = inspect.getsource(agent.act)

            # Look for key patterns
            print(f"\n  Code analysis:")

            if 'epsilon_rl' in source:
                print(f"    ✓ Uses epsilon_rl for action selection")

            if 'dqn' in source.lower():
                print(f"    ✓ References DQN in code")

            if 'random' in source.lower():
                print(f"    ✓ Has randomness/exploration")

            # Find what attributes are accessed
            import re
            self_attrs = re.findall(r'self\.(\w+)', source)
            unique_attrs = list(set(self_attrs))

            print(f"\n  Attributes accessed in act():")
            for attr in sorted(unique_attrs):
                if hasattr(agent, attr):
                    print(f"    • self.{attr}")

            # Show relevant code snippet
            print(f"\n  Code snippet (first 500 chars):")
            print(f"  ┌{'─' * 78}┐")
            for line in source[:500].split('\n')[:15]:
                print(f"  │ {line[:76]:<76} │")
            print(f"  └{'─' * 78}┘")

        except Exception as e:
            print(f"  ⚠️  Could not inspect source code: {e}")
    else:
        print(f"  ❌ No act() method found!")

    # Check for update/training methods
    print(f"\n🔄 TRAINING METHODS")
    print(f"{'─' * 80}")

    training_methods = ['update', 'train', 'learn', 'optimize', 'step']
    for method_name in training_methods:
        if hasattr(agent, method_name):
            method = getattr(agent, method_name)
            if callable(method):
                print(f"  ✓ {method_name}() method exists")

    # Check for replay buffer
    print(f"\n💾 REPLAY BUFFER")
    print(f"{'─' * 80}")

    buffer_keywords = ['replay', 'buffer', 'memory', 'experience']
    for attr_name in all_attrs:
        if any(keyword in attr_name.lower() for keyword in buffer_keywords):
            try:
                attr_value = getattr(agent, attr_name)
                if hasattr(attr_value, '__len__'):
                    print(f"  ✓ {attr_name}: {len(attr_value)} items")
                else:
                    print(f"    {attr_name}: {type(attr_value).__name__}")
            except:
                pass

    print(f"\n{'=' * 80}\n")

    return found_models


def compare_two_agents(agent1, name1, agent2, name2):
    """Compare two agents to find differences"""

    print(f"\n{'🔬' * 40}")
    print(f"{'=' * 80}")
    print(f"{'COMPARING TWO AGENTS':^80}")
    print(f"{'=' * 80}")
    print(f"{'🔬' * 40}\n")

    print(f"  Agent 1: {name1}")
    print(f"  Agent 2: {name2}\n")

    # Compare attributes
    attrs1 = set(dir(agent1))
    attrs2 = set(dir(agent2))

    common = attrs1 & attrs2
    only_1 = attrs1 - attrs2
    only_2 = attrs2 - attrs1

    print(f"📊 ATTRIBUTE COMPARISON")
    print(f"{'─' * 80}")
    print(f"  Common attributes: {len(common)}")
    print(f"  Only in {name1}: {len(only_1)}")
    print(f"  Only in {name2}: {len(only_2)}")

    if only_1:
        print(f"\n  Unique to {name1}:")
        for attr in sorted(only_1):
            if not attr.startswith('__'):
                print(f"    • {attr}")

    if only_2:
        print(f"\n  Unique to {name2}:")
        for attr in sorted(only_2):
            if not attr.startswith('__'):
                print(f"    • {attr}")

    # Compare epsilon values
    print(f"\n⚙️  CONFIGURATION COMPARISON")
    print(f"{'─' * 80}")

    config_attrs = ['epsilon_rl', 'epsilon_dqn', 'player', 'learning_rate', 'batch_size']
    for attr in config_attrs:
        val1 = getattr(agent1, attr, "N/A")
        val2 = getattr(agent2, attr, "N/A")
        match = "✓" if val1 == val2 else "✗"
        print(f"  {match} {attr:<20} {name1}: {val1:<15} {name2}: {val2}")

    print(f"\n{'=' * 80}\n")


def main():
    """Main function"""

    print(f"\n{'🔍' * 40}")
    print(f"{'=' * 80}")
    print(f"{'AGENT DQN DEBUGGER':^80}")
    print(f"{'=' * 80}")
    print(f"{'🔍' * 40}\n")

    print("This script will help you find where your DQN models are stored")
    print("and why the verification warning appears.\n")

    # Import agents
    print("🔧 Importing agents...")

    try:
        from agents.TwoAngelsForCharlie.agent_7.agent_afc_dqn import AfcAgent as Agent7
        print("  ✓ agent_7.AfcAgent imported")
    except Exception as e:
        print(f"  ❌ Could not import agent_7: {e}")
        Agent7 = None

    try:
        from agents.TwoAngelsForCharlie.agent_6.agent_afc_dqn import AfcAgent as Agent6
        print("  ✓ agent_6.AfcAgent imported")
    except Exception as e:
        print(f"  ❌ Could not import agent_6: {e}")
        Agent6 = None

    if not Agent7 and not Agent6:
        print("\n❌ Could not import any agents. Make sure you're in the project directory.")
        return

    # Create test agents
    print(f"\n🔧 Creating test agents...\n")

    test_agents = []

    if Agent7:
        # Create agent_7 with DQN checkpoint
        checkpoint_v3 = './agents/TwoAngelsForCharlie/agent_7/checkpoints/afc_dqn_v3.pth'
        if os.path.exists(checkpoint_v3):
            agent7_dqn = Agent7(
                player="player_0",
                load_path=checkpoint_v3,
                save_path=None,
                epsilon_rl=1.0,
                epsilon_dqn=0.05
            )
            agent7_dqn.set_to_eval_mode()
            test_agents.append((agent7_dqn, "Agent7-DQN100-v3"))
            print(f"  ✓ Created Agent7-DQN100-v3 with checkpoint")

        # Create agent_7 rules only
        agent7_rules = Agent7(
            player="player_0",
            load_path=None,
            save_path=None,
            epsilon_rl=0.0,
            epsilon_dqn=0.05
        )
        agent7_rules.set_to_eval_mode()
        test_agents.append((agent7_rules, "Agent7-Rules100"))
        print(f"  ✓ Created Agent7-Rules100 (no checkpoint)")

    if Agent6:
        # Create agent_6 with DQN checkpoint
        checkpoint_v3 = './agents/TwoAngelsForCharlie/agent_6/checkpoints/afc_dqn_v3.pth'
        if os.path.exists(checkpoint_v3):
            agent6_dqn = Agent6(
                player="player_0",
                load_path=checkpoint_v3,
                save_path=None,
                epsilon_rl=1.0,
                epsilon_dqn=0.05
            )
            agent6_dqn.set_to_eval_mode()
            test_agents.append((agent6_dqn, "Agent6-DQN100-v3"))
            print(f"  ✓ Created Agent6-DQN100-v3 with checkpoint")

    # Inspect each agent
    all_models = {}
    for agent, name in test_agents:
        models = inspect_agent_object(agent, name)
        all_models[name] = models

    # Summary
    print(f"\n{'=' * 80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'=' * 80}\n")

    for name, models in all_models.items():
        if models:
            print(f"✅ {name}:")
            print(f"   DQN stored in: {', '.join(models)}")
        else:
            print(f"⚠️  {name}: No DQN models found!")

    print(f"\n{'=' * 80}")
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Update your verification script to check the correct attribute name(s)")
    print(f"   found above instead of just 'dqn'.\n")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
"""
🔍 DQN Checkpoint Inspector
Analyzes .pth checkpoint files to reveal training information and metadata
"""

import torch
import os
import sys
from datetime import datetime
from pathlib import Path


def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def count_parameters(state_dict, prefix='dqn'):
    """Count parameters in model"""
    total = 0
    for key in state_dict.keys():
        if key.startswith(prefix):
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel()
    return total


def inspect_checkpoint(file_path):
    """Inspect a single checkpoint file"""

    print(f"\n{'=' * 80}")
    print(f"📁 CHECKPOINT: {os.path.basename(file_path)}")
    print(f"{'=' * 80}")

    # Basic file info
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found: {file_path}")
        return

    file_size = os.path.getsize(file_path)
    file_time = os.path.getmtime(file_path)
    file_date = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n📊 FILE INFORMATION")
    print(f"{'─' * 80}")
    print(f"  Full Path:     {os.path.abspath(file_path)}")
    print(f"  File Size:     {format_size(file_size)}")
    print(f"  Last Modified: {file_date}")

    # Load checkpoint
    try:
        print(f"\n🔄 Loading checkpoint...")
        checkpoint = torch.load(file_path, map_location='cpu')
        print(f"  ✓ Loaded successfully")
    except Exception as e:
        print(f"  ❌ ERROR loading checkpoint: {e}")
        return

    # Analyze checkpoint structure
    print(f"\n🗂️  CHECKPOINT STRUCTURE")
    print(f"{'─' * 80}")
    print(f"  Type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"  Keys found: {len(checkpoint.keys())}")
        print(f"\n  Available keys:")
        for key in sorted(checkpoint.keys()):
            value = checkpoint[key]
            value_type = type(value).__name__

            # Get size info
            if isinstance(value, torch.Tensor):
                size_info = f"shape={tuple(value.shape)}, dtype={value.dtype}"
            elif isinstance(value, dict):
                size_info = f"{len(value)} items"
            elif isinstance(value, (int, float)):
                size_info = f"value={value}"
            elif isinstance(value, str):
                size_info = f"'{value}'"
            else:
                size_info = ""

            print(f"    • {key:<25} ({value_type:<15}) {size_info}")

    # Training metadata
    print(f"\n📈 TRAINING METADATA")
    print(f"{'─' * 80}")

    metadata_keys = [
        'games_played', 'episodes', 'total_episodes', 'episode', 'n_episodes',
        'steps', 'total_steps', 'timesteps', 'train_steps',
        'epoch', 'epochs', 'iteration', 'iterations',
        'wins', 'losses', 'win_rate',
        'epsilon', 'epsilon_rl', 'epsilon_dqn',
        'learning_rate', 'lr',
        'total_reward', 'avg_reward', 'best_reward',
        'loss', 'avg_loss',
    ]

    found_metadata = False
    for key in metadata_keys:
        if key in checkpoint:
            found_metadata = True
            value = checkpoint[key]
            print(f"  {key:<20} = {value}")

    if not found_metadata:
        print(f"  ⚠️  No standard training metadata found")
        print(f"  (This might be stored in a custom format)")

    # Model information
    print(f"\n🧠 MODEL INFORMATION")
    print(f"{'─' * 80}")

    # Check for different model key names
    model_keys = ['model_state_dict', 'dqn', 'model', 'state_dict', 'network']
    model_found = False

    for model_key in model_keys:
        if model_key in checkpoint:
            model_found = True
            state_dict = checkpoint[model_key]

            print(f"  Model key: '{model_key}'")

            if isinstance(state_dict, dict):
                num_params = count_parameters(state_dict, prefix='')
                print(f"  Total parameters: {num_params:,}")

                # Show layer structure
                print(f"\n  Model layers:")
                layers = {}
                for key in state_dict.keys():
                    layer_name = key.split('.')[0]
                    if layer_name not in layers:
                        layers[layer_name] = []
                    layers[layer_name].append(key)

                for layer_name, params in sorted(layers.items()):
                    print(f"    • {layer_name:<20} ({len(params)} parameters)")
            break

    if not model_found:
        print(f"  ⚠️  No model state dict found with standard keys")
        print(f"  ℹ️  Checkpoint might be a full model or use custom keys")

    # Optimizer information
    print(f"\n⚙️  OPTIMIZER INFORMATION")
    print(f"{'─' * 80}")

    optimizer_keys = ['optimizer_state_dict', 'optimizer', 'optim']
    optimizer_found = False

    for opt_key in optimizer_keys:
        if opt_key in checkpoint:
            optimizer_found = True
            opt_state = checkpoint[opt_key]
            print(f"  Optimizer key: '{opt_key}'")

            if isinstance(opt_state, dict):
                if 'param_groups' in opt_state:
                    param_groups = opt_state['param_groups']
                    if param_groups:
                        print(f"  Learning rate: {param_groups[0].get('lr', 'N/A')}")
                        print(f"  Parameter groups: {len(param_groups)}")
            break

    if not optimizer_found:
        print(f"  ⚠️  No optimizer state found")

    # Replay buffer information
    print(f"\n💾 REPLAY BUFFER INFORMATION")
    print(f"{'─' * 80}")

    replay_keys = ['replay_buffer', 'buffer', 'memory', 'experience_replay']
    replay_found = False

    for replay_key in replay_keys:
        if replay_key in checkpoint:
            replay_found = True
            replay = checkpoint[replay_key]
            print(f"  Replay buffer key: '{replay_key}'")

            if isinstance(replay, dict):
                for key, value in replay.items():
                    if isinstance(value, (list, tuple)):
                        print(f"    {key}: {len(value)} items")
                    elif isinstance(value, torch.Tensor):
                        print(f"    {key}: shape={tuple(value.shape)}")
                    else:
                        print(f"    {key}: {value}")
            elif hasattr(replay, '__len__'):
                print(f"  Size: {len(replay)} experiences")
            break

    if not replay_found:
        print(f"  ⚠️  No replay buffer found")

    # Additional info
    print(f"\n📝 ADDITIONAL INFORMATION")
    print(f"{'─' * 80}")

    # Check for any custom metadata
    custom_keys = [k for k in checkpoint.keys() if k not in
                   ['model_state_dict', 'dqn', 'model', 'state_dict', 'network',
                    'optimizer_state_dict', 'optimizer', 'optim',
                    'replay_buffer', 'buffer', 'memory'] + metadata_keys]

    if custom_keys:
        print(f"  Custom keys found:")
        for key in custom_keys:
            value = checkpoint[key]
            if isinstance(value, (int, float, str, bool)):
                print(f"    • {key}: {value}")
            else:
                print(f"    • {key}: {type(value).__name__}")
    else:
        print(f"  No additional custom keys found")

    print(f"\n{'=' * 80}\n")


def main():
    """Main function"""

    print(f"\n{'🔍' * 40}")
    print(f"{'=' * 80}")
    print(f"{'DQN CHECKPOINT INSPECTOR':^80}")
    print(f"{'=' * 80}")
    print(f"{'🔍' * 40}\n")

    # Get file path(s) from command line or prompt
    if len(sys.argv) > 1:
        # Files provided as command line arguments
        file_paths = sys.argv[1:]
    else:
        # Prompt for file path
        print("Enter checkpoint file path(s) (one per line, empty line to finish):")
        file_paths = []
        while True:
            path = input("> ").strip()
            if not path:
                break
            file_paths.append(path)

    if not file_paths:
        print("❌ No files provided. Usage:")
        print("   python inspect_checkpoint.py <checkpoint1.pth> [checkpoint2.pth] ...")
        print("\nOr run without arguments to enter paths interactively.")
        return

    # Expand wildcards and paths
    expanded_paths = []
    for path in file_paths:
        # Handle wildcards
        if '*' in path or '?' in path:
            import glob
            expanded_paths.extend(glob.glob(path))
        else:
            expanded_paths.append(path)

    # Inspect each checkpoint
    for file_path in expanded_paths:
        inspect_checkpoint(file_path)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"✅ Inspected {len(expanded_paths)} checkpoint(s)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
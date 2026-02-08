import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple
import agents.hochstein.agent_1.feat_generator as fg

# ----------------------------
# Utility Functions
# ----------------------------
def direction_to(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    """
    Calculate direction from from_pos to to_pos.

    Args:
        from_pos: Current position (x, y)
        to_pos: Target position (x, y)

    Returns:
        Direction: 0=stay, 1=up, 2=right, 3=down, 4=left
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    # Choose direction based on larger delta
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2  # Right
        elif dx < 0:
            return 4  # Left
    else:
        if dy > 0:
            return 3  # Down
        elif dy < 0:
            return 1  # Up

    return 0  # Stay if already at target


def _direction_to(self, from_pos, to_pos):
    """
    Calculate direction from from_pos to to_pos.

    Args:
        from_pos: Current position (x, y)
        to_pos: Target position (x, y)

    Returns:
        Direction: 0=stay, 1=up, 2=right, 3=down, 4=left
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    # Choose direction based on larger delta
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2  # Right
        elif dx < 0:
            return 4  # Left
    else:
        if dy > 0:
            return 3  # Down
        elif dy < 0:
            return 1  # Up

    return 0  # Stay if already at target


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling training transitions.
    Stores transitions as numpy arrays for memory efficiency.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, done):
        """
        Add a transition to the buffer.

        Args:
            s: Current state (numpy array)
            a: Action taken (int)
            r: Reward received (float)
            ns: Next state (numpy array)
            done: Whether episode terminated (float, 0 or 1)
        """
        # Store raw numpy arrays / scalars for memory efficiency
        self.buffer.append((s.astype(np.float32), int(a), float(r), ns.astype(np.float32), float(done)))

    def sample(self, batch_size: int):
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.stack(states, axis=0), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.stack(next_states, axis=0), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


# ----------------------------
# Q-Network (UNCHANGED - preserves learned weights)
# ----------------------------
class QNet(nn.Module):
    """
    Q-Network for approximating action-value function Q(s,a).
    Simple feedforward neural network with 2 hidden layers.
    """

    def __init__(self, obs_dim: int, n_actions: int = 5):
        """
        Initialize the Q-Network.

        Args:
            obs_dim: Dimension of state observation vector
            n_actions: Number of possible actions (5 for movement directions)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),  # ⭐ 128 → 256
            nn.ReLU(),
            nn.Linear(256, 256),  # ⭐ Mehr Kapazität
            nn.ReLU(),
            nn.Linear(256, 256),  # ⭐ Extra Layer
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input state tensor

        Returns:
            Q-values for each action
        """
        return self.net(x)

class SimpleMetrics:
    """Lightweight metrics tracker."""

    def __init__(self, window_size=100):
        from collections import deque
        self.rewards = deque(maxlen=window_size)
        self.q_values = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.dqn_actions = 0
        self.rule_actions = 0
        self.sap_actions = 0
        self.total_reward = 0.0
        self.episodes = 0
        self.training_steps = 0

    def update(self, reward=None, q_value=None, loss=None,
               action_type=None, sapped=False):
        if reward: self.rewards.append(reward); self.total_reward += reward
        if q_value: self.q_values.append(q_value)
        if loss: self.losses.append(loss); self.training_steps += 1
        if action_type == 'dqn':
            self.dqn_actions += 1
        elif action_type == 'rule':
            self.rule_actions += 1
        if sapped: self.sap_actions += 1

    def reset_episode(self):
        self.dqn_actions = self.rule_actions = self.sap_actions = 0
        self.total_reward = 0.0
        self.episodes += 1

    def get_avg_reward(self):
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def get_avg_q(self):
        return sum(self.q_values) / len(self.q_values) if self.q_values else 0.0

    def get_avg_loss(self):
        return sum(self.losses) / len(self.losses) if self.losses else 0.0


# ----------------------------
# AfcAgent with Rule-Prioritized DQN (IMPROVED)
# ----------------------------
class AfcAgent:
    """
    Hybrid agent combining rule-based policy with DQN learning.

    Features:
    - Rule-based policy for most actions (exploration, collection)
    - DQN occasionally takes control (epsilon_rl probability)
    - Experience replay with shaped rewards
    - Target network for stable learning
    - Metrics tracking for performance monitoring

    API:
        act(step, obs) -> actions array (max_units, 3)
        update(step, obs, reward, action, terminated, next_obs) -> performs learning
    """

    def __init__(
            self,
            player: str,
            name: str = "AfcAgent",
            dqn_lr: float = 1e-4,
            gamma: float = 0.99,
            epsilon_rl: float = 1.0,
            epsilon_dqn: float = 0.1,
            epsilon_decay: float = 0.996,  # NEW: Decay rate per training step
            epsilon_min: float = 0.01,  # NEW: Minimum epsilon
            buffer_capacity: int = 100000,
            batch_size: int = 256,
            target_update_freq: int = 1000,
            train_freq: int = 1,  # NEW: Train every N steps instead of every step
            save_path: Optional[str] = None,
            load_path: Optional[str] = None,
            save_every_n_games: int = 10,
    ):
        """
        Initialize the AfcAgent.

        Args:
            player: Player identifier ('player_0' or 'player_1')
            name: Agent name for identification
            dqn_lr: Learning rate for DQN optimizer
            gamma: Discount factor for future rewards
            epsilon_rl: Probability of using DQN instead of rules per unit
            epsilon_dqn: Epsilon-greedy exploration rate for DQN
            epsilon_decay: Decay rate for epsilon_dqn
            epsilon_min: Minimum value for epsilon_dqn
            buffer_capacity: Size of experience replay buffer
            batch_size: Batch size for training
            target_update_freq: Steps between target network updates
            train_freq: Train every N environment steps
            save_path: Path to save checkpoints
            load_path: Path to load pre-trained checkpoint
            save_every_n_games: Auto-save checkpoint every N games
        """
        # Identity and team assignment
        self.player = player
        self.name = name
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment config (set later via set_env_cfg)
        self.env_cfg = None
        self.max_units = None
        self.W = None
        self.H = None
        self.max_energy = None

        # DQN parameters
        self.state_dim = None  # Will be auto-detected on first observation
        self.n_actions = 5  # {stay, up, right, down, left}
        self.gamma = gamma
        self.epsilon_rl = epsilon_rl
        self.epsilon_dqn = epsilon_dqn
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq  # NEW: Control training frequency
        self.save_path = save_path or f"agents/afc_{player}_checkpoint.pth"

        # Initialize Q-networks (will be properly initialized after state_dim is known)
        self.qnet = None
        self.qnet_target = None
        self.opt = None
        self.criterion = nn.MSELoss()

        # Experience replay buffer
        self.replay = ReplayBuffer(capacity=buffer_capacity)

        # Training counters
        self.train_steps = 0
        self.total_steps = 0
        self.games_played = 0
        self.save_every_n_games = save_every_n_games
        self.steps_since_train = 0  # NEW: Track steps since last training

        # Rule-based agent memory
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()
        self.unit_explore_steps = dict()  # NEW: Track steps at current target
        self.current_match_game = 0  # 0, 1, or 2 (games within match)
        # NEW: Match tracking for 3-match tournament (3 matches × 3 games = 9 games)
        self.current_match = 0  # 0-2 (3 matches total)
        self.current_match_game = 0  # 0-2 (3 games per match)
        self.unit_relic_assignments = {}  # Maps unit_id -> assigned_position_idx
        # Add to __init__
        self.productive_relic_tiles = {}  # Maps (x, y) -> points_gained_count
        self.unit_last_position = {}  # Track unit positions between steps
        self.unit_position_points = {}  # Track points at each position


        # Performance metrics (reset per episode)
        self.episode_metrics = {
            "relics_discovered": 0,
            "total_explore_targets": 0,
            "low_energy_stops": 0,
            "dqn_actions": 0,
            "rule_actions": 0,
        }

        # Training metrics (accumulate with moving average)
        self.training_metrics = {
            "train_loss": 0.0,
            "train_calls": 0,
            "avg_reward": 0.0,
            "reward_calls": 0
        }

        # Energy threshold for low-energy behavior
        self.E_LOW = 15

        # Evaluation mode flag
        self.eval_mode = False

        # DQN learning rate for optimizer initialization
        self.dqn_lr = dqn_lr

        # Load checkpoint if provided
        if load_path is not None:
            self.load_checkpoint(load_path)

        self.metrics = SimpleMetrics(window_size=100)
        self.last_team_points = 0

        # Statistical tracking for productive tiles
        self.position_stats = {}  # Maps pos -> {'occupied': int, 'points': float, 'confidence': float}
        self.productive_relic_tiles = {}  # Maps pos -> visit_count (high confidence tiles only)

    # -------------------------
    # Network Initialization
    # -------------------------
    def _initialize_networks(self, state_dim: int):
        """
        Initialize Q-networks after state dimension is detected.

        Args:
            state_dim: Detected state dimension from feature generator
        """
        if self.qnet is not None:
            # Already initialized
            return

        self.state_dim = state_dim
        self.qnet = QNet(self.state_dim, self.n_actions).to(self.device)
        self.qnet_target = QNet(self.state_dim, self.n_actions).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.opt = optim.Adam(self.qnet.parameters(), lr=self.dqn_lr)

        print(f"[{self.player}] Initialized Q-networks with state_dim={self.state_dim}")

    # -------------------------
    # Environment Configuration
    # -------------------------
    def set_env_cfg(self, env_cfg):
        """
        Configure agent with environment settings.

        Args:
            env_cfg: Dictionary containing environment configuration
        """
        self.env_cfg = env_cfg
        self.max_units = int(env_cfg["max_units"])
        self.W = int(env_cfg["map_width"])
        self.H = int(env_cfg["map_height"])
        self.max_energy = float(env_cfg.get("max_unit_energy", 100.0))

    # -------------------------
    # Evaluation Mode
    # -------------------------
    def set_to_eval_mode(self):
        """
        Switch agent to evaluation mode.
        - Disables exploration in DQN (epsilon_dqn = 0)
        - Keeps epsilon_rl unchanged so DQN can still be evaluated
        - Sets Q-networks to eval mode
        """
        self.eval_mode = True
        self.epsilon_dqn = 0.0  # No random exploration in DQN
        if self.qnet is not None:
            self.qnet.eval()
            self.qnet_target.eval()
        print(f"[{self.player}] Set to evaluation mode (epsilon_rl={self.epsilon_rl}, epsilon_dqn=0)")

    def set_to_train_mode(self):
        """Switch agent back to training mode."""
        self.eval_mode = False
        if self.qnet is not None:
            self.qnet.train()
        print(f"[{self.player}] Set to training mode")

    # -------------------------
    # DQN Action Selection
    # -------------------------
    def _dqn_select(self, state_vec: np.ndarray) -> int:
        """
        Select action using DQN with epsilon-greedy exploration.

        Args:
            state_vec: Full state vector

        Returns:
            Selected action index (0-4)
        """
        if self.qnet is None:
            # Networks not initialized yet, return random action
            return int(np.random.randint(0, self.n_actions))

        # Validate state vector
        if state_vec is None or len(state_vec) != self.state_dim:
            return int(np.random.randint(0, self.n_actions))

        # Exploration: random action
        if np.random.rand() < self.epsilon_dqn:
            return int(np.random.randint(0, self.n_actions))

        # Exploitation: best action from Q-network
        self.qnet.eval()
        with torch.no_grad():
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.qnet(s)
            a = int(torch.argmax(qv, dim=1).item())
        return a

    # -------------------------
    # Rule-Based Action Selection
    # -------------------------
    def _rule_action_for_unit(self, unit_id: int, step: int, obs) -> int:
        """
        Determine action using rule-based policy with smart relic distribution.
        Units spread around relics (5x5 grid) with persistent assignments.

        FIXES:
        - Correct match length (100 steps, not 1000)
        - Proper relic spawning understanding (1 relic per match in first k matches)
        - Persistent unit assignments that survive respawns
        """
        # Extract unit information with bounds checking
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])

        # Check if unit is available and within bounds
        if unit_id >= len(unit_mask) or not unit_mask[unit_id]:
            return 0

        if unit_id >= len(unit_positions) or unit_id >= len(unit_energys):
            return 0

        unit_pos = unit_positions[unit_id]

        # Safe energy extraction
        try:
            unit_energy = unit_energys[unit_id][0] if np.ndim(unit_energys[unit_id]) > 0 else float(
                unit_energys[unit_id])
        except (IndexError, TypeError):
            unit_energy = 0.0

        # Get ACTUAL active units (not max_units)
        active_unit_ids = np.where(unit_mask)[0]
        num_active_units = len(active_unit_ids)

        # Find unit's rank among active units (for role assignment)
        try:
            unit_rank = list(active_unit_ids).index(unit_id)
        except ValueError:
            return 0  # Unit not in active list

        # Determine collector/explorer roles based on game phase
        num_relics = len(self.relic_node_positions)
        match_steps = self.env_cfg.get("max_steps_in_match", 100)
        game_round = min((step // match_steps) + 1, 3)
        match_step = step % match_steps
        match_number = min((step // match_steps) + 1, 5)  # 1-5

        if match_number <= 3:  # Relic spawn period (first 3 matches)
            # We expect at least match_number relics by now
            # But each spawns at random time 0-50 in its match
            expected_relics = match_number

            if num_relics < expected_relics:
                # Missing relics! Need heavy exploration
                explorer_ratio = 0.9
            elif match_step < 60:
                # Still within spawn window + safety margin
                # Keep exploring in case more relics spawn
                explorer_ratio = 0.6
            else:
                # Past spawn window, focus on collection
                explorer_ratio = 0.3

        elif match_number == 4:
            # No new relics, but maintain some exploration
            explorer_ratio = 0.2
        else:  # match_number == 5
            # Final match: maximize collection
            explorer_ratio = 0.1

        # Assign roles based on ACTIVE units
        if num_relics == 0:
            num_collectors = 0
            num_explorers = num_active_units
        else:
            num_explorers = int(num_active_units * explorer_ratio)
            num_collectors = num_active_units - num_explorers

        # Use unit_rank (position among active units) for role assignment
        is_collector = unit_rank < num_collectors
        is_explorer = not is_collector

        # Low energy safety: stay in place
        if unit_energy < self.E_LOW:
            self.episode_metrics["low_energy_stops"] += 1
            return 0

        # ========================================================================
        # COLLECTOR behavior: Stay on productive tiles, explore 5x5 grid
        # ========================================================================
        if is_collector and num_relics > 0:
            # ⭐ FIX: Use .item() to convert numpy scalar to Python int ⭐
            current_pos = (int(unit_pos[0].item()), int(unit_pos[1].item()))

            # DEBUG: Print every 20 steps for unit 0
            if unit_id == 0 and step % 20 == 0:
                if self.productive_relic_tiles:
                    print(f"  First 5 productive tiles: {list(self.productive_relic_tiles.keys())[:5]}")

            # **PRIORITY 1: Stay if on a known productive tile**
            if current_pos in self.productive_relic_tiles:
                if unit_id == 0 and step % 20 == 0:
                    print(f"  → DECISION: STAYING on productive tile")
                return 0  # Stay action

            # **PRIORITY 2: Move to nearest known productive tile**
            if self.productive_relic_tiles:
                nearest_productive = min(
                    self.productive_relic_tiles.keys(),
                    key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1])
                )
                dist_to_productive = abs(nearest_productive[0] - current_pos[0]) + abs(
                    nearest_productive[1] - current_pos[1])

                if dist_to_productive > 0:
                    if unit_id == 0 and step % 20 == 0:
                        print(
                            f"  → DECISION: Moving to nearest productive tile {nearest_productive} (dist={dist_to_productive})")
                    # ⭐ Use current_pos (already converted) instead of tuple(unit_pos) ⭐
                    direction = int(self._direction_to(current_pos, nearest_productive))
                    return direction


            # **PRIORITY 3: Use persistent assignment to explore 5x5 grid**
            if unit_id not in self.unit_relic_assignments:
                positions_per_relic = 25
                total_positions = num_relics * positions_per_relic
                assigned_positions = set(self.unit_relic_assignments.values())

                assigned_position_idx = None
                for idx in range(total_positions):
                    if idx not in assigned_positions:
                        assigned_position_idx = idx
                        break

                if assigned_position_idx is None:
                    assigned_position_idx = unit_id % total_positions

                self.unit_relic_assignments[unit_id] = assigned_position_idx

            assigned_position_idx = self.unit_relic_assignments[unit_id]
            positions_per_relic = 25
            total_positions = num_relics * positions_per_relic

            if assigned_position_idx >= total_positions:
                assigned_position_idx = assigned_position_idx % total_positions
                self.unit_relic_assignments[unit_id] = assigned_position_idx

            assigned_relic_idx = assigned_position_idx // positions_per_relic
            position_in_grid = assigned_position_idx % positions_per_relic

            if assigned_relic_idx >= len(self.relic_node_positions):
                assigned_relic_idx = assigned_relic_idx % len(self.relic_node_positions)

            relic_pos = self.relic_node_positions[assigned_relic_idx]

            # Calculate 5x5 grid position
            grid_x = position_in_grid % 5
            grid_y = position_in_grid // 5
            offset_x = grid_x - 2
            offset_y = grid_y - 2

            target_pos = (
                max(0, min(self.W - 1, relic_pos[0] + offset_x)),
                max(0, min(self.H - 1, relic_pos[1] + offset_y))
            )

            manhattan = abs(unit_pos[0] - target_pos[0]) + abs(unit_pos[1] - target_pos[1])

            # If at assigned position, stay for a few steps to test it
            if manhattan == 0:
                # Stay for 2 steps to see if we get points
                if unit_id not in self.unit_explore_steps:
                    self.unit_explore_steps[unit_id] = 0

                self.unit_explore_steps[unit_id] += 1

                if self.unit_explore_steps[unit_id] < 3:
                    return 0  # Stay to test
                else:
                    # Tested this tile, move to adjacent tile in 5x5 grid
                    self.unit_explore_steps[unit_id] = 0
                    # Move to next position in sequence
                    self.unit_relic_assignments[unit_id] = (assigned_position_idx + 1) % total_positions
                    return random.choice([1, 2, 3, 4])
            else:
                # Move toward assigned position
                direction = int(self._direction_to(tuple(unit_pos), target_pos))
                return direction


        # ========================================================================
        # EXPLORER behavior: Systematic quadrant exploration
        # ========================================================================
        else:
            # FIXED: Proper spawn window detection
            # Relics can spawn in first 3 matches, timesteps 0-50 of each match
            in_early_matches = match_number <= 3
            in_spawn_window = in_early_matches and match_step < 60  # 60 for safety margin

            if in_spawn_window and unit_id == 0 and step % 100 == 0:
                print(f"  Explorer in spawn window: Match {match_number}/5, Step {match_step}/60")

            # Update exploration target less frequently
            if unit_id not in self.unit_explore_locations:
                # Explore systematically in quadrants to find new relics
                quadrant = unit_id % 4  # 4 quadrants
                if quadrant == 0:  # Top-left
                    rand_loc = (np.random.randint(0, self.W // 2), np.random.randint(0, self.H // 2))
                elif quadrant == 1:  # Top-right
                    rand_loc = (np.random.randint(self.W // 2, self.W), np.random.randint(0, self.H // 2))
                elif quadrant == 2:  # Bottom-left
                    rand_loc = (np.random.randint(0, self.W // 2), np.random.randint(self.H // 2, self.H))
                else:  # Bottom-right
                    rand_loc = (np.random.randint(self.W // 2, self.W), np.random.randint(self.H // 2, self.H))

                self.unit_explore_locations[unit_id] = rand_loc
                self.unit_explore_steps[unit_id] = 0
                self.episode_metrics["total_explore_targets"] += 1
            else:
                target = self.unit_explore_locations[unit_id]
                dist = abs(unit_pos[0] - target[0]) + abs(unit_pos[1] - target[1])

                # Update target if reached or stuck
                # Shorter timeout in spawn window for more coverage
                timeout = 30 if in_spawn_window else 80
                if dist <= 2 or self.unit_explore_steps[unit_id] > timeout:
                    # Pick new random location (favor unexplored quadrants in spawn window)
                    if in_spawn_window:
                        # More aggressive random exploration
                        rand_loc = (np.random.randint(0, self.W), np.random.randint(0, self.H))
                    else:
                        # Focus on current quadrant
                        quadrant = unit_id % 4
                        if quadrant == 0:
                            rand_loc = (np.random.randint(0, self.W // 2), np.random.randint(0, self.H // 2))
                        elif quadrant == 1:
                            rand_loc = (np.random.randint(self.W // 2, self.W), np.random.randint(0, self.H // 2))
                        elif quadrant == 2:
                            rand_loc = (np.random.randint(0, self.W // 2), np.random.randint(self.H // 2, self.H))
                        else:
                            rand_loc = (np.random.randint(self.W // 2, self.W),
                                        np.random.randint(self.H // 2, self.H))

                    self.unit_explore_locations[unit_id] = rand_loc
                    self.unit_explore_steps[unit_id] = 0
                    self.episode_metrics["total_explore_targets"] += 1
                else:
                    self.unit_explore_steps[unit_id] += 1

            direction = int(self._direction_to(tuple(unit_pos), self.unit_explore_locations[unit_id]))
            return direction

    # Helper method (add if not already present)
    def _direction_to(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """
        Calculate direction from from_pos to to_pos.

        Returns:
            Direction: 0=stay, 1=up, 2=right, 3=down, 4=left
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        # Choose direction based on larger delta
        if abs(dx) > abs(dy):
            if dx > 0:
                return 2  # Right
            elif dx < 0:
                return 4  # Left
        else:
            if dy > 0:
                return 3  # Down
            elif dy < 0:
                return 1  # Up

        return 0  # Stay if already at target

    # -------------------------
    # Main Action Selection (Public API)
    # -------------------------
    def act(self, step: int, obs):
        """
        Produce actions for all available units (main environment interface).

        Uses hybrid policy:
        - With probability (1 - epsilon_rl): use rule-based action
        - With probability epsilon_rl: use DQN action

        Args:
            step: Current game step
            obs: Environment observation dictionary

        Returns:
            Action array of shape (max_units, 3): [action_type_or_direction, dx, dy]
            # Hinweis: 0..4 = stay/up/right/down/left (Move), 5 = Sap (Δx/Δy relativ)
        """
        # FIXED: Update relic memory only once in act() to avoid race condition
        observed_relic_node_positions = np.array(obs["relic_nodes"])
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        # Track newly discovered relics (only here, not in update)
        for rid in visible_relic_node_ids:
            if rid not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(rid)
                if rid < len(observed_relic_node_positions):
                    self.relic_node_positions.append(tuple(observed_relic_node_positions[rid]))
                    self.episode_metrics["relics_discovered"] += 1

        # Get available units
        ally_mask, _, _ = fg.get_ally_arrays(self.team_id, obs)
        available_unit_ids = np.where(ally_mask)[0]

        # Initialize action array
        actions = np.zeros((self.max_units, 3), dtype=int)

        # ⭐ DEBUG: Track action selection and failures ⭐
        dqn_attempted = 0
        dqn_success = 0
        dqn_failed_no_state = 0
        rule_chosen_by_epsilon = 0

        # Select action for each available unit
        for uid in available_unit_ids:
            # Compute state for DQN
            state_vec = None
            try:
                state_vec = fg.state_for_unit(self, uid, step, obs)
                if self.qnet is None and state_vec is not None:
                    self._initialize_networks(len(state_vec))
            except Exception as e:
                # ⭐ More verbose error logging ⭐
                print(f"[{self.player}] ERROR getting state for unit {uid} at step {step}: {e}")
                import traceback
                traceback.print_exc()
                state_vec = None

            # Get rule-based action
            rule_act = self._rule_action_for_unit(uid, step, obs)

            # Decide whether to use rule or DQN
            use_dqn = False
            rand_val = np.random.rand()

            if not self.eval_mode and rand_val < self.epsilon_rl:
                # DQN should be used
                use_dqn = True
                dqn_attempted += 1

                if state_vec is not None:
                    action = self._dqn_select(state_vec)
                    dqn_success += 1
                else:
                    # ⭐ DQN failed because state_vec is None ⭐
                    action = rule_act
                    use_dqn = False
                    dqn_failed_no_state += 1
            else:
                # Rule chosen by epsilon
                action = rule_act
                rule_chosen_by_epsilon += 1

            # --- Bewegung standardmäßig setzen ---
            actions[uid, 0] = int(action)      # 0..4 = stay/up/right/down/left
            actions[uid, 1] = 0
            actions[uid, 2] = 0

            # Check if we should sap an opponent
            unit_positions = np.array(obs["units"]["position"][self.team_id])
            if uid < len(unit_positions):
                unit_pos = unit_positions[uid]
                sap_dx, sap_dy = self._get_sap_action(uid, unit_pos, obs)

                # Wenn wir sappen: Aktionstyp 5 setzen + Δx/Δy eintragen (keine gleichzeitige Bewegung)
                if sap_dx != 0 or sap_dy != 0:
                    actions[uid, 0] = 5          # WICHTIG: 5 = Sap
                    actions[uid, 1] = int(sap_dx)
                    actions[uid, 2] = int(sap_dy)
                    # Optionaler Sanity-Check:
                    # sap_range = int(self.env_cfg.get("unit_sap_range", self.env_cfg.get("sap_range", 3)))
                    # assert abs(actions[uid,1]) <= sap_range and abs(actions[uid,2]) <= sap_range

            # ⭐ UPDATE METRICS (ALWAYS, not just when sapping!) ⭐
            if not self.eval_mode:
                self.metrics.update(
                    action_type='dqn' if use_dqn else 'rule',
                    sapped=(actions[uid, 0] == 5)
                )

        # ⭐ Show progress every 100 steps with debugging info ⭐
        if not self.eval_mode and step > 0 and step % 100 == 0:
            print(f"[{self.player}] Step {step:3d} | "
                  f"R:{self.metrics.get_avg_reward():6.2f} | "
                  f"Q:{self.metrics.get_avg_q():6.1f} | "
                  f"L:{self.metrics.get_avg_loss():.4f} | "
                  f"ε_dqn:{self.epsilon_dqn:.4f} | "
                  f"ε_rl:{self.epsilon_rl:.2f}")
            print(f"  → DQN attempted: {dqn_attempted}, "
                  f"DQN success: {dqn_success}, "
                  f"DQN failed (no state): {dqn_failed_no_state}, "
                  f"Rule (by epsilon): {rule_chosen_by_epsilon}")

            # ⭐ Extra warning if many failures ⭐
            if dqn_failed_no_state > dqn_success:
                print(f"  ⚠️ WARNING: More DQN failures than successes! Check fg.state_for_unit()")

        return actions

    # -------------------------
    # Learning Update (Public API)
    # -------------------------
    def update(self, step: int, obs, reward, action, terminated: bool, next_obs):
        """
        Update agent after environment step using observed transitions.
        Uses statistical tracking to identify truly productive tiles.
        """
        # Skip learning in evaluation mode
        if self.eval_mode:
            return

        # Skip if networks not initialized yet
        if self.qnet is None:
            return

        # Normalize reward input
        team_reward = 0.0
        if isinstance(reward, dict):
            team_reward = float(reward.get(self.player, 0.0))
        else:
            team_reward = float(reward)

        # Update metrics
        self.metrics.update(reward=team_reward)

        # ======================================================================
        # ⭐ STATISTICAL PRODUCTIVE TILE DETECTION (units can move) ⭐
        # ======================================================================
        # In update(), replace point detection section:

        try:
            current_points = obs["team_points"][self.team_id]
            points_this_step = current_points - self.last_team_points

            unit_positions = np.array(obs["units"]["position"][self.team_id])
            ally_mask = np.array(obs["units_mask"][self.team_id])

            # Track positions NEAR RELIC NODES (within 5x5 range)
            near_relic_positions = []

            for uid in np.where(ally_mask)[0]:
                if uid < len(unit_positions):
                    pos = (int(unit_positions[uid][0].item()), int(unit_positions[uid][1].item()))

                    # Check if position is near any known relic node (within 5x5 = manhattan distance ≤ 4)
                    near_relic = False
                    if self.relic_node_positions:
                        for relic_pos in self.relic_node_positions:
                            manhattan_dist = abs(pos[0] - relic_pos[0]) + abs(pos[1] - relic_pos[1])
                            if manhattan_dist <= 3:  # Within 7x7 square
                                near_relic = True
                                break

                    if near_relic:
                        near_relic_positions.append(pos)

                        # Initialize stats
                        if pos not in self.position_stats:
                            self.position_stats[pos] = {
                                'occupied': 0,
                                'points': 0.0,
                                'confidence': 0.0
                            }

                        # Increment occupancy
                        self.position_stats[pos]['occupied'] += 1

            # ⭐ DEBUG: Track rewards ⭐
            if not self.eval_mode and step % 100 == 0 and len(self.replay) > 100:
                recent_rewards = [t[2] for t in list(self.replay.buffer)[-100:]]
                print(f"  Replay buffer: {len(self.replay)} samples")
                print(f"  Recent rewards: mean={np.mean(recent_rewards):.3f}, "
                      f"min={np.min(recent_rewards):.3f}, max={np.max(recent_rewards):.3f}")

            # If points gained, credit ONLY positions near relics
            if points_this_step > 0 and len(near_relic_positions) > 0:
                # print(f"\n[POINTS] +{points_this_step} at step {step}")
                # print(f"  Near-relic positions: {near_relic_positions}")

                # Distribute points among near-relic positions
                points_per_position = points_this_step / len(near_relic_positions)

                for pos in near_relic_positions:
                    self.position_stats[pos]['points'] += points_per_position

                    # Recalculate confidence
                    occupied = self.position_stats[pos]['occupied']
                    points = self.position_stats[pos]['points']
                    self.position_stats[pos]['confidence'] = points / occupied if occupied > 0 else 0.0

                    confidence = self.position_stats[pos]['confidence']

                    # Mark as productive if high confidence
                    if confidence > 0.5 and occupied >= 3:
                        if pos not in self.productive_relic_tiles:
                            print(f"  ✓ HIGH CONFIDENCE productive tile {pos} (confidence={confidence:.2f})")
                            self.productive_relic_tiles[pos] = 0
                        self.productive_relic_tiles[pos] += 1

            # Prune low-confidence tiles periodically
            if step % 100 == 0 and self.position_stats:
                tiles_to_remove = []
                for pos in list(self.productive_relic_tiles.keys()):
                    if pos in self.position_stats:
                        confidence = self.position_stats[pos]['confidence']
                        occupied = self.position_stats[pos]['occupied']

                        if confidence < 0.7 and occupied >= 10:
                            tiles_to_remove.append(pos)
                            print(f"  ✗ Removing low-confidence tile {pos} (confidence={confidence:.2f})")

                for pos in tiles_to_remove:
                    del self.productive_relic_tiles[pos]

            self.last_team_points = current_points

        except Exception as e:
            if step % 100 == 0:
                print(f"[{self.player}] Warning: {e}")

        # ======================================================================
        # Rest of update() - reward shaping and training
        # ======================================================================
        action_arr = np.array(action)

        # Get unit masks
        ally_mask_obs, _, _ = fg.get_ally_arrays(self.team_id, obs)
        ally_mask_next, _, _ = fg.get_ally_arrays(self.team_id, next_obs)

        # Get unit positions for reward shaping
        try:
            unit_positions = np.array(obs["units"]["position"][self.team_id])
            next_unit_positions = np.array(next_obs["units"]["position"][self.team_id])
        except:
            unit_positions = None
            next_unit_positions = None

        # Process each active unit
        for uid in np.where(ally_mask_obs)[0]:
            try:
                # Get state before action
                s = fg.state_for_unit(self, uid, step, obs)
                if s is None or len(s) != self.state_dim:
                    continue

                # Get action taken
                if uid >= len(action_arr):
                    continue
                a_dir = int(action_arr[uid][0])
                if a_dir < 0 or a_dir >= self.n_actions:
                    continue

                # Get next state after action
                ns = fg.state_for_unit(self, uid, step + 1, next_obs)
                if ns is None or len(ns) != self.state_dim:
                    continue

                # Check if unit is done
                unit_done = bool(terminated or (uid >= len(ally_mask_next) or not ally_mask_next[uid]))

                # ======================================================================
                # ⭐ IMPROVED REWARD SHAPING WITH PRODUCTIVE TILE BONUSES ⭐
                # ======================================================================
                r = team_reward * 3.0

                # Get current and next positions (convert to Python int)
                current_pos = None
                next_pos = None
                if unit_positions is not None and uid < len(unit_positions):
                    current_pos = (int(unit_positions[uid][0].item()), int(unit_positions[uid][1].item()))
                if next_unit_positions is not None and uid < len(next_unit_positions):
                    next_pos = (int(next_unit_positions[uid][0].item()), int(next_unit_positions[uid][1].item()))

                # 1. **HUGE BONUS for being on HIGH-CONFIDENCE productive tile**
                if current_pos and current_pos in self.productive_relic_tiles:
                    confidence = self.position_stats.get(current_pos, {}).get('confidence', 0)
                    r += 10.0 * confidence  # Scale reward by confidence

                    # Extra bonus if we stayed (didn't move)
                    if next_pos and current_pos == next_pos:
                        r += 5.0 * confidence

                # 2. **BONUS for moving TO a high-confidence productive tile**
                if next_pos and next_pos in self.productive_relic_tiles:
                    if current_pos != next_pos:  # Only if we moved there
                        confidence = self.position_stats.get(next_pos, {}).get('confidence', 0)
                        r += 8.0 * confidence

                # 3. **Distance improvement to nearest HIGH-CONFIDENCE relic**
                dist_before = float(s[7]) if len(s) > 7 else 0.0
                dist_after = float(ns[7]) if len(ns) > 7 else 0.0

                # If we know productive tiles, use distance to nearest productive tile
                if self.productive_relic_tiles and current_pos:
                    nearest_productive = min(
                        self.productive_relic_tiles.keys(),
                        key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1])
                    )
                    dist_to_productive = abs(nearest_productive[0] - current_pos[0]) + abs(
                        nearest_productive[1] - current_pos[1])

                    if next_pos:
                        next_dist_to_productive = abs(nearest_productive[0] - next_pos[0]) + abs(
                            nearest_productive[1] - next_pos[1])
                        productive_improvement = dist_to_productive - next_dist_to_productive

                        # Reward moving toward known productive tiles
                        if productive_improvement > 0:
                            r += 2.0 * productive_improvement / (1.0 + next_dist_to_productive)
                else:
                    # Fallback to nearest relic
                    if dist_before > 0:
                        dist_improvement = dist_before - dist_after
                        r += 1.0 * dist_improvement / (1.0 + dist_after)

                # 4. **Energy management**
                energy_before = float(s[2]) * self.max_energy if len(s) > 2 else 0.0
                energy_after = float(ns[2]) * self.max_energy if len(ns) > 2 else 0.0
                energy_change = energy_after - energy_before

                if energy_change > 0:
                    r += 0.3 * (energy_change / 10.0)

                if energy_after < self.E_LOW:
                    if not (current_pos and current_pos in self.productive_relic_tiles):
                        r -= 1.0

                # 5. **Movement bonus** (encourage exploration)
                pos_before = (int(s[0] * self.W), int(s[1] * self.H)) if len(s) > 1 else (0, 0)
                pos_after = (int(ns[0] * self.W), int(ns[1] * self.H)) if len(ns) > 1 else (0, 0)

                if pos_before != pos_after:
                    if not (current_pos and current_pos in self.productive_relic_tiles):
                        r += 0.1

                # 6. **Survival bonus**
                r += 0.05

                # 7. **PENALTY for leaving productive tile**
                if current_pos and current_pos in self.productive_relic_tiles:
                    if next_pos and next_pos != current_pos:
                        if next_pos not in self.productive_relic_tiles:
                            confidence = self.position_stats.get(current_pos, {}).get('confidence', 0)
                            r -= 3.0 * confidence

                # Add transition to replay buffer
                self.replay.add(s, a_dir, r, ns, 1.0 if unit_done else 0.0)

            except Exception as e:
                if step % 100 == 0:
                    print(f"[{self.player}] Error processing unit {uid} transition: {e}")
                continue

        self.total_steps += 1
        self.steps_since_train += 1

        # Train periodically
        if self.steps_since_train >= self.train_freq:
            if len(self.replay) >= max(self.batch_size, 1000):
                self._train_step()
            self.steps_since_train = 0

    # -------------------------
    # Training Step
    # -------------------------
    def _train_step(self):
        """
        Perform one training step: sample mini-batch and update Q-network.
        Uses target network for stable Q-value targets.
        """
        if len(self.replay) < self.batch_size or self.qnet is None:
            return

        try:
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            # Set network to training mode
            self.qnet.train()

            # Compute current Q-values for taken actions
            q_preds = self.qnet(states).gather(1, actions)

            # Compute target Q-values using target network
            with torch.no_grad():
                # Use online network to SELECT action
                next_actions = self.qnet(next_states).argmax(dim=1, keepdim=True)
                # Use target network to EVALUATE action
                max_next_q = self.qnet_target(next_states).gather(1, next_actions)
                q_targets = rewards + (1.0 - dones) * (self.gamma * max_next_q)

            # Compute loss and update network
            loss = self.criterion(q_preds, q_targets)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=5.0)
            self.opt.step()

            # Update metrics with exponential moving average
            alpha = 0.01
            self.training_metrics["train_loss"] = (
                    alpha * loss.item() + (1 - alpha) * self.training_metrics["train_loss"]
            )
            self.training_metrics["train_calls"] += 1

            # Apply epsilon decay
            if self.epsilon_dqn > self.epsilon_min:
                self.epsilon_dqn = max(self.epsilon_min, self.epsilon_dqn * self.epsilon_decay)

            # Periodically update target network
            self.train_steps += 1
            if self.train_steps % self.target_update_freq == 0:
                self.qnet_target.load_state_dict(self.qnet.state_dict())

            # ⭐ ADD THESE LINES ⭐
            self.metrics.update(
                loss=loss.item(),
                q_value=q_preds.mean().item()
            )

        except KeyboardInterrupt:
            print(f"\n[{self.player}] Training interrupted! Saving checkpoint...")
            self.save_checkpoint()
            raise
        except Exception as e:
            print(f"[{self.player}] Training error: {e}")

    # -------------------------
    # Episode End Handler
    # -------------------------
    def on_episode_end(self):
        """Called at the end of each MATCH."""
        self.current_match += 1

        # ⭐ DEBUG: Print productive tiles summary with confidence ⭐
        print(f"\n{'=' * 60}")
        print(f"[{self.player}] Match {self.current_match}/3 ENDED")
        print(f"  High-confidence productive tiles: {len(self.productive_relic_tiles)}")

        if self.productive_relic_tiles:
            print(f"  Productive tiles with stats:")
            for pos in sorted(self.productive_relic_tiles.keys()):
                stats = self.position_stats.get(pos, {})
                confidence = stats.get('confidence', 0)
                occupied = stats.get('occupied', 0)
                points = stats.get('points', 0)
                print(f"    {pos}: confidence={confidence:.2f} ({points:.1f} pts / {occupied} occupations)")

        # Show medium-confidence tiles that might become productive
        medium_conf_tiles = {
            pos: stats for pos, stats in self.position_stats.items()
            if 0.5 < stats.get('confidence', 0) < 0.8 and stats.get('occupied', 0) >= 5
        }
        if medium_conf_tiles:
            print(f"  Medium-confidence tiles (under observation):")
            for pos, stats in sorted(medium_conf_tiles.items(), key=lambda x: x[1]['confidence'], reverse=True)[:5]:
                print(f"    {pos}: confidence={stats['confidence']:.2f} ({stats['points']:.1f} / {stats['occupied']})")

        print(f"{'=' * 60}\n")

        # One-line match summary
        print(f"[{self.player}] Match {self.current_match}/3 done | "
              f"Relics:{len(self.relic_node_positions)} | "
              f"DQN:{self.metrics.dqn_actions} | "
              f"Rule:{self.metrics.rule_actions}")

        self.metrics.reset_episode()

        # Reset per-match metrics
        self.episode_metrics = {
            "relics_discovered": 0,
            "total_explore_targets": 0,
            "low_energy_stops": 0,
            "dqn_actions": 0,
            "rule_actions": 0,
        }

        # Reset per-match memory (exploration targets)
        self.unit_explore_locations = dict()
        self.unit_explore_steps = dict()

        # ⭐ DO NOT RESET position_stats, productive_relic_tiles, or last_team_points here!
        # They should persist across matches within the same game

        # Check if GAME is complete (3 matches done)
        if self.current_match >= 3:
            print(f"\n{'=' * 60}")
            print(f"[{self.player}] ★ GAME {self.games_played + 1} COMPLETE! ★")

            # Final game statistics
            print(f"[{self.player}] Final productive tiles discovered:")
            if self.productive_relic_tiles:
                for pos in sorted(self.productive_relic_tiles.keys()):
                    stats = self.position_stats.get(pos, {})
                    confidence = stats.get('confidence', 0)
                    occupied = stats.get('occupied', 0)
                    points = stats.get('points', 0)
                    print(f"    {pos}: confidence={confidence:.2f} ({points:.1f} pts / {occupied} occupations)")

            print(f"[{self.player}] Starting new game...")
            print(f"{'=' * 60}\n")

            self.current_match = 0
            self.games_played += 1

            # ⭐ ONLY RESET THESE BETWEEN GAMES (not between matches) ⭐
            self.relic_node_positions = []
            self.discovered_relic_nodes_ids = set()
            self.unit_relic_assignments = {}
            self.productive_relic_tiles = {}  # NEW GAME = NEW MAP = NEW TILES
            self.position_stats = {}  # Reset statistics for new game
            self.last_team_points = 0  # Reset point tracking

            # Auto-save checkpoint periodically
            if self.games_played % self.save_every_n_games == 0:
                self.save_checkpoint()

    def _get_sap_action(self, unit_id: int, unit_pos, obs) -> Tuple[int, int]:
        """
        Verbesserte Zielwahl fürs Sapping:
        - scor‘t Tiles im AoE-Fenster via Heatmap (direkt > Nachbarn)
        - antizipiert simple Bewegung der Gegner (1-Step-Projektion)
        - vermeidet Friendly-Fire / schlechte Trades
        - schießt nur, wenn Score-Schwelle erreicht
        """
        # --- Env-Parameter ---
        sap_range = int(self.env_cfg.get("unit_sap_range", 3))
        sap_cost = float(self.env_cfg.get("unit_sap_cost", 10.0))
        dropoff = float(self.env_cfg.get("unit_sap_dropoff_factor", 0.4))  # 0..1, Impact auf Nachbarfeldern
        min_margin = float(self.env_cfg.get("unit_sap_min_energy_margin", 1.5))  # ggf. >1.0 konservativer
        score_floor = float(self.env_cfg.get("unit_sap_min_score", 7.0))  # Fire only if >= score_floor
        dist_lambda = float(self.env_cfg.get("unit_sap_distance_penalty", 0.15))  # leichte Präferenz für nah

        W, H = self.W, self.H

        # --- Eigene Energie ---
        try:
            my_energies = np.array(obs["units"]["energy"][self.team_id])
            my_energy = my_energies[unit_id][0] if np.ndim(my_energies[unit_id]) > 0 else float(my_energies[unit_id])
        except Exception:
            my_energy = 0.0

        # Nur sappen, wenn genug Energie + Sicherheitsmarge
        if my_energy < sap_cost * min_margin:
            return (0, 0)

        # --- Gegnerdaten ---
        opp_mask = np.array(obs["units_mask"][self.opp_team_id])
        opp_positions = np.array(obs["units"]["position"][self.opp_team_id])
        opp_energies = np.array(obs["units"]["energy"][self.opp_team_id])

        # --- Friendly Positionen (für Friendly-Fire-Filter) ---
        ally_mask = np.array(obs["units_mask"][self.team_id])
        ally_positions = np.array(obs["units"]["position"][self.team_id])

        # --- einfache Bewegungsschätzung (1-Step Projektion) ---
        # Merke: wir haben evtl. last positions nicht persistent – fallback = aktuelle Position
        # Optional: Du kannst self._opp_last_pos dict pflegen; hier nur simple Heuristik = Status quo
        predicted_positions = []
        for oid in np.where(opp_mask)[0]:
            if oid >= len(opp_positions):
                continue
            opos = opp_positions[oid]
            # Heuristik: bleibe, oder gehe 1 Schritt Richtung Relic/uns (gewichtete Zufälligkeit vermeiden)
            # Für Stabilität hier: keine Bewegung, aber kleines Jitter-Gewicht
            predicted_positions.append((int(opos[0]), int(opos[1])))

        # Falls keine Gegner sichtbar → kein Sap
        if not predicted_positions:
            return (0, 0)

        ux, uy = int(unit_pos[0]), int(unit_pos[1])

        # --- Scoring über alle Ziel-Tiles im Sap-Fenster ---
        best_score = -1e9
        best_target = None

        # Precompute ally set (friendly fire vermeiden)
        ally_tiles = set()
        for aid in np.where(ally_mask)[0]:
            if aid < len(ally_positions):
                ap = ally_positions[aid]
                ally_tiles.add((int(ap[0]), int(ap[1])))

        # Gegner auf Tile zählen (Heatmap)
        from collections import Counter
        opp_counter = Counter(predicted_positions)

        # Für jeden möglichen Zielpunkt (dx,dy) im Chebyshev-Radius
        for dx in range(-sap_range, sap_range + 1):
            for dy in range(-sap_range, sap_range + 1):
                if dx == 0 and dy == 0:
                    continue
                tx, ty = ux + dx, uy + dy
                if tx < 0 or tx >= W or ty < 0 or ty >= H:
                    continue

                # AoE-Bewertung: direktes Zieltile zählt voll, Nachbaren mit Dropoff
                # (Chebyshev 1-Nachbarschaft als Splash)
                direct_score = opp_counter[(tx, ty)] * 10.0  # hoher Wert für Volltreffer
                splash_score = 0.0
                for sx in (tx - 1, tx, tx + 1):
                    for sy in (ty - 1, ty, ty + 1):
                        if sx == tx and sy == ty:
                            continue
                        if 0 <= sx < W and 0 <= sy < H:
                            splash_score += opp_counter[(sx, sy)] * (3.0 * dropoff)

                raw_enemy_score = direct_score + splash_score

                # Friendly-Fire-Malus: wenn Allies im Zentrum/Adjazent stehen, reduziere stark
                friendly_pen = 0.0
                if (tx, ty) in ally_tiles:
                    friendly_pen += 8.0  # zentrum besetzt
                for sx in (tx - 1, tx, tx + 1):
                    for sy in (ty - 1, ty, ty + 1):
                        if (sx, sy) in ally_tiles:
                            friendly_pen += 2.0  # nachbarn

                # Distanz-Malus (leichte Präferenz für nahe Ziele – stabiler)
                cheby_dist = max(abs(dx), abs(dy))
                dist_pen = dist_lambda * cheby_dist

                # Endscore
                score = raw_enemy_score - friendly_pen - dist_pen

                if score > best_score:
                    best_score = score
                    best_target = (dx, dy)

        # Fire/No-Fire Entscheidung: nur schießen, wenn es sich „lohnt“
        if best_target is not None and best_score >= score_floor:
            return best_target

        return (0, 0)

    # -------------------------
    # Checkpoint Management
    # -------------------------
    def save_checkpoint(self, path: Optional[str] = None):
        """
        Save complete agent state to checkpoint file.
        FIXED: Does not save replay buffer (too large).

        Args:
            path: Optional custom path (uses self.save_path if not provided)
        """
        if self.qnet is None:
            print(f"[{self.player}] Networks not initialized, skipping save")
            return

        path = path or self.save_path

        # Create directory if needed
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception as e:
            print(f"[{self.player}] Warning: Could not create directory: {e}")
            return

        try:
            checkpoint = {
                "qnet": self.qnet.state_dict(),
                "qnet_target": self.qnet_target.state_dict(),
                "optimizer": self.opt.state_dict(),
                # FIXED: Don't save replay buffer (too large, not needed for resume)
                "replay_size": len(self.replay),  # Just track size
                "train_steps": self.train_steps,
                "total_steps": self.total_steps,
                "games_played": self.games_played,
                "epsilon_dqn": self.epsilon_dqn,
                "epsilon_rl": self.epsilon_rl,
                "state_dim": self.state_dim,
                "training_metrics": self.training_metrics,
            }
            torch.save(checkpoint, path)
            print(f"[{self.player}] ✓ Checkpoint saved to {path}")
        except Exception as e:
            print(f"[{self.player}] ✗ Error saving checkpoint: {e}")

    def load_checkpoint(self, path: Optional[str] = None):
        """
        Load complete agent state from checkpoint file.
        FIXED: Compatible with old checkpoints that have replay buffer.

        Args:
            path: Optional custom path (uses self.save_path if not provided)
        """
        path = path or self.save_path
        if not os.path.exists(path):
            print(f"[{self.player}] No checkpoint found at {path}, starting fresh")
            return

        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Get state dimension and initialize networks if needed
            state_dim = checkpoint.get("state_dim", 17)  # Default to 17 for old checkpoints
            if self.qnet is None:
                self._initialize_networks(state_dim)

            # Restore network states
            self.qnet.load_state_dict(checkpoint["qnet"])
            self.qnet_target.load_state_dict(checkpoint["qnet_target"])
            self.opt.load_state_dict(checkpoint["optimizer"])

            # FIXED: Don't load replay buffer (not saved anymore)
            # Old checkpoints might have it, but we skip it
            if "replay" in checkpoint:
                print(f"[{self.player}] Old checkpoint format detected, skipping replay buffer")

            # Restore counters and parameters
            self.train_steps = checkpoint.get("train_steps", 0)
            self.total_steps = checkpoint.get("total_steps", 0)
            self.games_played = checkpoint.get("games_played", 0)
            self.epsilon_dqn = checkpoint.get("epsilon_dqn", self.epsilon_dqn)
            self.epsilon_rl = checkpoint.get("epsilon_rl", self.epsilon_rl)
            self.training_metrics = checkpoint.get("training_metrics", self.training_metrics)

            print(f"[{self.player}] ✓ Checkpoint loaded from {path}")
            print(f"[{self.player}] Resuming: {self.games_played} games, {self.total_steps} steps, "
                  f"epsilon={self.epsilon_dqn:.4f}")
        except Exception as e:
            print(f"[{self.player}] ✗ Error loading checkpoint: {e}")
            print(f"[{self.player}] Starting fresh...")

    # -------------------------
    # Metrics Display
    # -------------------------
    def print_metrics(self, step: int):
        """
        Print performance metrics periodically.

        Args:
            step: Current game step
        """
        if step % 100 == 0:
            print(
                f"[{self.player}] Step {step} | "
                f"Total {self.total_steps} | "
                f"DQN {self.episode_metrics['dqn_actions']} | "
                f"Rule {self.episode_metrics['rule_actions']} | "
                f"Relics {len(self.relic_node_positions)} | "
                f"Replay {len(self.replay)} | "
                f"ε_dqn {self.epsilon_dqn:.4f} | "
                f"Loss {self.training_metrics['train_loss']:.4f}"
            )

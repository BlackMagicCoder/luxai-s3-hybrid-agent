import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
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
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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
# AfcAgent with IMPROVED SCORING STRATEGY
# ----------------------------
class AfcAgent:
    """
    Hybrid agent combining rule-based policy with DQN learning.

    ⭐ OPTIMIZED FOR HIGH SCORES ⭐
    - Aggressive productive tile discovery
    - Systematic 5x5 grid testing
    - Simplified collector/explorer ratio
    - Better staying behavior on productive tiles
    """

    def __init__(
            self,
            player: str,
            name: str = "AfcAgent",
            dqn_lr: float = 1e-4,
            gamma: float = 0.99,
            epsilon_rl: float = 1.0,
            epsilon_dqn: float = 0.1,
            epsilon_decay: float = 0.996,
            epsilon_min: float = 0.01,
            buffer_capacity: int = 100000,
            batch_size: int = 256,
            target_update_freq: int = 1000,
            train_freq: int = 1,
            save_path: Optional[str] = None,
            load_path: Optional[str] = None,
            save_every_n_games: int = 10,
    ):
        """Initialize the AfcAgent."""
        # Identity and team assignment
        self._collector_stay_counters = {}
        self._test_counters = {}  # ⭐ NEW: Track tile testing

        # Score tracking
        self.last_match_score = 0
        self.match_scores = []
        self.game_history = []

        self.player = player
        self.name = name
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment config
        self.env_cfg = None
        self.max_units = None
        self.W = None
        self.H = None
        self.max_energy = None

        # DQN parameters
        self.state_dim = None
        self.n_actions = 5
        self.gamma = gamma
        self.epsilon_rl = epsilon_rl
        self.epsilon_dqn = epsilon_dqn
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.save_path = save_path or f"agents/afc_{player}_checkpoint.pth"

        # Initialize Q-networks
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
        self.steps_since_train = 0

        # Rule-based agent memory
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()
        self.unit_explore_steps = dict()
        self.current_match_game = 0
        self.current_match = 0
        self.unit_relic_assignments = {}
        self.productive_relic_tiles = {}
        self.unit_last_position = {}
        self.unit_position_points = {}

        # Performance metrics
        self.episode_metrics = {
            "relics_discovered": 0,
            "total_explore_targets": 0,
            "low_energy_stops": 0,
            "dqn_actions": 0,
            "rule_actions": 0,
        }

        self.training_metrics = {
            "train_loss": 0.0,
            "train_calls": 0,
            "avg_reward": 0.0,
            "reward_calls": 0
        }

        # Energy threshold
        self.E_LOW = 15
        self.unit_move_cost = 1
        self.unit_sap_cost = 40

        # Evaluation mode flag
        self.eval_mode = False

        # DQN learning rate
        self.dqn_lr = dqn_lr

        # Load checkpoint if provided
        if load_path is not None:
            self.load_checkpoint(load_path)

        self.metrics = SimpleMetrics(window_size=100)
        self.last_team_points = 0

        # Enhanced points tracking
        self.points_history = []
        self.points_last_10_steps = deque(maxlen=10)
        self.points_per_match = []

        # Statistical tracking
        self.position_stats = {}
        self.inferred_productive_tiles = set()

        self.opp_position_history = {}
        self.relic_discovery_log = []

        # Tile tracking
        self.asteroid_tiles = set()
        self.nebula_tiles = {}
        self.nebula_vision_reduction = 0.0
        self.nebula_energy_reduction = 0.0

    # -------------------------
    # Network Initialization
    # -------------------------
    def _initialize_networks(self, state_dim: int):
        """Initialize Q-networks after state dimension is detected."""
        if self.qnet is not None:
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
        """Configure agent with environment settings."""
        self.env_cfg = env_cfg
        self.max_units = int(env_cfg["max_units"])
        self.W = int(env_cfg["map_width"])
        self.H = int(env_cfg["map_height"])
        self.max_energy = float(env_cfg.get("max_unit_energy", 100.0))

        self.unit_move_cost = int(env_cfg.get("unit_move_cost", 1))
        self.unit_sap_cost = int(env_cfg.get("unit_sap_cost", 40))
        self.E_LOW = max(15, self.unit_move_cost * 3 + 5)

        self.nebula_vision_reduction = float(env_cfg.get("nebula_tile_vision_reduction", 0.0))
        self.nebula_energy_reduction = float(env_cfg.get("nebula_tile_energy_reduction", 0.0))

        print(f"[{self.player}] Env params: move_cost={self.unit_move_cost}, "
              f"sap_cost={self.unit_sap_cost}, E_LOW={self.E_LOW}")
        print(f"[{self.player}] Tile params: nebula_vision={self.nebula_vision_reduction}, "
              f"nebula_energy={self.nebula_energy_reduction}")

    # -------------------------
    # Tile Utility Methods
    # -------------------------
    def _update_tile_info(self, obs):
        """Update asteroid and nebula tile tracking from observation."""
        if "map_features" in obs and "asteroid" in obs["map_features"]:
            asteroid_map = np.array(obs["map_features"]["asteroid"])
            self.asteroid_tiles = set()
            for y in range(self.H):
                for x in range(self.W):
                    if asteroid_map[y, x] > 0:
                        self.asteroid_tiles.add((x, y))

        if "map_features" in obs and "nebula" in obs["map_features"]:
            nebula_map = np.array(obs["map_features"]["nebula"])
            self.nebula_tiles = {}
            for y in range(self.H):
                for x in range(self.W):
                    if nebula_map[y, x] > 0:
                        self.nebula_tiles[(x, y)] = {
                            'vision_reduction': self.nebula_vision_reduction,
                            'energy_reduction': self.nebula_energy_reduction
                        }

    def _is_tile_passable(self, pos: Tuple[int, int]) -> bool:
        """Check if a tile is passable (not an asteroid)."""
        if pos[0] < 0 or pos[0] >= self.W or pos[1] < 0 or pos[1] >= self.H:
            return False
        return pos not in self.asteroid_tiles

    def _get_tile_penalty(self, pos: Tuple[int, int]) -> float:
        """Get penalty score for moving to a tile."""
        if pos in self.asteroid_tiles:
            return 1000.0

        if pos in self.nebula_tiles:
            return self.nebula_tiles[pos]['energy_reduction'] * 2.0

        return 0.0

    def _is_unit_stuck_on_asteroid(self, unit_pos: Tuple[int, int]) -> bool:
        """Check if unit is stuck on an asteroid tile."""
        return tuple(unit_pos) in self.asteroid_tiles

    def _get_adjacent_passable_tiles(self, pos: Tuple[int, int]) -> list:
        """Get list of adjacent passable tiles."""
        x, y = pos
        adjacent = []

        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if self._is_tile_passable(new_pos):
                adjacent.append(new_pos)

        return adjacent

    # -------------------------
    # Evaluation Mode
    # -------------------------
    def set_to_eval_mode(self):
        """Switch agent to evaluation mode."""
        self.eval_mode = True
        self.epsilon_dqn = 0.0
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
        """Select action using DQN with epsilon-greedy exploration."""
        if self.qnet is None:
            return int(np.random.randint(0, self.n_actions))

        if state_vec is None or len(state_vec) != self.state_dim:
            return int(np.random.randint(0, self.n_actions))

        if np.random.rand() < self.epsilon_dqn:
            return int(np.random.randint(0, self.n_actions))

        self.qnet.eval()
        with torch.no_grad():
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.qnet(s)
            a = int(torch.argmax(qv, dim=1).item())
        return a

    # -------------------------
    # ⭐ UPDATED: Tile-Aware Direction Calculation
    # -------------------------
    def _direction_to(self, from_pos, to_pos):
        """
        Calculate direction from from_pos to to_pos.
        Avoids asteroids and nebula tiles.
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        candidates = []

        if abs(dx) > abs(dy):
            if dx > 0:
                candidates.append((2, (from_pos[0] + 1, from_pos[1])))
            elif dx < 0:
                candidates.append((4, (from_pos[0] - 1, from_pos[1])))

            if dy > 0:
                candidates.append((3, (from_pos[0], from_pos[1] + 1)))
            elif dy < 0:
                candidates.append((1, (from_pos[0], from_pos[1] - 1)))
        else:
            if dy > 0:
                candidates.append((3, (from_pos[0], from_pos[1] + 1)))
            elif dy < 0:
                candidates.append((1, (from_pos[0], from_pos[1] - 1)))

            if dx > 0:
                candidates.append((2, (from_pos[0] + 1, from_pos[1])))
            elif dx < 0:
                candidates.append((4, (from_pos[0] - 1, from_pos[1])))

        passable_dirs = []
        for direction, next_pos in candidates:
            if self._is_tile_passable(next_pos):
                penalty = self._get_tile_penalty(next_pos)
                passable_dirs.append((direction, penalty))

        if passable_dirs:
            best_dir = min(passable_dirs, key=lambda x: x[1])[0]
            return best_dir

        return 0

    # -------------------------
    # ⭐ NEW: Systematic Tile Testing
    # -------------------------
    def _get_untested_relic_tiles(self, relic_pos):
        """
        Get all 5x5 tiles around relic that haven't been visited enough.
        ⭐ KEY OPTIMIZATION: Test ALL tiles systematically! ⭐
        """
        untested = []
        rx, ry = relic_pos

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                gx, gy = rx + dx, ry + dy
                if 0 <= gx < self.W and 0 <= gy < self.H:
                    pos = (gx, gy)
                    if self._is_tile_passable(pos):
                        # ⭐ CHANGED: Lower threshold from 3 to 2 visits ⭐
                        visits = self.position_stats.get(pos, {}).get('occupied', 0)
                        if visits < 2:  # Test each tile at least 2 times
                            untested.append(pos)

        return untested

    # -------------------------
    # ⭐ IMPROVED: Rule-Based Action Selection
    # -------------------------
    def _rule_action_for_unit(self, unit_id: int, step: int, obs) -> int:
        """
        Determine action using OPTIMIZED rule-based policy.

        ⭐ KEY CHANGES:
        1. Systematic 5x5 grid testing (not random rotation)
        2. Longer staying on productive tiles
        3. Simplified explorer/collector ratio
        4. Increased range for productive tile seeking
        """
        # Extract unit information
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])

        if unit_id >= len(unit_mask) or not unit_mask[unit_id]:
            return 0

        if unit_id >= len(unit_positions) or unit_id >= len(unit_energys):
            return 0

        unit_pos = unit_positions[unit_id]

        try:
            unit_energy = unit_energys[unit_id][0] if np.ndim(unit_energys[unit_id]) > 0 else float(
                unit_energys[unit_id])
        except (IndexError, TypeError):
            unit_energy = 0.0

        # Check if stuck on asteroid
        current_pos = (int(unit_pos[0]), int(unit_pos[1]))
        if self._is_unit_stuck_on_asteroid(current_pos):
            adjacent_passable = self._get_adjacent_passable_tiles(current_pos)
            if adjacent_passable:
                nearest = min(adjacent_passable,
                              key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
                return self._direction_to(current_pos, nearest)
            else:
                return 0

        # Get active units
        active_unit_ids = np.where(unit_mask)[0]
        num_active_units = len(active_unit_ids)

        try:
            unit_rank = list(active_unit_ids).index(unit_id)
        except ValueError:
            return 0

        # ⭐ SIMPLIFIED COLLECTOR/EXPLORER RATIO ⭐
        num_relics = len(self.relic_node_positions)
        num_productive = len(self.productive_relic_tiles)
        match_steps = self.env_cfg.get("max_steps_in_match", 100)
        match_number = min((step // match_steps) + 1, 5)
        match_step = step % match_steps

        # Calculate recent point rate
        points_per_step = 0.0
        if len(self.points_last_10_steps) > 0:
            points_per_step = sum(self.points_last_10_steps) / len(self.points_last_10_steps)

        # ⭐ SIMPLIFIED AGGRESSIVE COLLECTION ⭐
        if match_number <= 2:  # First 2 matches - find relics
            if num_productive < 5:
                explorer_ratio = 0.6  # Heavy exploration until we have tiles
            else:
                explorer_ratio = 0.3  # Start collecting once we have enough tiles
        else:  # Matches 3-5 - maximize collection
            explorer_ratio = 0.15 if num_productive >= 5 else 0.4

        # Assign roles
        if num_relics == 0:
            num_collectors = 0
            num_explorers = num_active_units
        else:
            num_explorers = int(num_active_units * explorer_ratio)
            num_collectors = num_active_units - num_explorers

        is_collector = unit_rank < num_collectors
        is_explorer = not is_collector

        # Debug output
        if unit_rank == 0 and step % 50 == 0:
            print(f"  [Strategy] Step {step}: {num_collectors} collectors / {num_explorers} explorers "
                  f"(ratio={1 - explorer_ratio:.0%}, pts/step={points_per_step:.2f}, "
                  f"productive={num_productive})")

        # Low energy handling
        if unit_energy < self.E_LOW:
            self.episode_metrics["low_energy_stops"] += 1
            if current_pos in self.nebula_tiles:
                adjacent_passable = self._get_adjacent_passable_tiles(current_pos)
                non_nebula = [p for p in adjacent_passable if p not in self.nebula_tiles]
                if non_nebula:
                    nearest = min(non_nebula,
                                  key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
                    return self._direction_to(current_pos, nearest)
            return 0

        # ========================================================================
        # EXPLORER behavior: SYMMETRIC GRID SEARCH
        # ========================================================================
        if is_explorer:
            in_early_matches = match_number <= 3
            in_spawn_window = in_early_matches and match_step < 60

            if in_spawn_window:
                if not hasattr(self, '_search_waypoints'):
                    self._search_waypoints = self._generate_symmetric_search_pattern(
                        self.W, self.H, spacing=3
                    )
                    self._waypoint_assignments = {}
                    print(f"[{self.player}] Generated {len(self._search_waypoints)} symmetric waypoints")

                if unit_id not in self._waypoint_assignments:
                    assigned_indices = set(self._waypoint_assignments.values())
                    for idx in range(len(self._search_waypoints)):
                        if idx not in assigned_indices:
                            self._waypoint_assignments[unit_id] = idx
                            break
                    else:
                        self._waypoint_assignments[unit_id] = unit_id % len(self._search_waypoints)

                wp_idx = self._waypoint_assignments[unit_id]

                if wp_idx >= len(self._search_waypoints):
                    wp_idx = unit_id % len(self._search_waypoints)
                    self._waypoint_assignments[unit_id] = wp_idx

                target_pos = self._search_waypoints[wp_idx]
                dist = abs(unit_pos[0] - target_pos[0]) + abs(unit_pos[1] - target_pos[1])

                if dist <= 1:
                    next_idx = (wp_idx + 1) % len(self._search_waypoints)
                    self._waypoint_assignments[unit_id] = next_idx
                    target_pos = self._search_waypoints[next_idx]

                direction = int(self._direction_to(tuple(unit_pos), target_pos))
                return direction

            else:
                # Passive random exploration
                if unit_id not in self.unit_explore_locations:
                    quadrant = unit_id % 4
                    if quadrant == 0:
                        rand_loc = (np.random.randint(0, self.W // 2),
                                    np.random.randint(0, self.H // 2))
                    elif quadrant == 1:
                        rand_loc = (np.random.randint(self.W // 2, self.W),
                                    np.random.randint(0, self.H // 2))
                    elif quadrant == 2:
                        rand_loc = (np.random.randint(0, self.W // 2),
                                    np.random.randint(self.H // 2, self.H))
                    else:
                        rand_loc = (np.random.randint(self.W // 2, self.W),
                                    np.random.randint(self.H // 2, self.H))

                    self.unit_explore_locations[unit_id] = rand_loc
                    self.unit_explore_steps[unit_id] = 0
                    self.episode_metrics["total_explore_targets"] += 1
                else:
                    target = self.unit_explore_locations[unit_id]
                    dist = abs(unit_pos[0] - target[0]) + abs(unit_pos[1] - target[1])

                    if dist <= 2 or self.unit_explore_steps[unit_id] > 80:
                        quadrant = unit_id % 4
                        if quadrant == 0:
                            rand_loc = (np.random.randint(0, self.W // 2),
                                        np.random.randint(0, self.H // 2))
                        elif quadrant == 1:
                            rand_loc = (np.random.randint(self.W // 2, self.W),
                                        np.random.randint(0, self.H // 2))
                        elif quadrant == 2:
                            rand_loc = (np.random.randint(0, self.W // 2),
                                        np.random.randint(self.H // 2, self.H))
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

        # ========================================================================
        # ⭐ OPTIMIZED COLLECTOR BEHAVIOR ⭐
        # ========================================================================
        if is_collector and num_relics > 0:
            current_pos = (int(unit_pos[0].item()), int(unit_pos[1].item()))

            # **PRIORITY 1: STAY LONGER on known productive tiles**
            if current_pos in self.productive_relic_tiles:
                confidence = self.position_stats.get(current_pos, {}).get('confidence', 0)
                adjacent_to_enemy = self._is_adjacent_to_enemies(current_pos, obs)

                # ⭐ CHANGED: More aggressive staying (threshold lowered from 0.5 to 0.4) ⭐
                if confidence > 0.4:  # Lower threshold!
                    if current_pos not in self.nebula_tiles and not adjacent_to_enemy:
                        return 0  # Stay!
                    elif confidence > 0.7 and not adjacent_to_enemy:  # Very productive
                        return 0  # Worth the nebula cost
                    elif adjacent_to_enemy and confidence < 0.9:
                        # Move away from energy void
                        safe_adjacent = []
                        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                            adj_pos = (current_pos[0] + dx, current_pos[1] + dy)
                            if (self._is_tile_passable(adj_pos) and
                                    not self._is_adjacent_to_enemies(adj_pos, obs)):
                                safe_adjacent.append(adj_pos)

                        if safe_adjacent:
                            nearest_safe = min(safe_adjacent,
                                               key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
                            return self._direction_to(current_pos, nearest_safe)

            # **PRIORITY 2: Move to nearest known productive tile (INCREASED RANGE)**
            if self.productive_relic_tiles:
                safe_productive_tiles = []
                risky_productive_tiles = []

                for tile_pos in self.productive_relic_tiles.keys():
                    if self._is_adjacent_to_enemies(tile_pos, obs):
                        risky_productive_tiles.append(tile_pos)
                    else:
                        safe_productive_tiles.append(tile_pos)

                candidate_tiles = safe_productive_tiles if safe_productive_tiles else risky_productive_tiles

                if candidate_tiles:
                    nearest_productive = min(
                        candidate_tiles,
                        key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1])
                    )

                    dist_to_productive = abs(nearest_productive[0] - current_pos[0]) + abs(
                        nearest_productive[1] - current_pos[1])

                    # ⭐ CHANGED: Increased range from 15 to 20 ⭐
                    if dist_to_productive <= 20:  # Go further for productive tiles!
                        direction = int(self._direction_to(current_pos, nearest_productive))
                        return direction

            # ⭐ NEW PRIORITY 3: SYSTEMATIC TILE TESTING ⭐
            nearest_relic = min(
                self.relic_node_positions,
                key=lambda r: abs(r[0] - current_pos[0]) + abs(r[1] - current_pos[1])
            )

            untested = self._get_untested_relic_tiles(nearest_relic)

            if untested:
                # Assign this unit to test an untested tile
                if unit_id not in self.unit_relic_assignments or \
                        self.unit_relic_assignments[unit_id] not in untested:
                    # Pick the nearest untested tile
                    nearest_untested = min(untested,
                                           key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
                    self.unit_relic_assignments[unit_id] = nearest_untested

                target = self.unit_relic_assignments[unit_id]

                # Go to target and stay there to test
                dist = abs(current_pos[0] - target[0]) + abs(current_pos[1] - target[1])
                if dist <= 1:
                    # ⭐ Stay for 2 steps to test this tile ⭐
                    if unit_id not in self._test_counters:
                        self._test_counters[unit_id] = 0

                    self._test_counters[unit_id] += 1
                    if self._test_counters[unit_id] < 2:  # Stay 2 steps
                        return 0  # Stay to test
                    else:
                        self._test_counters[unit_id] = 0
                        # Move to next untested tile
                        self.unit_relic_assignments[unit_id] = None

                return self._direction_to(current_pos, target)

            # **FALLBACK: Use existing assignment logic if all tiles tested**
            rx, ry = nearest_relic

            if unit_id not in self.unit_relic_assignments:
                grid_positions = []
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        gx, gy = rx + dx, ry + dy
                        if 0 <= gx < self.W and 0 <= gy < self.H:
                            if self._is_tile_passable((gx, gy)):
                                grid_positions.append((gx, gy))

                if grid_positions:
                    assigned_idx = unit_id % len(grid_positions)
                    self.unit_relic_assignments[unit_id] = grid_positions[assigned_idx]

            if unit_id in self.unit_relic_assignments:
                assigned_pos = self.unit_relic_assignments[unit_id]
                dist_to_assigned = abs(current_pos[0] - assigned_pos[0]) + abs(current_pos[1] - assigned_pos[1])

                if dist_to_assigned <= 1:
                    if not hasattr(self, '_collector_stay_counters'):
                        self._collector_stay_counters = {}

                    if unit_id not in self._collector_stay_counters:
                        self._collector_stay_counters[unit_id] = 0

                    self._collector_stay_counters[unit_id] += 1

                    # ⭐ Stay longer (3 steps instead of 2-3) ⭐
                    stay_duration = 3

                    if self._collector_stay_counters[unit_id] < stay_duration:
                        return 0
                    else:
                        self._collector_stay_counters[unit_id] = 0

                        grid_positions = []
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                gx, gy = rx + dx, ry + dy
                                if 0 <= gx < self.W and 0 <= gy < self.H:
                                    if self._is_tile_passable((gx, gy)):
                                        grid_positions.append((gx, gy))

                        if grid_positions:
                            current_idx = grid_positions.index(assigned_pos) if assigned_pos in grid_positions else 0
                            next_idx = (current_idx + 1) % len(grid_positions)
                            self.unit_relic_assignments[unit_id] = grid_positions[next_idx]
                            assigned_pos = grid_positions[next_idx]

                direction = int(self._direction_to(current_pos, assigned_pos))
                return direction

        return 0

    # -------------------------
    # Main Action Selection (Public API)
    # -------------------------
    def act(self, step: int, obs):
        """Produce actions for all available units."""
        self._update_tile_info(obs)

        # Track points
        try:
            current_points = obs["team_points"][self.team_id]
            points_this_step = current_points - self.last_team_points
            self.points_last_10_steps.append(points_this_step)
            self.points_history.append(current_points)
        except:
            pass

        if step == 0 and not hasattr(self, '_params_logged'):
            print(f"\n{'=' * 60}")
            print(f"[{self.player}] ⭐ OPTIMIZED AGENT - PARAMETERS ⭐")
            print(f"{'=' * 60}")
            print(f"  epsilon_rl:  {self.epsilon_rl}")
            print(f"  epsilon_dqn: {self.epsilon_dqn}")
            print(f"  eval_mode:   {self.eval_mode}")
            print(f"  games_played: {self.games_played}")
            print(f"  train_steps:  {self.train_steps}")
            print(f"  replay size:  {len(self.replay)}")
            print(f"{'=' * 60}\n")
            self._params_logged = True

        # Cache arrays
        cached_ally_positions = np.array(obs["units"]["position"][self.team_id])
        cached_opp_mask = np.array(obs["units_mask"][self.opp_team_id])

        # Update relic memory
        observed_relic_node_positions = np.array(obs["relic_nodes"])
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        for rid in visible_relic_node_ids:
            if rid not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(rid)
                if rid < len(observed_relic_node_positions):
                    relic_pos = tuple(observed_relic_node_positions[rid])

                    match_steps = self.env_cfg.get("max_steps_in_match", 100)
                    current_match = (step // match_steps) + 1
                    match_step = step % match_steps

                    discovery_info = {
                        'relic_id': rid,
                        'position': relic_pos,
                        'global_step': step,
                        'match': current_match,
                        'match_step': match_step,
                        'inferred': False
                    }
                    self.relic_discovery_log.append(discovery_info)

                    inferred_new = self._add_inferred_relic(relic_pos)

                    if inferred_new:
                        symmetric_pos = self._infer_symmetric_relic(relic_pos)
                        inferred_info = {
                            'relic_id': -1,
                            'position': symmetric_pos,
                            'global_step': step,
                            'match': current_match,
                            'match_step': match_step,
                            'inferred': True
                        }
                        self.relic_discovery_log.append(inferred_info)

                    print(f"  Total Relics Known: {len(self.relic_node_positions)}")
                    self.episode_metrics["relics_discovered"] += 1

        # Get available units
        ally_mask, _, _ = fg.get_ally_arrays(self.team_id, obs)
        available_unit_ids = np.where(ally_mask)[0]

        # Initialize action array
        actions = np.zeros((self.max_units, 3), dtype=int)

        any_enemies_visible = np.any(cached_opp_mask)

        # Select action for each available unit
        for uid in available_unit_ids:
            sap_dx, sap_dy = 0, 0
            should_sap = False

            if any_enemies_visible and uid < len(cached_ally_positions):
                unit_pos = cached_ally_positions[uid]
                sap_dx, sap_dy = self._get_sap_action(uid, unit_pos, obs)
                should_sap = (sap_dx != 0 or sap_dy != 0)

            if should_sap:
                actions[uid, 0] = 5
                actions[uid, 1] = int(sap_dx)
                actions[uid, 2] = int(sap_dy)
                if not self.eval_mode:
                    self.metrics.update(action_type='rule', sapped=True)
                continue

            use_dqn = False
            rand_val = np.random.rand()

            rule_act = self._rule_action_for_unit(uid, step, obs)

            if not self.eval_mode and rand_val < self.epsilon_rl:
                state_vec = None
                try:
                    state_vec = fg.state_for_unit(self, uid, step, obs)
                    if self.qnet is None and state_vec is not None:
                        self._initialize_networks(len(state_vec))
                except Exception as e:
                    state_vec = None

                if state_vec is not None:
                    action = self._dqn_select(state_vec)
                    use_dqn = True
                else:
                    action = rule_act
                    use_dqn = False
            else:
                action = rule_act

            actions[uid, 0] = int(action)
            actions[uid, 1] = 0
            actions[uid, 2] = 0

            if not self.eval_mode:
                self.metrics.update(action_type='dqn' if use_dqn else 'rule', sapped=False)

        # Show progress
        if not self.eval_mode and step > 0 and step % 100 == 0:
            points_per_step = 0.0
            if len(self.points_last_10_steps) > 0:
                points_per_step = sum(self.points_last_10_steps) / len(self.points_last_10_steps)

            try:
                current_points = obs["team_points"][self.team_id]
            except:
                current_points = 0

            print(f"[{self.player}] Step {step:3d} | "
                  f"💰 Points:{current_points:4d} (+{points_per_step:.2f}/step) | "
                  f"R:{self.metrics.get_avg_reward():6.2f} | "
                  f"Productive:{len(self.productive_relic_tiles)}")

        return actions

    # -------------------------
    # Learning Update (Public API)
    # -------------------------
    def update(self, step: int, obs, reward, action, terminated: bool, next_obs):
        """Update agent after environment step."""
        try:
            current_score = obs["team_points"][self.team_id]
            self.last_match_score = current_score
        except:
            pass

        if self.qnet is None:
            return

        if self.eval_mode:
            return

        team_reward = 0.0
        if isinstance(reward, dict):
            team_reward = float(reward.get(self.player, 0.0))
        else:
            team_reward = float(reward)

        self.metrics.update(reward=team_reward)

        # Update productive tile tracking
        try:
            current_points = obs["team_points"][self.team_id]
            points_this_step = current_points - self.last_team_points

            unit_positions = np.array(obs["units"]["position"][self.team_id])
            ally_mask = np.array(obs["units_mask"][self.team_id])

            near_relic_positions = []

            for uid in np.where(ally_mask)[0]:
                if uid < len(unit_positions):
                    pos = tuple(map(int, unit_positions[uid]))

                    near_relic = False
                    if self.relic_node_positions:
                        for relic_pos in self.relic_node_positions:
                            manhattan_dist = abs(pos[0] - relic_pos[0]) + abs(pos[1] - relic_pos[1])
                            if manhattan_dist <= 3:
                                near_relic = True
                                break

                    if near_relic:
                        near_relic_positions.append(pos)

                        if pos not in self.position_stats:
                            self.position_stats[pos] = {
                                'occupied': 0,
                                'points': 0.0,
                                'confidence': 0.0
                            }

                        self.position_stats[pos]['occupied'] += 1

            if points_this_step > 0 and len(near_relic_positions) > 0:
                points_per_position = points_this_step / len(near_relic_positions)

                for pos in near_relic_positions:
                    self.position_stats[pos]['points'] += points_per_position

                    occupied = self.position_stats[pos]['occupied']
                    points = self.position_stats[pos]['points']
                    self.position_stats[pos]['confidence'] = points / occupied if occupied > 0 else 0.0

                    confidence = self.position_stats[pos]['confidence']

                    # ⭐ CHANGED: Lower threshold from 0.5 to 0.3, fewer visits needed (2 instead of 3) ⭐
                    if confidence > 0.3 and occupied >= 2:
                        if pos not in self.productive_relic_tiles:
                            nebula_marker = " (NEBULA!)" if pos in self.nebula_tiles else ""
                            print(
                                f"  ✓ HIGH CONFIDENCE productive tile {pos}{nebula_marker} (confidence={confidence:.2f})")
                            self.productive_relic_tiles[pos] = 0

                            # Infer symmetric productive tile
                            symmetric_pos = self._infer_symmetric_productive_tile(pos)
                            if symmetric_pos is not None:
                                if symmetric_pos not in self.productive_relic_tiles:
                                    self.inferred_productive_tiles.add(symmetric_pos)
                                    sym_nebula = " (NEBULA!)" if symmetric_pos in self.nebula_tiles else ""
                                    print(f"  ⭐ INFERRED symmetric productive tile {symmetric_pos}{sym_nebula}")
                                    self.productive_relic_tiles[symmetric_pos] = 0
                        self.productive_relic_tiles[pos] += 1

            # Remove low-confidence tiles less aggressively
            if step % 100 == 0 and self.position_stats:
                tiles_to_remove = []
                for pos in list(self.productive_relic_tiles.keys()):
                    if pos in self.position_stats:
                        confidence = self.position_stats[pos]['confidence']
                        occupied = self.position_stats[pos]['occupied']

                        # ⭐ CHANGED: Only remove if confidence < 0.5 (was 0.7) ⭐
                        if confidence < 0.5 and occupied >= 10:
                            tiles_to_remove.append(pos)
                            print(f"  ✗ Removing low-confidence tile {pos} (confidence={confidence:.2f})")

                for pos in tiles_to_remove:
                    del self.productive_relic_tiles[pos]

            self.last_team_points = current_points

        except Exception as e:
            if step % 100 == 0:
                print(f"[{self.player}] Warning: {e}")

        action_arr = np.array(action)

        ally_mask_obs, _, _ = fg.get_ally_arrays(self.team_id, obs)
        ally_mask_next, _, _ = fg.get_ally_arrays(self.team_id, next_obs)

        try:
            unit_positions = np.array(obs["units"]["position"][self.team_id])
            next_unit_positions = np.array(next_obs["units"]["position"][self.team_id])
        except:
            unit_positions = None
            next_unit_positions = None

        for uid in np.where(ally_mask_obs)[0]:
            try:
                s = fg.state_for_unit(self, uid, step, obs)
                if s is None or len(s) != self.state_dim:
                    continue

                if uid >= len(action_arr):
                    continue
                a_dir = int(action_arr[uid][0])
                if a_dir < 0 or a_dir >= self.n_actions:
                    continue

                ns = fg.state_for_unit(self, uid, step + 1, next_obs)
                if ns is None or len(ns) != self.state_dim:
                    continue

                unit_done = bool(terminated or (uid >= len(ally_mask_next) or not ally_mask_next[uid]))

                # Improved reward shaping
                r = team_reward * 3.0

                current_pos = None
                next_pos = None
                if unit_positions is not None and uid < len(unit_positions):
                    current_pos = tuple(map(int, unit_positions[uid]))
                if next_unit_positions is not None and uid < len(next_unit_positions):
                    next_pos = (int(next_unit_positions[uid][0].item()), int(next_unit_positions[uid][1].item()))

                # Productive tile bonus
                if current_pos and current_pos in self.productive_relic_tiles:
                    confidence = self.position_stats.get(current_pos, {}).get('confidence', 0)
                    r += 10.0 * confidence

                    if next_pos and current_pos == next_pos:
                        r += 5.0 * confidence

                # Bonus for moving TO productive tile
                if next_pos and next_pos in self.productive_relic_tiles:
                    if current_pos != next_pos:
                        confidence = self.position_stats.get(next_pos, {}).get('confidence', 0)
                        r += 8.0 * confidence

                # Nebula penalties
                if current_pos and current_pos in self.nebula_tiles:
                    energy_reduction = self.nebula_tiles[current_pos]['energy_reduction']
                    r -= 2.0 * energy_reduction

                if next_pos and next_pos in self.nebula_tiles:
                    if current_pos != next_pos:
                        energy_reduction = self.nebula_tiles[next_pos]['energy_reduction']
                        r -= 3.0 * energy_reduction

                # Distance improvement
                dist_before = float(s[7]) if len(s) > 7 else 0.0
                dist_after = float(ns[7]) if len(ns) > 7 else 0.0

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

                        if productive_improvement > 0:
                            r += 2.0 * productive_improvement / (1.0 + next_dist_to_productive)
                else:
                    if dist_before > 0:
                        dist_improvement = dist_before - dist_after
                        r += 1.0 * dist_improvement / (1.0 + dist_after)

                # Energy management
                energy_before = float(s[2]) * self.max_energy if len(s) > 2 else 0.0
                energy_after = float(ns[2]) * self.max_energy if len(ns) > 2 else 0.0
                energy_change = energy_after - energy_before

                if energy_change > 0:
                    r += 0.3 * (energy_change / 10.0)

                if energy_after < self.E_LOW:
                    if not (current_pos and current_pos in self.productive_relic_tiles):
                        r -= 1.0

                # Movement bonus
                pos_before = (int(s[0] * self.W), int(s[1] * self.H)) if len(s) > 1 else (0, 0)
                pos_after = (int(ns[0] * self.W), int(ns[1] * self.H)) if len(ns) > 1 else (0, 0)

                if pos_before != pos_after:
                    if not (current_pos and current_pos in self.productive_relic_tiles):
                        r += 0.1

                # Survival bonus
                r += 0.05

                # Penalty for leaving productive tile
                if current_pos and current_pos in self.productive_relic_tiles:
                    if next_pos and next_pos != current_pos:
                        if next_pos not in self.productive_relic_tiles:
                            confidence = self.position_stats.get(current_pos, {}).get('confidence', 0)
                            r -= 3.0 * confidence

                self.replay.add(s, a_dir, r, ns, 1.0 if unit_done else 0.0)

            except Exception as e:
                if step % 100 == 0:
                    print(f"[{self.player}] Error processing unit {uid} transition: {e}")
                continue

        self.total_steps += 1
        self.steps_since_train += 1

        if self.steps_since_train >= self.train_freq:
            if len(self.replay) >= max(self.batch_size, 1000):
                self._train_step()
            self.steps_since_train = 0

    # -------------------------
    # Training Step
    # -------------------------
    def _train_step(self):
        """Perform one training step."""
        if len(self.replay) < self.batch_size or self.qnet is None:
            return

        try:
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            self.qnet.train()

            q_preds = self.qnet(states).gather(1, actions)

            with torch.no_grad():
                next_actions = self.qnet(next_states).argmax(dim=1, keepdim=True)
                max_next_q = self.qnet_target(next_states).gather(1, next_actions)
                q_targets = rewards + (1.0 - dones) * (self.gamma * max_next_q)

            loss = self.criterion(q_preds, q_targets)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=5.0)
            self.opt.step()

            alpha = 0.01
            self.training_metrics["train_loss"] = (
                    alpha * loss.item() + (1 - alpha) * self.training_metrics["train_loss"]
            )
            self.training_metrics["train_calls"] += 1

            if self.epsilon_dqn > self.epsilon_min:
                self.epsilon_dqn = max(self.epsilon_min, self.epsilon_dqn * self.epsilon_decay)

            self.train_steps += 1
            if self.train_steps % self.target_update_freq == 0:
                self.qnet_target.load_state_dict(self.qnet.state_dict())

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

        match_score = self.last_match_score if hasattr(self, 'last_match_score') else 0
        self.match_scores.append(match_score)

        match_steps = self.env_cfg.get("max_steps_in_match", 100)
        points_per_step_match = match_score / match_steps if match_steps > 0 else 0
        self.points_per_match.append(points_per_step_match)

        print(f"\n{'=' * 60}")
        print(f"[{self.player}] 🏁 MATCH {self.current_match}/3 COMPLETE")
        print(f"{'=' * 60}")
        print(f"  💰 Match Score: {match_score} points ({points_per_step_match:.2f} pts/step)")
        print(f"  Relics Known: {len(self.relic_node_positions)}")
        print(f"  Productive Tiles: {len(self.productive_relic_tiles)}")
        discovered_productive = len(self.productive_relic_tiles) - len(self.inferred_productive_tiles)
        inferred_productive = len(self.inferred_productive_tiles)
        if discovered_productive > 0:
            print(f"    - Discovered: {discovered_productive}")
            print(f"    - Inferred (symmetry): {inferred_productive}")
            if inferred_productive > 0:
                efficiency = (inferred_productive / discovered_productive) * 100
                print(f"    - Symmetry Efficiency: {efficiency:.0f}%")
        print(f"{'=' * 60}\n")

        self.metrics.reset_episode()
        self.episode_metrics = {
            "relics_discovered": 0,
            "total_explore_targets": 0,
            "low_energy_stops": 0,
            "dqn_actions": 0,
            "rule_actions": 0,
        }

        self.unit_explore_locations = dict()
        self.unit_explore_steps = dict()

        if hasattr(self, '_collector_stay_counters'):
            self._collector_stay_counters = {}
        if hasattr(self, '_test_counters'):
            self._test_counters = {}

        self.unit_relic_assignments = {}

        if self.current_match >= 3:
            total_game_score = sum(self.match_scores)

            print(f"\n{'🎮' * 30}")
            print(f"{'=' * 70}")
            print(f"[{self.player}] 🎮 GAME {self.games_played + 1} COMPLETE 🎮")
            print(f"{'=' * 70}")
            print(f"\n📊 MATCH SCORES:")
            for i, score in enumerate(self.match_scores, 1):
                print(f"  Match {i}/3: {score:4d} points")
            print(f"  {'─' * 40}")
            print(f"  TOTAL:      {total_game_score:4d} points")

            # ⭐ AVERAGE PER MATCH ⭐
            avg_per_match = total_game_score / 3
            print(f"  AVG/MATCH:  {avg_per_match:4.1f} points")
            print(f"\n{'=' * 70}")

            game_result = {
                'game_number': self.games_played + 1,
                'match_1': self.match_scores[0] if len(self.match_scores) > 0 else 0,
                'match_2': self.match_scores[1] if len(self.match_scores) > 1 else 0,
                'match_3': self.match_scores[2] if len(self.match_scores) > 2 else 0,
                'total': total_game_score,
                'avg_per_match': avg_per_match
            }
            self.game_history.append(game_result)

            if len(self.game_history) > 1:
                print(f"\n📈 GAME HISTORY:")
                print(f"{'Game':<8} {'Match 1':<10} {'Match 2':<10} {'Match 3':<10} {'Total':<10} {'Avg/Match':<10}")
                print(f"{'─' * 65}")
                for game in self.game_history:
                    print(f"{game['game_number']:<8} "
                          f"{game['match_1']:<10} "
                          f"{game['match_2']:<10} "
                          f"{game['match_3']:<10} "
                          f"{game['total']:<10} "
                          f"{game['avg_per_match']:<10.1f}")

                avg_score = sum(g['total'] for g in self.game_history) / len(self.game_history)
                avg_per_match_overall = sum(g['avg_per_match'] for g in self.game_history) / len(self.game_history)
                print(f"{'─' * 65}")
                print(f"Average: {avg_score:.1f} points per game ({avg_per_match_overall:.1f} per match)")

            print(f"{'=' * 70}")
            print(f"{'🎮' * 30}\n")

            self.relic_discovery_log = []
            self.current_match = 0
            self.match_scores = []
            self.games_played += 1

            if hasattr(self, '_search_waypoints'):
                del self._search_waypoints
                del self._waypoint_assignments

            self.relic_node_positions = []
            self.discovered_relic_nodes_ids = set()
            self.unit_relic_assignments = {}
            self.productive_relic_tiles = {}
            self.inferred_productive_tiles = set()
            self.position_stats = {}
            self.last_team_points = 0
            self.last_match_score = 0

            self.points_history = []
            self.points_last_10_steps.clear()
            self.points_per_match = []

            if self.games_played % self.save_every_n_games == 0:
                self.save_checkpoint()

    def _get_sap_action(self, unit_id: int, unit_pos, obs) -> Tuple[int, int]:
        """Improved sap targeting with AoE scoring."""
        opp_mask = np.array(obs["units_mask"][self.opp_team_id])
        if not np.any(opp_mask):
            return (0, 0)

        sap_range = int(self.env_cfg.get("unit_sap_range", 3))
        sap_cost = float(self.env_cfg.get("unit_sap_cost", 10.0))
        dropoff = float(self.env_cfg.get("unit_sap_dropoff_factor", 0.4))
        min_margin = float(self.env_cfg.get("unit_sap_min_energy_margin", 1.5))
        score_floor = float(self.env_cfg.get("unit_sap_min_score", 7.0))
        dist_lambda = float(self.env_cfg.get("unit_sap_distance_penalty", 0.15))

        W, H = self.W, self.H

        try:
            my_energies = np.array(obs["units"]["energy"][self.team_id])
            my_energy = my_energies[unit_id][0] if np.ndim(my_energies[unit_id]) > 0 else float(my_energies[unit_id])
        except Exception:
            my_energy = 0.0

        if my_energy < sap_cost * min_margin:
            return (0, 0)

        opp_positions = np.array(obs["units"]["position"][self.opp_team_id])

        ally_mask = np.array(obs["units_mask"][self.team_id])
        ally_positions = np.array(obs["units"]["position"][self.team_id])

        predicted_positions = []

        for oid in np.where(opp_mask)[0]:
            if oid >= len(opp_positions):
                continue

            curr_x = int(opp_positions[oid][0])
            curr_y = int(opp_positions[oid][1])
            curr_pos = (curr_x, curr_y)

            if oid in self.opp_position_history:
                last_pos = self.opp_position_history[oid]

                vx = curr_x - last_pos[0]
                vy = curr_y - last_pos[1]

                pred_x = curr_x + vx
                pred_y = curr_y + vy

                if 0 <= pred_x < W and 0 <= pred_y < H:
                    predicted_positions.append((pred_x, pred_y))
                else:
                    predicted_positions.append(curr_pos)
            else:
                predicted_positions.append(curr_pos)

            self.opp_position_history[oid] = curr_pos

        if not predicted_positions:
            return (0, 0)

        ux, uy = int(unit_pos[0]), int(unit_pos[1])

        best_score = -1e9
        best_target = None

        ally_tiles = set()
        for aid in np.where(ally_mask)[0]:
            if aid < len(ally_positions):
                ap = ally_positions[aid]
                ally_tiles.add((int(ap[0]), int(ap[1])))

        opp_counter = Counter(predicted_positions)

        for dx in range(-sap_range, sap_range + 1):
            for dy in range(-sap_range, sap_range + 1):
                if dx == 0 and dy == 0:
                    continue
                tx, ty = ux + dx, uy + dy
                if tx < 0 or tx >= W or ty < 0 or ty >= H:
                    continue

                direct_score = opp_counter[(tx, ty)] * 10.0
                splash_score = 0.0
                for sx in (tx - 1, tx, tx + 1):
                    for sy in (ty - 1, ty, ty + 1):
                        if sx == tx and sy == ty:
                            continue
                        if 0 <= sx < W and 0 <= sy < H:
                            splash_score += opp_counter[(sx, sy)] * (3.0 * dropoff)

                raw_enemy_score = direct_score + splash_score

                friendly_pen = 0.0
                if (tx, ty) in ally_tiles:
                    friendly_pen += 8.0
                for sx in (tx - 1, tx, tx + 1):
                    for sy in (ty - 1, ty, ty + 1):
                        if (sx, sy) in ally_tiles:
                            friendly_pen += 2.0

                cheby_dist = max(abs(dx), abs(dy))
                dist_pen = dist_lambda * cheby_dist

                score = raw_enemy_score - friendly_pen - dist_pen

                if score > best_score:
                    best_score = score
                    best_target = (dx, dy)

        if best_target is not None and best_score >= score_floor:
            return best_target

        return (0, 0)

    # -------------------------
    # Checkpoint Management
    # -------------------------
    def save_checkpoint(self, path: Optional[str] = None):
        """Save agent state to checkpoint file."""
        if self.qnet is None:
            print(f"[{self.player}] Networks not initialized, skipping save")
            return

        path = path or self.save_path

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
                "replay_size": len(self.replay),
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
        """Load agent state from checkpoint file."""
        path = path or self.save_path
        if not os.path.exists(path):
            print(f"[{self.player}] No checkpoint found at {path}, starting fresh")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            state_dim = checkpoint.get("state_dim", 17)
            if self.qnet is None:
                self._initialize_networks(state_dim)

            self.qnet.load_state_dict(checkpoint["qnet"])
            self.qnet_target.load_state_dict(checkpoint["qnet_target"])
            self.opt.load_state_dict(checkpoint["optimizer"])

            self.train_steps = checkpoint.get("train_steps", 0)
            self.total_steps = checkpoint.get("total_steps", 0)
            self.games_played = checkpoint.get("games_played", 0)
            self.training_metrics = checkpoint.get("training_metrics", self.training_metrics)

            print(f"[{self.player}] ✓ Checkpoint loaded from {path}")
            print(f"[{self.player}] Resuming: {self.games_played} games, {self.total_steps} steps")
        except Exception as e:
            print(f"[{self.player}] ✗ Error loading checkpoint: {e}")
            print(f"[{self.player}] Starting fresh...")

    # -------------------------
    # Helper Methods
    # -------------------------
    def _infer_symmetric_relic(self, relic_pos):
        """Given a discovered relic, infer its symmetric partner."""
        x, y = relic_pos
        W, H = self.W, self.H

        symmetric_x = W - 1 - y
        symmetric_y = H - 1 - x
        symmetric_pos = (symmetric_x, symmetric_y)

        if x + y == W - 1:
            return None

        if 0 <= symmetric_pos[0] < W and 0 <= symmetric_pos[1] < H:
            return symmetric_pos

        return None

    def _infer_symmetric_productive_tile(self, productive_pos):
        """Given a productive tile, infer its symmetric partner."""
        return self._infer_symmetric_relic(productive_pos)

    def _is_adjacent_to_enemies(self, pos: Tuple[int, int], obs) -> bool:
        """Check if position is cardinally adjacent to enemy units."""
        try:
            opp_mask = np.array(obs["units_mask"][self.opp_team_id])
            opp_positions = np.array(obs["units"]["position"][self.opp_team_id])

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                adj_x, adj_y = pos[0] + dx, pos[1] + dy

                for uid in np.where(opp_mask)[0]:
                    if uid < len(opp_positions):
                        enemy_pos = opp_positions[uid]
                        if int(enemy_pos[0]) == adj_x and int(enemy_pos[1]) == adj_y:
                            return True

            return False
        except:
            return False

    def _generate_symmetric_search_pattern(self, W, H, spacing=3):
        """Generate search pattern that exploits anti-diagonal symmetry."""
        waypoints = []

        corners = [
            (W - 1, 0),
            (W - 1, H - 1),
            (0, H - 1),
        ]

        for corner in corners:
            if 0 <= corner[0] < W and 0 <= corner[1] < H:
                waypoints.append(corner)

        edge_spacing = 1

        for y in range(edge_spacing, H, spacing):
            waypoints.append((W - 1, y))

        for x in range(edge_spacing, W, spacing):
            waypoints.append((x, H - 1))

        for y in range(0, H, spacing):
            if y == 0 or y == H - 1:
                continue

            min_x = max(0, W - 1 - y)

            if (y // spacing) % 2 == 0:
                for x in range(min_x, W - 1, spacing):
                    if x + y >= W - 1:
                        waypoints.append((x, y))
            else:
                for x in range(W - 2, min_x - 1, -spacing):
                    if x >= 0 and x + y >= W - 1:
                        waypoints.append((x, y))

        seen = set()
        unique_waypoints = []
        for wp in waypoints:
            if wp not in seen:
                seen.add(wp)
                unique_waypoints.append(wp)

        return unique_waypoints

    def _add_inferred_relic(self, relic_pos):
        """Add a relic and its symmetric partner to known relics."""
        if relic_pos not in self.relic_node_positions:
            self.relic_node_positions.append(relic_pos)
            print(f"  ✓ Added discovered relic: {relic_pos}")

        symmetric_pos = self._infer_symmetric_relic(relic_pos)

        if symmetric_pos is not None:
            if symmetric_pos not in self.relic_node_positions:
                self.relic_node_positions.append(symmetric_pos)
                print(f"  ⭐ INFERRED symmetric relic: {symmetric_pos}")
                return True
            else:
                print(f"  ℹ️ Symmetric relic {symmetric_pos} already known")
        else:
            print(f"  ℹ️ Relic {relic_pos} is on anti-diagonal")

        return False
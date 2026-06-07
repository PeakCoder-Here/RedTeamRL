"""
network_env.py
────────────────────────────────────────────────────────────────
Custom Gymnasium environment implementing the red-team MDP.

Markov Decision Process (MDP) spec:
───────────────────────────────────
  State  S  : binary/numerical matrix representing attacker knowledge
  Action A  : 4 action types × 9 nodes = 36 discrete actions
  Reward Rt : Rt = R_access + R_stealth − C_step
  Terminal  : ROOT access achieved on TARGET_NODE (AD server)

Observation Vector (shape: [NUM_NODES × 4] = 36 floats):
  [ discovered[N] | access_level[N] | alerted[N] | current_pos_onehot[N] ]

Action Encoding:
  action = action_type * NUM_NODES + target_node
  → SCAN(0..8), EXPLOIT(9..17), PRIV_ESC(18..26), LAT_MOVE(27..35)

Reward Constants (document spec):
  R_capture_target = +100   Pwned AD domain controller
  R_root_access    =  +50   Root on any non-target node
  R_user_access    =  +10   User-level shell gained
  R_new_discovery  =   +5   First recon of a host
  R_ids_alert      =  -25   IDS detection triggered
  C_step           =  -0.1  Per-step time penalty (forces efficient paths)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import networkx as nx

from env.network_topology import (
    build_network_graph,
    NODE_REGISTRY,
    NUM_NODES,
    TARGET_NODE,
    SUBNETS,
)

# ──────────────────────────────────────────────────────────────
# Enums / Constants
# ──────────────────────────────────────────────────────────────
class AccessLevel:
    NONE = 0
    USER = 1
    ROOT = 2

class ActionType:
    SCAN      = 0   # Enumerate / discover a host
    EXPLOIT   = 1   # Remote code execution → USER shell
    PRIV_ESC  = 2   # Local privilege escalation → ROOT
    LAT_MOVE  = 3   # Lateral movement to owned adjacent node

ACTION_NAMES   = {0: "SCAN", 1: "EXPLOIT", 2: "PRIV_ESC", 3: "LAT_MOVE"}
NUM_ACT_TYPES  = 4
ACTION_SPACE_SIZE = NUM_ACT_TYPES * NUM_NODES   # 36

# Reward constants (match document specification)
R_CAPTURE_TARGET = +100.0
R_ROOT_ACCESS    =  +50.0
R_USER_ACCESS    =  +10.0
R_NEW_DISCOVERY  =   +5.0
R_IDS_ALERT      =  -25.0
C_STEP           =   -0.1

# Safety cutoff
MAX_STEPS = 500


# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────
class RedTeamEnv(gym.Env):
    """
    Simulated enterprise network environment for RL-based red-team agents.

    The attacker starts with an initial USER foothold on node 0
    (the public DMZ web server — simulates a pre-exploited entry point).
    The objective is to reach ROOT access on node 8 (Active Directory).

    Attack phases the agent must learn:
      1. SCAN  → discover adjacent hosts in each subnet
      2. EXPLOIT  → gain USER shell (remote exploit via CVE)
      3. LAT_MOVE → physically move to compromised host
      4. PRIV_ESC → escalate to ROOT (local kernel / token abuse)
      5. Repeat pivot across Firewall-1 → Corporate → Firewall-2 → DC
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.graph: nx.DiGraph = build_network_graph()

        # ── Observation space ─────────────────────────────────
        # 4 arrays of length NUM_NODES, values in [0, 2]
        obs_dim = NUM_NODES * 4
        self.observation_space = spaces.Box(
            low=0.0, high=2.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ── Action space ──────────────────────────────────────
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # State arrays (reset each episode)
        self.discovered:    np.ndarray = None
        self.access_level:  np.ndarray = None
        self.alerted:       np.ndarray = None
        self.current_node:  int        = None
        self.step_count:    int        = 0
        self.total_reward:  float      = 0.0
        self.attack_path:   list       = []

    # ──────────────────────────────────────────────────────────
    # Gym Core Methods
    # ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Attacker starts at node 0 (public web server) with initial foothold
        self.current_node = 0
        self.discovered   = np.zeros(NUM_NODES, dtype=np.float32)
        self.access_level = np.zeros(NUM_NODES, dtype=np.float32)
        self.alerted      = np.zeros(NUM_NODES, dtype=np.float32)
        self.step_count   = 0
        self.total_reward = 0.0
        self.attack_path  = [0]

        # Grant initial USER foothold on entry node
        self.discovered[0]   = 1.0
        self.access_level[0] = AccessLevel.USER

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time-step of the MDP.

        Args:
            action: int in [0, ACTION_SPACE_SIZE)

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.step_count += 1

        # Always apply step cost (forces efficiency)
        reward = C_STEP

        # Decode flat action integer
        action_type  = int(action) // NUM_NODES
        target_node  = int(action) %  NUM_NODES

        terminated = False
        truncated  = self.step_count >= MAX_STEPS

        # Validate action preconditions
        valid, reason = self._validate_action(action_type, target_node)

        if not valid:
            # Small extra penalty for invalid / wasted actions
            reward -= 1.0
            info = {
                "action_result": f"INVALID — {reason}",
                "action_type":   ACTION_NAMES.get(action_type, "?"),
                "target_node":   NODE_REGISTRY[target_node].name,
            }
        else:
            reward_delta, terminated, action_info = self._execute_action(
                action_type, target_node
            )
            reward += reward_delta
            info    = action_info

        self.total_reward += reward
        info.update(self._get_info())

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_step(action_type, target_node, reward, info)

        return obs, float(reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────
    # Action Validation
    # ──────────────────────────────────────────────────────────

    def _validate_action(
        self, action_type: int, target_node: int
    ) -> Tuple[bool, str]:
        """
        Check preconditions for the given (action_type, target_node) pair.
        Returns (is_valid, reason_string).
        """
        if action_type == ActionType.SCAN:
            if self.discovered[target_node] == 1.0:
                return False, "already discovered"
            if not self._is_reachable(target_node):
                return False, "not reachable from current footprint"

        elif action_type == ActionType.EXPLOIT:
            if self.discovered[target_node] == 0.0:
                return False, "node not yet discovered"
            if self.access_level[target_node] >= AccessLevel.USER:
                return False, "access already gained — use PRIV_ESC instead"
            if not self._is_reachable(target_node):
                return False, "not reachable from current footprint"

        elif action_type == ActionType.PRIV_ESC:
            # Must physically be at the node to run local exploits
            if target_node != self.current_node:
                return False, "must be AT the node to escalate (use LAT_MOVE first)"
            if self.access_level[target_node] != AccessLevel.USER:
                return False, "need USER access first (or already ROOT)"

        elif action_type == ActionType.LAT_MOVE:
            if self.access_level[target_node] < AccessLevel.USER:
                return False, "need at least USER access on target node"
            if not self.graph.has_edge(self.current_node, target_node):
                return False, "no direct network path to target"
            if target_node == self.current_node:
                return False, "already at this node"

        else:
            return False, "unknown action type"

        return True, "OK"

    def _is_reachable(self, target_node: int) -> bool:
        """
        Target is reachable if it is a graph neighbour of ANY node
        the attacker currently controls (≥ USER access).
        """
        for owned in range(NUM_NODES):
            if self.access_level[owned] >= AccessLevel.USER:
                if self.graph.has_edge(owned, target_node):
                    return True
        return False

    # ──────────────────────────────────────────────────────────
    # Action Execution
    # ──────────────────────────────────────────────────────────

    def _execute_action(
        self, action_type: int, target_node: int
    ) -> Tuple[float, bool, Dict]:
        """
        Dispatch to the appropriate action handler.
        Returns (reward_delta, terminated, info_dict).
        """
        props = NODE_REGISTRY[target_node]

        if   action_type == ActionType.SCAN:
            reward, terminated = self._do_scan(target_node, props), False
        elif action_type == ActionType.EXPLOIT:
            reward, terminated = self._do_exploit(target_node, props)
        elif action_type == ActionType.PRIV_ESC:
            reward, terminated = self._do_priv_esc(target_node, props)
        elif action_type == ActionType.LAT_MOVE:
            reward, terminated = self._do_lateral_move(target_node), False

        return reward, terminated, {
            "action_type":  ACTION_NAMES[action_type],
            "target_node":  props.name,
            "access_level": int(self.access_level[target_node]),
            "ids_alert":    bool(self.alerted[target_node]),
        }

    # -- Individual action handlers --------------------------------

    def _do_scan(self, target_node: int, props) -> float:
        reward = 0.0
        self.discovered[target_node] = 1.0
        reward += R_NEW_DISCOVERY

        # Bonus for scanning into a new subnet for the first time
        current_subnet = NODE_REGISTRY[self.current_node].subnet
        target_subnet  = NODE_REGISTRY[target_node].subnet
        if target_subnet != current_subnet:
            reward += 15.0

        # Tiny IDS chance even for passive scans on monitored segments
        if props.ids_monitored and self.np_random.random() < 0.05:
            self.alerted[target_node] = 1.0
            reward += R_IDS_ALERT

        return reward

    def _do_exploit(self, target_node: int, props) -> Tuple[float, bool]:
        """
        Attempt remote code execution via the node's highest-probability CVE.
        Grants USER access on success. Triggers IDS stochastically.
        """
        reward = 0.0

        # Select service with the best exploit probability
        best = max(props.services, key=lambda s: s["exploit_prob"])
        exploit_prob = best["exploit_prob"]
        detect_prob  = best["detect_prob"]

        # Prior IDS alerts raise defender awareness → harder to stay hidden
        if self.alerted[target_node]:
            detect_prob = min(1.0, detect_prob * 1.50)

        # IDS check (independent of exploit outcome)
        if self.np_random.random() < detect_prob:
            self.alerted[target_node] = 1.0
            reward += R_IDS_ALERT

        # Exploit success roll
        if self.np_random.random() < exploit_prob:
            self.access_level[target_node] = AccessLevel.USER
            reward += R_USER_ACCESS

        return reward, False   # Win only achieved via PRIV_ESC on target

    def _do_priv_esc(self, target_node: int, props) -> Tuple[float, bool]:
        """
        Escalate USER → ROOT via local exploit (kernel, token, sudo abuse).
        Win condition fires when ROOT is achieved on TARGET_NODE.
        """
        reward     = 0.0
        terminated = False

        # Local priv-esc: moderate success, higher IDS exposure than scan
        priv_esc_prob = 0.70
        detect_prob   = 0.50 if props.ids_monitored else 0.20

        if self.alerted[target_node]:
            detect_prob = min(1.0, detect_prob * 1.50)

        # IDS check
        if self.np_random.random() < detect_prob:
            self.alerted[target_node] = 1.0
            reward += R_IDS_ALERT

        # Escalation success roll
        if self.np_random.random() < priv_esc_prob:
            self.access_level[target_node] = AccessLevel.ROOT

            if target_node == TARGET_NODE:
                # ── MISSION COMPLETE ──────────────────────────
                reward    += R_ROOT_ACCESS + R_CAPTURE_TARGET
                terminated = True
            else:
                reward += R_ROOT_ACCESS

        return reward, terminated

    def _do_lateral_move(self, target_node: int) -> float:
        """
        Pivot to an already-compromised adjacent host.
        Movement itself is silent (no IDS trigger).
        """
        self.current_node = target_node
        self.attack_path.append(target_node)
        return 0.0

    # ──────────────────────────────────────────────────────────
    # Observation & Info
    # ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build flat observation vector:
          [ discovered(N) | access_level(N) | alerted(N) | position_onehot(N) ]
        """
        pos_onehot = np.zeros(NUM_NODES, dtype=np.float32)
        pos_onehot[self.current_node] = 1.0

        return np.concatenate([
            self.discovered,    # [0, 1]^N
            self.access_level,  # [0, 1, 2]^N
            self.alerted,       # [0, 1]^N
            pos_onehot,         # one-hot position
        ]).astype(np.float32)

    def _get_info(self) -> Dict:
        return {
            "step":             self.step_count,
            "current_node":     NODE_REGISTRY[self.current_node].name,
            "nodes_discovered": int(self.discovered.sum()),
            "nodes_owned":      int((self.access_level > 0).sum()),
            "nodes_root":       int((self.access_level == 2).sum()),
            "alerts_triggered": int(self.alerted.sum()),
            "target_access":    int(self.access_level[TARGET_NODE]),
            "total_reward":     round(self.total_reward, 2),
        }

    # ──────────────────────────────────────────────────────────
    # Action Mask helper (for MaskablePPO upgrade path)
    # ──────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean mask of valid actions at the current state.
        Drop-in compatible with sb3-contrib MaskablePPO.
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for a in range(ACTION_SPACE_SIZE):
            atype  = a // NUM_NODES
            anode  = a %  NUM_NODES
            valid, _ = self._validate_action(atype, anode)
            mask[a]  = valid
        return mask

    # ──────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────

    def _render_step(self, action_type: int, target_node: int,
                     reward: float, info: Dict):
        print(
            f"  Step {self.step_count:>4} │ "
            f"{ACTION_NAMES.get(action_type,'?'):<9} → "
            f"{NODE_REGISTRY[target_node].name:<28} │ "
            f"r={reward:+6.1f} │ "
            f"pos={NODE_REGISTRY[self.current_node].name:<22} │ "
            f"owned={int((self.access_level > 0).sum())}/{NUM_NODES}"
        )

    def render(self):
        if self.render_mode == "ansi":
            return self._get_info()

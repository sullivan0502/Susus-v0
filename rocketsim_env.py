"""
High-performance C++ RocketSim + rlgym-sim environment builder for susu_bot (Susu-v0).
Features:
1. C++ RocketSim backend via rlgym_sim.make() with automatic collision_meshes initialization.
2. VariedRandomStateSetter: Randomizes ball and car states across:
   - Ground play (20%)
   - Roof Dribble / Flick / Catch-after-Flick / Ground Musty setups (20%)
   - High Aerial / Mid-Air Flip Reset / Air Keep-Up / Aerial Musty setups (25%)
   - Walls & Backboard play (20%)
   - Kickoffs (Speedflip & Slow Kickoff training) & RandomState (15%)
3. LookupAction: 90-action discrete Susu-style LookupAction space.
4. Multi-Horizon Ball Trajectory Perception (0.1s, 0.5s, 1.0s, 2.0s) in SusuSimObsBuilder (94 floats).
5. Role-Based Specialist & Mechanics Reward System (SusuMechanicsSimReward + CombinedReward):
   - High goal-scoring rewards + mechanic goal jackpots
   - Bot-specific specialty multipliers (flick_dribble, musty, air_reset_keepup, kickoff_master, all_rounder)
   - Dribble carry duration + Flick + Catch-after-Flick + Goal rewards
   - Ground Musty & higher Aerial Musty + Goal rewards
   - True Mid-Air Flip Reset (above ground z > 420 only; penalizes jumping onto ground ball for fake reset)
   - Continuous Air Keep-Up reward (ball in air without touching ground, wall, or ceiling) + Goal jackpot
   - Clean Speedflip & Slow Kickoff rewards + penalty for random/premature flicking on kickoff
"""

import math
import os
import random
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Force 1 thread per worker to prevent OpenMP/MKL thread explosion & IPC pipe.recv() deadlocks on 30-core servers
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import gym.spaces
import numpy as np

# Dynamic project root resolution (100% portable across Windows/Linux servers)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import RocketSim as rsim


def ensure_rocketsim_meshes_initialized() -> str:
    """
    Ensures RocketSim C++ knows where collision_meshes are located using dynamic paths.
    Called BEFORE importing rlgym_sim so vanilla rlgym-sim on any server finds collision_meshes.
    """
    project_root = Path(__file__).resolve().parent.parent
    local_meshes = project_root / "collision_meshes"
    cwd_meshes = Path.cwd() / "collision_meshes"
    site_meshes = Path(rsim.__file__).resolve().parent / "rlgym" / "rocket_league" / "sim" / "collision_meshes"

    # Ensure working directory has access to collision_meshes if vanilla rlgym-sim calls RocketSim.init("collision_meshes")
    if local_meshes.is_dir() and not cwd_meshes.is_dir():
        try:
            os.chdir(str(project_root))
        except Exception:
            pass

    for candidate in (local_meshes, cwd_meshes, site_meshes):
        if candidate.is_dir():
            try:
                rsim.init(str(candidate))
                return str(candidate)
            except Exception:
                # RocketSim.init may raise if already initialized in this process
                return str(candidate)
    return str(local_meshes)


# Initialize RocketSim collision meshes dynamically BEFORE importing rlgym_sim
ensure_rocketsim_meshes_initialized()

import rlgym_sim
from rlgym_sim.utils import ObsBuilder, RewardFunction, StateSetter
from rlgym_sim.utils.action_parsers import ActionParser
from rlgym_sim.utils.common_values import (
    BACK_WALL_Y,
    BALL_MAX_SPEED,
    BALL_RADIUS,
    BLUE_GOAL_BACK,
    BLUE_TEAM,
    CAR_MAX_SPEED,
    CEILING_Z,
    ORANGE_GOAL_BACK,
    ORANGE_TEAM,
    SIDE_WALL_X,
)
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.math import rand_vec3
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    AlignBallGoal,
    EventReward,
    FaceBallReward,
    LiuDistanceBallToGoalReward,
    LiuDistancePlayerToBallReward,
    SaveBoostReward,
    TouchBallReward,
    VelocityBallToGoalReward,
    VelocityPlayerToBallReward,
)
from rlgym_sim.utils.state_setters import DefaultState, RandomState, StateWrapper
from rlgym_sim.utils.terminal_conditions import TerminalCondition
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)

from src.susu_obs import (
    get_lookup_table,
    PREDICTION_HORIZONS,
    PREDICTION_HORIZONS_20,
    EXTRA_HORIZONS_12,
    predict_ball_multi_horizon,
)

KICKOFF_SPAWNS = (
    (-2048.0, -2560.0, math.pi * 0.25),
    (2048.0, -2560.0, math.pi * 0.75),
    (-256.0, -3840.0, math.pi * 0.50),
    (256.0, -3840.0, math.pi * 0.50),
    (0.0, -4608.0, math.pi * 0.50),
)


def _car_basis_from_pyr(pitch: float, yaw: float, roll: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    right = np.array([cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr], dtype=np.float32)
    up = np.array([-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr], dtype=np.float32)
    return forward, right, up


class ScenarioTerminal(TerminalCondition):
    """
    Scenario-aware short episode terminal condition (Claude section 3.3):
    - Ends immediately when a goal is scored.
    - For 'kickoff': ends 45 steps (3s) after the first touch or at max_steps=120 (8s).
    - For 'dribble_flick', 'musty_ground', 'musty_air', 'air_keepup', 'air_dribble', 'air_reset', 'air_takeoff':
      ends if the ball touches the ground after `ground_grace` steps, or at scenario `max_steps`.
    - For normal/ground/wall play: falls back to standard timeout.
    """

    SCENARIO_LIMITS = {
        "kickoff": (120, 10, False, 45),
        "dribble_flick": (300, 18, True, None),
        "bouncing_ball": (180, 45, True, 35),
        "musty_ground": (150, 14, True, None),
        "musty_air": (135, 12, True, None),
        "air_keepup": (165, 12, True, None),
        "air_dribble": (165, 12, True, None),
        "air_reset": (150, 10, True, None),
        "air_takeoff": (165, 20, True, None),
        "air_cross": (180, 15, True, None),
        "air_backboard": (180, 15, True, None),
    }

    def __init__(self, scenario_tracker: Dict[str, Any]):
        super().__init__()
        self.tracker = scenario_tracker
        self.t = 0
        self.first_touch = -1
        self.max_steps = 4500
        self.ground_grace = 15
        self.end_on_ball_ground = False
        self.end_after_touch = None

    def reset(self, initial_state: GameState):
        self.t = 0
        self.first_touch = -1
        scn = self.tracker.get("scenario", "normal")
        if scn in self.SCENARIO_LIMITS:
            self.max_steps, self.ground_grace, self.end_on_ball_ground, self.end_after_touch = self.SCENARIO_LIMITS[scn]
        else:
            self.max_steps = 4500
            self.ground_grace = 15
            self.end_on_ball_ground = False
            self.end_after_touch = None

    def is_terminal(self, current_state: GameState) -> bool:
        self.t += 1
        ball = current_state.ball
        if any(p.ball_touched for p in current_state.players):
            if self.first_touch < 0:
                self.first_touch = self.t

        if self.end_on_ball_ground and self.t > self.ground_grace and float(ball.position[2]) < (BALL_RADIUS + 18.0):
            return True

        if self.end_after_touch is not None and self.first_touch >= 0 and (self.t - self.first_touch) >= self.end_after_touch:
            return True

        if self.t >= self.max_steps:
            return True

        return False

KEY_BOOST_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 29, 30, 33],
    dtype=np.int32,
)


class LookupAction(ActionParser):
    """
    Self-contained 90-action discrete LookupAction parser for rlgym-sim.
    Embedded directly in the project so transferring to a new server requires
    zero manual patching of site-packages.
    """

    def __init__(self, bins=None):
        super().__init__()
        self._lookup_table = get_lookup_table()

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        indexes = np.asarray(actions, dtype=np.int32).reshape(-1)
        indexes = np.clip(indexes, 0, len(self._lookup_table) - 1)
        return self._lookup_table[indexes]


class VariedRandomStateSetter(StateSetter):
    """
    Scenario State Setter combining Claude's specialized mechanics drills with general match states:
    - Supports `SUSU_BOT_MECHANIC` env var ('mix', 'kickoff', 'dribble_flick', 'musty_ground', 'musty_air',
      'air_keepup', 'air_dribble', 'air_reset', 'air_takeoff', 'normal').
    - Communicates the active scenario to `SusuMechanicsSimReward` and `ScenarioTerminal` via `scenario_tracker`
      so the bot's specialist role and terminal condition always match the exact physical setup!
    """

    MIX_SCENARIOS = (
        ("bouncing_ball", 0.14),
        ("dribble_flick", 0.14),
        ("wall", 0.10),
        ("air_takeoff", 0.12),
        ("air_keepup", 0.10),
        ("air_reset", 0.10),
        ("musty_air", 0.08),
        ("musty_ground", 0.08),
        ("kickoff", 0.08),
        ("ground", 0.06),
    )

    SPECIALIZATION_SCENARIOS = {
        "aerial": (
            ("air_takeoff", 0.30),
            ("air_cross", 0.25),
            ("air_backboard", 0.20),
            ("air_dribble", 0.15),
            ("air_reset", 0.10),
        ),
        "flick": (
            ("dribble_flick", 0.50),
            ("musty_ground", 0.25),
            ("ground", 0.15),
            ("kickoff", 0.10),
        ),
        "movement": (
            ("wall", 0.40),
            ("ground", 0.30),
            ("kickoff", 0.20),
            ("bouncing_ball", 0.10),
        ),
        "shooting": (
            ("bouncing_ball", 0.40),
            ("air_cross", 0.30),
            ("wall", 0.20),
            ("kickoff", 0.10),
        ),
        "mix": MIX_SCENARIOS,
    }

    def __init__(
        self,
        ground_prob: float = 0.20,
        dribble_prob: float = 0.20,
        aerial_prob: float = 0.25,
        wall_prob: float = 0.20,
        random_state_prob: float = 0.05,
        scenario_tracker: Dict[str, Any] = None,
        forced_mechanic: str = None,
        specialization: str = None,
        infinite_boost: bool = False,
    ):
        super().__init__()
        self.ground_prob = ground_prob
        self.dribble_prob = dribble_prob
        self.aerial_prob = aerial_prob
        self.wall_prob = wall_prob
        self.random_state_prob = random_state_prob
        self.tracker = scenario_tracker if scenario_tracker is not None else {}
        self.specialization = (specialization or os.environ.get("SUSU_BOT_SPECIALIZATION", "mix")).strip().lower()
        self.infinite_boost = bool(infinite_boost or os.environ.get("SUSU_BOT_INFINITE_BOOST", "0").strip().lower() in ("1", "true", "yes"))
        self.forced_mechanic = forced_mechanic or os.environ.get("SUSU_BOT_MECHANIC", "mix").strip().lower()
        self._base_random = RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=False)
        self._kickoff = DefaultState()

    def _choose_scenario(self) -> str:
        mech = self.forced_mechanic
        if mech and mech not in ("mix", "all", "auto", ""):
            return mech
        pool = self.SPECIALIZATION_SCENARIOS.get(self.specialization, self.MIX_SCENARIOS)
        roll = random.random()
        cum = 0.0
        for name, prob in pool:
            cum += prob
            if roll < cum:
                return name
        return "kickoff"

    def reset(self, state_wrapper: StateWrapper):
        scn = self._choose_scenario()
        self.tracker["scenario"] = scn
        self.tracker["attacker_idx"] = 0

        if scn == "kickoff":
            self._set_kickoff_scenario(state_wrapper)
        elif scn == "dribble_flick":
            self._set_dribble_flick_scenario(state_wrapper)
        elif scn == "bouncing_ball":
            self._set_bouncing_ball_scenario(state_wrapper)
        elif scn == "musty_ground":
            self._set_musty_ground_scenario(state_wrapper)
        elif scn == "musty_air":
            self._set_musty_air_scenario(state_wrapper)
        elif scn in ("air_keepup", "air_dribble"):
            self._set_air_keepup_scenario(state_wrapper)
        elif scn == "air_reset":
            self._set_flip_reset_scenario(state_wrapper)
        elif scn == "air_takeoff":
            self._set_air_takeoff_scenario(state_wrapper)
        elif scn == "air_cross":
            self._set_air_cross_scenario(state_wrapper)
        elif scn == "air_backboard":
            self._set_air_backboard_scenario(state_wrapper)
        elif scn == "ground":
            self._set_ground_scenario(state_wrapper)
        elif scn == "wall":
            self._set_wall_scenario(state_wrapper)
        else:
            if random.random() < 0.35:
                self._kickoff.reset(state_wrapper)
            else:
                self._set_ground_scenario(state_wrapper)

        if self.infinite_boost:
            for car in state_wrapper.cars:
                car.boost = 1.0

    def _park_or_challenge_defender(
        self,
        state_wrapper: StateWrapper,
        attacker_idx: int,
        bx: float,
        by: float,
        bz: float,
        forward_dir: float,
        pressure_prob: float = 0.35,
    ):
        for idx, car in enumerate(state_wrapper.cars):
            if idx == attacker_idx:
                continue
            if random.random() < pressure_prob:
                # Active challenging defender between ball and goal
                ox = float(np.clip(bx + random.uniform(-800.0, 800.0), -2800.0, 2800.0))
                oy = float(np.clip(by + forward_dir * random.uniform(1400.0, 2400.0), -4600.0, 4600.0))
                oyaw = float(math.atan2(by - oy, bx - ox))
                car.set_pos(ox, oy, 17.0)
                car.set_rot(pitch=0.0, yaw=oyaw, roll=0.0)
                car.set_lin_vel(450.0 * math.cos(oyaw), 450.0 * math.sin(oyaw), 0.0)
                car.set_ang_vel(0.0, 0.0, 0.0)
                car.boost = random.uniform(0.35, 0.90)
            else:
                # Parked defender in corner so attacker can cleanly execute the mechanic
                px = random.choice([-3500.0, 3500.0])
                py = forward_dir * 4200.0
                car.set_pos(px, py, 17.0)
                car.set_rot(pitch=0.0, yaw=-forward_dir * (math.pi / 2.0), roll=0.0)
                car.set_lin_vel(0.0, 0.0, 0.0)
                car.set_ang_vel(0.0, 0.0, 0.0)
                car.boost = 0.33

    def _set_kickoff_scenario(self, state_wrapper: StateWrapper):
        """Standard 5 kickoff spawns with mirror symmetry and subtle 30% noise (Claude section 4.1)."""
        bx, by, byaw = random.choice(KICKOFF_SPAWNS)
        if random.random() < 0.30:
            bx += random.uniform(-20.0, 20.0)
            by += random.uniform(-20.0, 20.0)
            byaw += random.uniform(-0.03, 0.03)
        state_wrapper.ball.set_pos(0.0, 0.0, BALL_RADIUS)
        state_wrapper.ball.set_lin_vel(0.0, 0.0, 0.0)
        state_wrapper.ball.set_ang_vel(0.0, 0.0, 0.0)

        for idx, car in enumerate(state_wrapper.cars):
            if car.team_num == BLUE_TEAM:
                car.set_pos(bx, by, 17.0)
                car.set_rot(pitch=0.0, yaw=byaw, roll=0.0)
            else:
                car.set_pos(-bx, -by, 17.0)
                car.set_rot(pitch=0.0, yaw=byaw + math.pi, roll=0.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = 0.33

    def _set_dribble_flick_scenario(self, state_wrapper: StateWrapper):
        """
        Places the attacker on the ground with the ball balanced on its roof
        heading toward the opponent goal to train Dribble Carry, Flick, and Catch-after-Flick (Claude section 5.1).
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        cx = random.uniform(-2500.0, 2500.0)
        cy = random.uniform(-3200.0, 1400.0) * forward_dir
        cz = 17.0
        yaw = float(math.atan2(forward_dir * 4800.0 - cy, -cx * 0.6) + random.uniform(-0.35, 0.35))
        speed = random.uniform(600.0, 1450.0)
        vx = speed * math.cos(yaw)
        vy = speed * math.sin(yaw)

        attacker.set_pos(cx, cy, cz)
        attacker.set_rot(pitch=0.0, yaw=yaw, roll=0.0)
        attacker.set_lin_vel(vx, vy, 0.0)
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = random.uniform(0.35, 1.0)

        off = random.uniform(5.0, 38.0)
        lat_noise = random.uniform(-22.0, 22.0) if random.random() < 0.40 else 0.0
        bx = cx + math.cos(yaw) * off - math.sin(yaw) * lat_noise
        by = cy + math.sin(yaw) * off + math.cos(yaw) * lat_noise
        bz = random.uniform(130.0, 142.0)
        state_wrapper.ball.set_pos(bx, by, bz)
        state_wrapper.ball.set_lin_vel(
            vx + random.uniform(-45.0, 45.0),
            vy + random.uniform(-45.0, 45.0),
            random.uniform(0.0, 60.0),
        )
        state_wrapper.ball.set_ang_vel(*rand_vec3(1.2))
        self._park_or_challenge_defender(state_wrapper, attacker_idx, bx, by, bz, forward_dir, pressure_prob=0.40)

    def _set_bouncing_ball_scenario(self, state_wrapper: StateWrapper):
        """
        Bouncing Ball & Low Aerial Contest scenario:
        Spawns a ball bouncing at z=220..580 uu in front of the car, traveling toward the opponent's half.
        Forces the bot to learn aerial jumps, single-jump pops, and wall/ground aerials to intercept
        and shoot before the ball drops or an opponent challenges.
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        bx = random.uniform(-2200.0, 2200.0)
        by = random.uniform(-2200.0, 1800.0) * forward_dir
        bz = random.uniform(220.0, 580.0)
        bvx = random.uniform(-400.0, 400.0)
        bvy = forward_dir * random.uniform(300.0, 950.0)
        bvz = random.uniform(150.0, 520.0) if random.random() < 0.60 else random.uniform(-350.0, -80.0)

        state_wrapper.ball.set_pos(bx, by, bz)
        state_wrapper.ball.set_lin_vel(bvx, bvy, bvz)
        state_wrapper.ball.set_ang_vel(*rand_vec3(3.0))

        dist_back = random.uniform(420.0, 1100.0)
        cx = float(np.clip(bx + random.uniform(-350.0, 350.0), -3200.0, 3200.0))
        cy = float(np.clip(by - forward_dir * dist_back, -4400.0, 4400.0))
        yaw = float(math.atan2(by - cy, bx - cx) + random.uniform(-0.25, 0.25))
        speed = random.uniform(550.0, 1400.0)

        attacker.set_pos(cx, cy, 17.0)
        attacker.set_rot(pitch=0.0, yaw=yaw, roll=0.0)
        attacker.set_lin_vel(speed * math.cos(yaw), speed * math.sin(yaw), 0.0)
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = random.uniform(0.45, 1.0)

        self._park_or_challenge_defender(state_wrapper, attacker_idx, bx, by, bz, forward_dir, pressure_prob=0.35)

    def _set_musty_ground_scenario(self, state_wrapper: StateWrapper):
        """
        Ground Musty setup (Claude section 6.1 & 6.2):
        - 65%: Car on ground with ball slightly forward on the nose (off 55..85).
        - 35%: 'Petit saut' variant (car airborne z=65..135 with nose already tilted down pitch=-0.38..-0.24)
          so the policy immediately discovers the backflip whip!
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        cx = random.uniform(-2300.0, 2300.0)
        cy = random.uniform(-3000.0, 1200.0) * forward_dir
        yaw = float(math.atan2(forward_dir * 4800.0 - cy, -cx * 0.5) + random.uniform(-0.4, 0.4))
        speed = random.uniform(550.0, 1250.0)
        vx = speed * math.cos(yaw)
        vy = speed * math.sin(yaw)

        if random.random() < 0.35:
            cz = random.uniform(68.0, 132.0)
            pitch = random.uniform(-0.42, -0.24)
            vz = random.uniform(40.0, 140.0)
            bz = cz + random.uniform(55.0, 85.0)
        else:
            cz = 17.0
            pitch = 0.0
            vz = 0.0
            bz = random.uniform(132.0, 145.0)

        attacker.set_pos(cx, cy, cz)
        attacker.set_rot(pitch=pitch, yaw=yaw, roll=0.0)
        attacker.set_lin_vel(vx, vy, vz)
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = random.uniform(0.30, 0.90)

        off = random.uniform(55.0, 85.0)
        bx = cx + math.cos(yaw) * off
        by = cy + math.sin(yaw) * off
        state_wrapper.ball.set_pos(bx, by, bz)
        state_wrapper.ball.set_lin_vel(vx, vy, vz + random.uniform(0.0, 60.0))
        state_wrapper.ball.set_ang_vel(*rand_vec3(1.5))
        self._park_or_challenge_defender(state_wrapper, attacker_idx, bx, by, bz, forward_dir, pressure_prob=0.25)

    def _set_musty_air_scenario(self, state_wrapper: StateWrapper):
        """
        Aerial Musty setup (Claude section 6.3):
        High air-dribble (z=400..850), upward velocity (+250) to counteract gravity, ball right in front of the nose.
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        cx = random.uniform(-1900.0, 1900.0)
        cy = random.uniform(-2200.0, 1500.0) * forward_dir
        cz = random.uniform(400.0, 850.0)
        yaw = float(math.atan2(forward_dir * 5120.0 - cy, -cx * 0.5) + random.uniform(-0.35, 0.35))
        pitch = random.uniform(-0.38, -0.18) if random.random() < 0.45 else random.uniform(0.20, 0.65)
        fwd, _, up = _car_basis_from_pyr(pitch, yaw, 0.0)

        xy_speed = random.uniform(650.0, 1280.0)
        vel = np.array([math.cos(yaw) * xy_speed, math.sin(yaw) * xy_speed, random.uniform(200.0, 340.0)], dtype=np.float32)

        attacker.set_pos(cx, cy, cz)
        attacker.set_rot(pitch=pitch, yaw=yaw, roll=0.0)
        attacker.set_lin_vel(float(vel[0]), float(vel[1]), float(vel[2]))
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = random.uniform(0.55, 1.0)

        off = random.uniform(115.0, 165.0)
        bpos = np.array([cx, cy, cz], dtype=np.float32) + np.array([math.cos(yaw) * off, math.sin(yaw) * off, random.uniform(45.0, 95.0)], dtype=np.float32)
        state_wrapper.ball.set_pos(float(bpos[0]), float(bpos[1]), float(bpos[2]))
        state_wrapper.ball.set_lin_vel(float(vel[0]), float(vel[1]), float(vel[2] + random.uniform(20.0, 90.0)))
        state_wrapper.ball.set_ang_vel(*rand_vec3(2.0))
        self._park_or_challenge_defender(state_wrapper, attacker_idx, float(bpos[0]), float(bpos[1]), float(bpos[2]), forward_dir, pressure_prob=0.20)

    def _set_air_keepup_scenario(self, state_wrapper: StateWrapper):
        """
        Air Keep-Up & Continuous Air Dribble setup (Claude section 7.1):
        Car airborne (z=480..1080) right underneath/behind the ball with positive vertical velocity.
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        cx = random.uniform(-1900.0, 1900.0)
        cy = random.uniform(-2400.0, 1500.0) * forward_dir
        cz = random.uniform(480.0, 1050.0)
        yaw = float(math.atan2(forward_dir * 5120.0 - cy, -cx * 0.5) + random.uniform(-0.4, 0.4))
        pitch = random.uniform(0.35, 0.88)
        fwd, _, _ = _car_basis_from_pyr(pitch, yaw, 0.0)

        vel = fwd * random.uniform(400.0, 950.0) + np.array([0.0, 0.0, random.uniform(240.0, 360.0)], dtype=np.float32)
        attacker.set_pos(cx, cy, cz)
        attacker.set_rot(pitch=pitch, yaw=yaw, roll=0.0)
        attacker.set_lin_vel(float(vel[0]), float(vel[1]), float(vel[2]))
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = random.uniform(0.65, 1.0)

        bpos = np.array([cx, cy, cz], dtype=np.float32) + fwd * random.uniform(140.0, 260.0) + np.array([0.0, 0.0, random.uniform(70.0, 170.0)], dtype=np.float32)
        bpos[2] = float(np.clip(bpos[2], 520.0, 1620.0))
        bvel = vel + np.array([random.uniform(-90.0, 90.0), random.uniform(-90.0, 90.0), random.uniform(30.0, 180.0)], dtype=np.float32)
        state_wrapper.ball.set_pos(float(bpos[0]), float(bpos[1]), float(bpos[2]))
        state_wrapper.ball.set_lin_vel(float(bvel[0]), float(bvel[1]), float(bvel[2]))
        state_wrapper.ball.set_ang_vel(*rand_vec3(2.5))
        self._park_or_challenge_defender(state_wrapper, attacker_idx, float(bpos[0]), float(bpos[1]), float(bpos[2]), forward_dir, pressure_prob=0.25)

    def _set_flip_reset_scenario(self, state_wrapper: StateWrapper):
        """
        True Mid-Air Flip Reset setup (Claude section 7.2):
        Car inverted (roll ~ pi) at z=800..1350 with wheels oriented toward the ball (-up * d),
        and ball moving toward the underside of the car so it gets the 4-wheel reset and strikes toward net.
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        cx = random.uniform(-1700.0, 1700.0)
        cy = random.uniform(-1800.0, 2000.0) * forward_dir
        cz = random.uniform(780.0, 1320.0)
        yaw = float(math.atan2(forward_dir * 5120.0 - cy, -cx * 0.5) + random.uniform(-0.4, 0.4))
        pitch = random.uniform(0.45, 1.10)
        roll = math.pi + random.uniform(-0.22, 0.22)
        fwd, _, up = _car_basis_from_pyr(pitch, yaw, roll)

        vel = np.array([0.0, 0.0, random.uniform(160.0, 420.0)], dtype=np.float32) + fwd * random.uniform(250.0, 650.0)
        attacker.set_pos(cx, cy, cz)
        attacker.set_rot(pitch=pitch, yaw=yaw, roll=roll)
        attacker.set_lin_vel(float(vel[0]), float(vel[1]), float(vel[2]))
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = random.uniform(0.50, 1.0)

        # Since roll ~ pi, `up` points downward/away from the roof, so `-up` points in the direction of the 4 wheels!
        d = random.uniform(135.0, 215.0)
        bpos = np.array([cx, cy, cz], dtype=np.float32) - up * d
        bpos[2] = float(np.clip(bpos[2], max(480.0, cz + 40.0), 1650.0))
        bvel = vel + up * random.uniform(60.0, 220.0)
        state_wrapper.ball.set_pos(float(bpos[0]), float(bpos[1]), float(bpos[2]))
        state_wrapper.ball.set_lin_vel(float(bvel[0]), float(bvel[1]), float(bvel[2]))
        state_wrapper.ball.set_ang_vel(*rand_vec3(2.5))
        self._park_or_challenge_defender(state_wrapper, attacker_idx, float(bpos[0]), float(bpos[1]), float(bpos[2]), forward_dir, pressure_prob=0.20)

    def _set_air_takeoff_scenario(self, state_wrapper: StateWrapper):
        """
        Ground-to-Air Fast Aerial Takeoff setup (Claude section 7.4 Etape D):
        Car starts on the ground with boost facing a high ascending ball (z=750..1450, vz=+320..+680 uu/s)
        so the bot is forced to jump, pitch up, and boost into the sky to intercept it in the air!
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        bx = random.uniform(-1800.0, 1800.0)
        by = random.uniform(-600.0, 2400.0) * forward_dir
        bz = random.uniform(750.0, 1450.0)
        bvx = random.uniform(-300.0, 300.0)
        bvy = forward_dir * random.uniform(250.0, 800.0)
        bvz = random.uniform(320.0, 680.0)  # Ascendant en cloche, oblige a décoller immediatement !
        state_wrapper.ball.set_pos(bx, by, bz)
        state_wrapper.ball.set_lin_vel(bvx, bvy, bvz)
        state_wrapper.ball.set_ang_vel(*rand_vec3(3.0))

        dist_back = random.uniform(800.0, 1600.0)
        cx = float(np.clip(bx + random.uniform(-400.0, 400.0), -3200.0, 3200.0))
        cy = float(np.clip(by - forward_dir * dist_back, -4400.0, 4400.0))
        yaw = float(math.atan2(by - cy, bx - cx) + random.uniform(-0.15, 0.15))
        speed = random.uniform(700.0, 1400.0)

        attacker.set_pos(cx, cy, 17.0)
        attacker.set_rot(pitch=0.0, yaw=yaw, roll=0.0)
        attacker.set_lin_vel(speed * math.cos(yaw), speed * math.sin(yaw), 0.0)
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = 1.0
        self._park_or_challenge_defender(state_wrapper, attacker_idx, bx, by, bz, forward_dir, pressure_prob=0.30)

    def _set_air_cross_scenario(self, state_wrapper: StateWrapper):
        """
        High Aerial Cross Setup:
        High horizontal crossing pass from the left or right flank passing across the box
        at z=950..1600 uu, forcing the bot to double-jump fast aerial and intercept in mid-air!
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        is_left_to_right = random.random() < 0.50
        sign_x = -1.0 if is_left_to_right else 1.0

        bx = sign_x * random.uniform(2500.0, 3200.0)
        by = random.uniform(500.0, 2800.0) * forward_dir
        bz = random.uniform(950.0, 1550.0)

        bvx = -sign_x * random.uniform(850.0, 1500.0)
        bvy = forward_dir * random.uniform(-150.0, 350.0)
        bvz = random.uniform(120.0, 420.0)  # Flottant/ascendant, jamais au ras du sol

        state_wrapper.ball.set_pos(bx, by, bz)
        state_wrapper.ball.set_lin_vel(bvx, bvy, bvz)
        state_wrapper.ball.set_ang_vel(*rand_vec3(3.0))

        cx = random.uniform(-800.0, 800.0)
        cy = random.uniform(-1800.0, 400.0) * forward_dir
        cz = 17.0
        target_intercept = np.array([bx + bvx * 1.2, by + bvy * 1.2, bz], dtype=np.float32)
        yaw = float(math.atan2(target_intercept[1] - cy, target_intercept[0] - cx) + random.uniform(-0.15, 0.15))
        speed = random.uniform(700.0, 1350.0)

        attacker.set_pos(cx, cy, cz)
        attacker.set_rot(pitch=0.0, yaw=yaw, roll=0.0)
        attacker.set_lin_vel(speed * math.cos(yaw), speed * math.sin(yaw), 0.0)
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = 1.0
        self._park_or_challenge_defender(state_wrapper, attacker_idx, bx, by, bz, forward_dir, pressure_prob=0.30)

    def _set_air_backboard_scenario(self, state_wrapper: StateWrapper):
        """
        High Backboard Aerial Rebound Setup:
        Ball rebounding off the opponent backboard at high altitude (z=1100..1750 uu),
        coming back out into the field at vy=-600..-1100 uu/s. The bot must read the high bounce
        and aerial up to double-tap or redirect into net!
        """
        attacker_idx = random.randint(0, max(0, len(state_wrapper.cars) - 1))
        self.tracker["attacker_idx"] = attacker_idx
        attacker = state_wrapper.cars[attacker_idx]
        forward_dir = 1.0 if attacker.team_num == BLUE_TEAM else -1.0

        bx = random.uniform(-1400.0, 1400.0)
        by = forward_dir * random.uniform(4400.0, 4950.0)
        bz = random.uniform(1100.0, 1750.0)

        bvx = random.uniform(-350.0, 350.0)
        bvy = -forward_dir * random.uniform(600.0, 1100.0)
        bvz = random.uniform(140.0, 400.0)

        state_wrapper.ball.set_pos(bx, by, bz)
        state_wrapper.ball.set_lin_vel(bvx, bvy, bvz)
        state_wrapper.ball.set_ang_vel(*rand_vec3(3.5))

        cx = random.uniform(-900.0, 900.0)
        cy = forward_dir * random.uniform(0.0, 1600.0)
        cz = 17.0
        yaw = float(math.atan2(by - cy, bx - cx) + random.uniform(-0.12, 0.12))
        speed = random.uniform(850.0, 1400.0)

        attacker.set_pos(cx, cy, cz)
        attacker.set_rot(pitch=0.0, yaw=yaw, roll=0.0)
        attacker.set_lin_vel(speed * math.cos(yaw), speed * math.sin(yaw), 0.0)
        attacker.set_ang_vel(0.0, 0.0, 0.0)
        attacker.boost = 1.0
        self._park_or_challenge_defender(state_wrapper, attacker_idx, bx, by, bz, forward_dir, pressure_prob=0.25)

    def _set_ground_scenario(self, state_wrapper: StateWrapper):
        """Ball on or near the ground with random rolling/bouncing velocity."""
        bx = random.uniform(-3400.0, 3400.0)
        by = random.uniform(-4200.0, 4200.0)
        bz = random.uniform(94.0, 160.0)
        state_wrapper.ball.set_pos(bx, by, bz)

        bvx = random.uniform(-1400.0, 1400.0)
        bvy = random.uniform(-1400.0, 1400.0)
        bvz = random.uniform(-100.0, 350.0)
        state_wrapper.ball.set_lin_vel(bvx, bvy, bvz)
        state_wrapper.ball.set_ang_vel(*rand_vec3(4.0))

        placed = [np.array([bx, by, bz])]
        for car in state_wrapper.cars:
            cx, cy = self._sample_non_overlapping_xy(placed, max_x=3500.0, max_y=4500.0)
            cz = 17.0
            placed.append(np.array([cx, cy, cz]))

            angle_to_ball = np.arctan2(by - cy, bx - cx)
            yaw = float(angle_to_ball + random.uniform(-np.pi * 0.6, np.pi * 0.6))
            speed = random.uniform(0.0, 1800.0)

            car.set_pos(cx, cy, cz)
            car.set_rot(pitch=0.0, yaw=yaw, roll=0.0)
            car.set_lin_vel(speed * float(np.cos(yaw)), speed * float(np.sin(yaw)), 0.0)
            car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = random.uniform(0.15, 1.0)

    def _set_aerial_scenario(self, state_wrapper: StateWrapper):
        """Ball high in the air (450..1750 uu) with 3D trajectory; cars airborne or taking off."""
        bx = random.uniform(-3200.0, 3200.0)
        by = random.uniform(-4000.0, 4000.0)
        bz = random.uniform(450.0, 1750.0)
        state_wrapper.ball.set_pos(bx, by, bz)

        bvx = random.uniform(-1200.0, 1200.0)
        bvy = random.uniform(-1400.0, 1400.0)
        bvz = random.uniform(-300.0, 650.0)
        state_wrapper.ball.set_lin_vel(bvx, bvy, bvz)
        state_wrapper.ball.set_ang_vel(*rand_vec3(5.0))

        placed = [np.array([bx, by, bz])]
        for car in state_wrapper.cars:
            is_airborne = random.random() < 0.55
            cx, cy = self._sample_non_overlapping_xy(placed, max_x=3400.0, max_y=4400.0, center=(bx, by), radius=1600.0)
            cz = random.uniform(250.0, min(1500.0, bz + 200.0)) if is_airborne else 17.0
            placed.append(np.array([cx, cy, cz]))

            disp = np.array([bx - cx, by - cy, bz - cz], dtype=np.float32)
            dist = max(1.0, float(np.linalg.norm(disp)))
            dir_to_ball = disp / dist

            yaw = float(np.arctan2(dir_to_ball[1], dir_to_ball[0]) + random.uniform(-0.35, 0.35))
            pitch = float(np.arcsin(np.clip(dir_to_ball[2], -0.9, 0.9))) if is_airborne else 0.0
            roll = float(random.uniform(-0.5, 0.5)) if is_airborne else 0.0
            speed = random.uniform(600.0, 1700.0)

            car.set_pos(cx, cy, cz)
            car.set_rot(pitch=pitch, yaw=yaw, roll=roll)
            if is_airborne:
                car.set_lin_vel(*(dir_to_ball * speed + np.array([0.0, 0.0, random.uniform(100.0, 350.0)])))
                car.set_ang_vel(*rand_vec3(2.0))
            else:
                car.set_lin_vel(speed * float(np.cos(yaw)), speed * float(np.sin(yaw)), 0.0)
                car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = random.uniform(0.45, 1.0)

    def _set_wall_scenario(self, state_wrapper: StateWrapper):
        """Ball placed along the side walls or high on the backboard."""
        is_side_wall = random.random() < 0.70
        if is_side_wall:
            wall_sign = random.choice([-1.0, 1.0])
            bx = wall_sign * random.uniform(SIDE_WALL_X - 220.0, SIDE_WALL_X - 100.0)
            by = random.uniform(-3600.0, 3600.0)
            bz = random.uniform(180.0, 1400.0)
            bvx = -wall_sign * random.uniform(0.0, 300.0)
            bvy = random.uniform(-1400.0, 1400.0)
            bvz = random.uniform(-200.0, 600.0)
        else:
            wall_y_sign = random.choice([-1.0, 1.0])
            bx = random.uniform(-2600.0, 2600.0)
            by = wall_y_sign * random.uniform(BACK_WALL_Y - 260.0, BACK_WALL_Y - 110.0)
            bz = random.uniform(750.0, 1550.0)
            bvx = random.uniform(-800.0, 800.0)
            bvy = -wall_y_sign * random.uniform(100.0, 600.0)
            bvz = random.uniform(-250.0, 350.0)

        state_wrapper.ball.set_pos(bx, by, bz)
        state_wrapper.ball.set_lin_vel(bvx, bvy, bvz)
        state_wrapper.ball.set_ang_vel(*rand_vec3(4.5))

        placed = [np.array([bx, by, bz])]
        for idx, car in enumerate(state_wrapper.cars):
            pitch = 0.0
            roll = 0.0
            vz = 0.0
            if idx == 0 and is_side_wall:
                on_wall = random.random() < 0.50
                if on_wall:
                    wall_s = np.sign(bx) if abs(bx) > 10.0 else 1.0
                    cx = float(wall_s * (SIDE_WALL_X - 18.0))
                    cy = float(np.clip(by - random.uniform(300.0, 900.0), -4200.0, 4200.0))
                    cz = float(random.uniform(250.0, 900.0))
                    roll = float(-math.pi / 2.0 * wall_s)
                    yaw = float(math.pi / 2.0 if by >= cy else -math.pi / 2.0)
                    speed = random.uniform(650.0, 1500.0)
                    vx = 0.0
                    vy = speed if by >= cy else -speed
                    vz = random.uniform(0.0, 250.0)
                else:
                    cx = float(np.clip(bx - np.sign(bx) * random.uniform(150.0, 600.0), -3850.0, 3850.0))
                    cy = float(np.clip(by + random.uniform(-900.0, 900.0), -4400.0, 4400.0))
                    cz = 17.0
                    yaw = float(np.arctan2(by - cy, bx - cx) + random.uniform(-0.3, 0.3))
                    speed = random.uniform(600.0, 1600.0)
                    vx = speed * float(np.cos(yaw))
                    vy = speed * float(np.sin(yaw))
            else:
                cx, cy = self._sample_non_overlapping_xy(placed, max_x=3500.0, max_y=4400.0)
                cz = 17.0
                yaw = float(np.arctan2(by - cy, bx - cx) + random.uniform(-0.4, 0.4))
                speed = random.uniform(500.0, 1600.0)
                vx = speed * float(np.cos(yaw))
                vy = speed * float(np.sin(yaw))

            placed.append(np.array([cx, cy, cz]))
            car.set_pos(cx, cy, cz)
            car.set_rot(pitch=pitch, yaw=yaw, roll=roll)
            car.set_lin_vel(vx, vy, vz)
            car.set_ang_vel(0.0, 0.0, 0.0)
            car.boost = random.uniform(0.40, 1.0)

    def _sample_non_overlapping_xy(self, placed: List[np.ndarray], max_x: float, max_y: float, center=None, radius=None):
        for _ in range(25):
            if center is not None and radius is not None:
                cx = float(np.clip(center[0] + random.uniform(-radius, radius), -max_x, max_x))
                cy = float(np.clip(center[1] + random.uniform(-radius, radius), -max_y, max_y))
            else:
                cx = random.uniform(-max_x, max_x)
                cy = random.uniform(-max_y, max_y)
            pos = np.array([cx, cy, 17.0])
            if all(np.linalg.norm(pos[:2] - p[:2]) > 280.0 for p in placed):
                return cx, cy
        return random.uniform(-max_x, max_x), random.uniform(-max_y, max_y)

    def _sanitize_positions(self, state_wrapper: StateWrapper):
        """Clamps RandomState positions inside valid Soccar arena bounds and prevents overlaps."""
        bx = float(np.clip(state_wrapper.ball.position[0], -3700.0, 3700.0))
        by = float(np.clip(state_wrapper.ball.position[1], -4700.0, 4700.0))
        bz = float(np.clip(state_wrapper.ball.position[2], 94.0, 1850.0))
        state_wrapper.ball.set_pos(bx, by, bz)

        placed = [np.array([bx, by, bz])]
        for car in state_wrapper.cars:
            cx = float(np.clip(car.position[0], -3700.0, 3700.0))
            cy = float(np.clip(car.position[1], -4700.0, 4700.0))
            cz = float(np.clip(car.position[2], 17.0, 1850.0))
            pos = np.array([cx, cy, cz])
            if any(np.linalg.norm(pos - p) < 260.0 for p in placed):
                cx, cy = self._sample_non_overlapping_xy(placed, max_x=3500.0, max_y=4500.0)
                pos = np.array([cx, cy, cz])
            car.set_pos(cx, cy, cz)
            placed.append(pos)


class SusuMechanicsSimReward(RewardFunction):
    """
    Complete Scenario-Aware & Specialist Mechanics Reward System (Gemini + Claude fusion):
    1. Automatically syncs each car's specialist role ('air_dribble', 'flick_dribble', 'musty',
       'air_reset_keepup', 'kickoff_master', 'all_rounder') with the arena's active scenario
       (`scenario_tracker["scenario"]`) or `SUSU_BOT_FORCE_ROLE`.
    2. Kickoffs: Speedflip (+0.35), Slow Kickoff (+0.16), 50/50 win (+4.5), and anti-bad-jump penalty (-1.50).
    3. Dribble Carry + Anti-Farming + Flick + Post-Flick Catch Guidance + Catch-after-Flick (+9.5) + Jackpots.
    4. Ground Musty (+6.8..+9.5) & Aerial Musty (+14.0..+24.0) strictly verified with ball contact/acceleration.
    5. Aerial Mastery:
       - Ground-to-Air Fast Aerial Takeoff & Approach reward (Claude 7.4: climbing `vz > 260` toward predicted high ball).
       - Continuous Air Dribble Carry under the ball (Gemini section 3).
       - Air Keep-Up consecutive aerial touches without touching ground/walls/ceiling.
       - True Mid-Air Flip Reset (z > 420) + wheel alignment approach + Flip Reset Shot (+12.5) + anti-ground-reset penalty.
    6. Strict Cross-Penalties (Sol vs Air): penalizes staying glued to the ground during aerial drills
       or jumping aimlessly when the ball is low during ground drills.
    """

    SPECIALTIES = (
        "air_dribble",
        "flick_dribble",
        "musty",
        "air_reset_keepup",
        "kickoff_master",
        "all_rounder",
    )

    SCENARIO_TO_ROLE = {
        "kickoff": "kickoff_master",
        "dribble_flick": "flick_dribble",
        "bouncing_ball": "air_reset_keepup",
        "musty_ground": "musty",
        "musty_air": "musty",
        "air_keepup": "air_dribble",
        "air_dribble": "air_dribble",
        "air_reset": "air_reset_keepup",
        "air_takeoff": "air_reset_keepup",
        "air_cross": "air_dribble",
        "air_backboard": "air_reset_keepup",
        "wall": "air_reset_keepup",
    }

    def __init__(self, scenario_tracker: Dict[str, Any] = None, specialization: str = "mix"):
        super().__init__()
        self.tracker = scenario_tracker if scenario_tracker is not None else {}
        self.specialization = (specialization or os.environ.get("SUSU_BOT_SPECIALIZATION", "mix")).strip().lower()
        self.forced_role = os.environ.get("SUSU_BOT_FORCE_ROLE", "").strip().lower()
        self.infinite_boost = bool(
            os.environ.get("SUSU_BOT_INFINITE_BOOST", "0").strip().lower() in ("1", "true", "yes")
            or self.specialization == "aerial"
        )
        self._prev_ball_vel = np.zeros(3, dtype=np.float32)
        self._prev_boost: Dict[int, float] = {}
        self._prev_has_flip: Dict[int, bool] = {}
        self._bot_specialty: Dict[int, str] = {}
        self._carry_steps: Dict[int, int] = {}
        self._air_dribble_steps: Dict[int, int] = {}
        self._air_together_steps: Dict[int, int] = {}
        self._flick_state: Dict[int, Dict[str, Any]] = {}
        self._air_keepup_steps: Dict[int, int] = {}
        self._air_keepup_touches: Dict[int, int] = {}
        self._had_air_reset: Dict[int, bool] = {}
        self._last_mechanic: Dict[int, str] = {}
        self._episode_steps = 0
        self._started_as_kickoff = False
        self._prev_blue_score = 0
        self._prev_orange_score = 0

    def reset(self, initial_state: GameState):
        self._prev_ball_vel = np.asarray(initial_state.ball.linear_velocity, dtype=np.float32).copy()
        self._prev_boost = {p.car_id: float(p.boost_amount) for p in initial_state.players}
        self._prev_has_flip = {p.car_id: bool(p.has_flip) for p in initial_state.players}
        self._carry_steps = {p.car_id: 0 for p in initial_state.players}
        self._air_dribble_steps = {p.car_id: 0 for p in initial_state.players}
        self._air_together_steps = {p.car_id: 0 for p in initial_state.players}
        self._flick_state = {
            p.car_id: {"active": False, "steps": 0, "carry_steps": 0, "caught": False}
            for p in initial_state.players
        }
        self._air_keepup_steps = {p.car_id: 0 for p in initial_state.players}
        self._air_keepup_touches = {p.car_id: 0 for p in initial_state.players}
        self._had_air_reset = {p.car_id: False for p in initial_state.players}
        self._last_mechanic = {p.car_id: "" for p in initial_state.players}
        self._episode_steps = 0

        ball_pos = np.asarray(initial_state.ball.position, dtype=np.float32)
        ball_vel = np.asarray(initial_state.ball.linear_velocity, dtype=np.float32)
        scn = self.tracker.get("scenario", "")
        attacker_idx = int(self.tracker.get("attacker_idx", 0))
        self._started_as_kickoff = bool(
            scn == "kickoff"
            or (abs(ball_pos[0]) < 15.0 and abs(ball_pos[1]) < 15.0 and float(np.linalg.norm(ball_vel)) < 30.0)
        )
        self._prev_blue_score = int(getattr(initial_state, "blue_score", 0))
        self._prev_orange_score = int(getattr(initial_state, "orange_score", 0))

        matched_role = self.SCENARIO_TO_ROLE.get(scn, "")
        for idx, p in enumerate(initial_state.players):
            if self.forced_role:
                self._bot_specialty[p.car_id] = self.forced_role
            elif self.specialization == "aerial":
                self._bot_specialty[p.car_id] = "air_dribble" if idx == attacker_idx else "air_reset_keepup"
            elif self.specialization == "flick":
                self._bot_specialty[p.car_id] = "flick_dribble"
            elif self.specialization in ("movement", "shooting"):
                self._bot_specialty[p.car_id] = "all_rounder"
            elif self._started_as_kickoff:
                self._bot_specialty[p.car_id] = "kickoff_master"
            elif matched_role and idx == attacker_idx:
                self._bot_specialty[p.car_id] = matched_role
            elif matched_role:
                self._bot_specialty[p.car_id] = "all_rounder"
            else:
                self._bot_specialty[p.car_id] = random.choice(self.SPECIALTIES)

    def _mult(self, car_id: int, target_specialty: str) -> float:
        spec = self._bot_specialty.get(car_id, "all_rounder")
        if spec == target_specialty:
            return 2.4
        if target_specialty in ("air_reset_keepup", "air_dribble") and spec in ("air_reset_keepup", "air_dribble"):
            return 2.0
        if spec == "all_rounder":
            return 1.3
        return 1.0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0
        cid = player.car_id
        is_orange = (player.team_num == ORANGE_TEAM)
        forward_dir = -1.0 if is_orange else 1.0
        scn = self.tracker.get("scenario", "normal")
        role = self._bot_specialty.get(cid, "all_rounder")

        prev_act = (
            np.asarray(previous_action, dtype=np.float32)
            if previous_action is not None
            else np.zeros(8, dtype=np.float32)
        )
        if prev_act.ndim == 0 or len(prev_act) < 8:
            prev_act = np.zeros(8, dtype=np.float32)

        car_pos = np.asarray(player.car_data.position, dtype=np.float32)
        car_vel = np.asarray(player.car_data.linear_velocity, dtype=np.float32)
        car_ang_vel = np.asarray(player.car_data.angular_velocity, dtype=np.float32)
        rot_mtx = player.car_data.rotation_mtx()
        forward = rot_mtx[:, 0]
        up = rot_mtx[:, 2]

        car_speed = float(np.linalg.norm(car_vel))
        ball_pos = np.asarray(state.ball.position, dtype=np.float32)
        ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)
        ball_speed = float(np.linalg.norm(ball_vel))

        offset = ball_pos - car_pos
        dist = float(np.linalg.norm(offset))
        horiz_dist = float(np.linalg.norm(offset[:2]))
        vert_diff = float(offset[2])
        to_ball_dir = offset / (dist + 1e-6)

        is_in_air = (not player.on_ground) and (car_pos[2] > 55.0)
        is_ball_high = (ball_pos[2] > 260.0)
        is_supersonic = bool(car_speed > 2200.0)
        v_goal = float(ball_vel[1] * forward_dir)
        car_v_goal = float(car_vel[1] * forward_dir)
        delta_ball_speed = float(np.linalg.norm(ball_vel - self._prev_ball_vel))

        spec_flick = self._mult(cid, "flick_dribble")
        spec_musty = self._mult(cid, "musty")
        spec_air = self._mult(cid, "air_reset_keepup")
        spec_air_drib = self._mult(cid, "air_dribble")
        spec_kickoff = self._mult(cid, "kickoff_master")

        if cid == state.players[0].car_id:
            self._episode_steps += 1

        # =====================================================================
        # 1. KICKOFF SPEEDFLIP, SLOW KICKOFF & ANTI-RANDOM-FLICK DISCIPLINE
        # =====================================================================
        in_kickoff_window = self._started_as_kickoff and (
            self._episode_steps <= 55 or (abs(ball_pos[0]) < 25.0 and abs(ball_pos[1]) < 25.0)
        )
        if in_kickoff_window:
            v_to_ball = float(np.dot(car_vel, to_ball_dir))
            vel_align = float(np.dot(car_vel / (car_speed + 1e-6), to_ball_dir)) if car_speed > 100.0 else 1.0
            nose_align = float(np.dot(forward, to_ball_dir))

            is_speedflip_motion = abs(prev_act[2]) > 0.45 and abs(prev_act[4]) > 0.45
            if dist > 420.0 and v_to_ball > 1320.0 and vel_align > 0.87 and is_speedflip_motion:
                reward += 0.38 * spec_kickoff

            if dist > 400.0 and player.on_ground and 900.0 < car_speed < 1680.0 and nose_align > 0.92 and prev_act[5] < 0.5:
                reward += 0.16 * spec_kickoff

            if player.ball_touched and v_goal > 200.0:
                reward += 4.8 * spec_kickoff

            # Dead-zone narrowed to 360..1120 so diagonal speedflips (initiated around dist 1400..2600) are not penalized
            bad_midrange_jump = (360.0 < dist < 1120.0) and is_in_air and (car_speed < 1250.0 or not is_speedflip_motion)
            bad_unaligned_flip = is_in_air and (vel_align < 0.82)
            if bad_midrange_jump or bad_unaligned_flip:
                reward -= 1.50

        # =====================================================================
        # 2. DRIBBLE CARRY (ANTI-FARMING) + FLICK + CATCH GUIDANCE + CATCH
        # =====================================================================
        fstate = self._flick_state.setdefault(
            cid, {"active": False, "steps": 0, "carry_steps": 0, "caught": False}
        )
        if fstate["active"]:
            fstate["steps"] += 1
            if ball_pos[2] <= 108.0 or fstate["steps"] > 55:
                fstate["active"] = False
            elif not fstate["caught"] and ball_pos[2] > 135.0 and player.on_ground:
                # Claude section 5.6: progressive proximity bonus for staying under the flicked ball to catch it!
                preds = predict_ball_multi_horizon(ball_pos, ball_vel, horizons=PREDICTION_HORIZONS)
                landing_xy = preds[0.25][:2] if ball_pos[2] < 260.0 else preds[0.5][:2]
                dist_to_catch = float(np.linalg.norm(car_pos[:2] - landing_xy))
                if dist_to_catch < 480.0:
                    reward += 0.22 * (1.0 - dist_to_catch / 480.0) * spec_flick

        ball_on_hood = (horiz_dist < 175.0) and (65.0 < vert_diff < 205.0) and (ball_pos[2] > 110.0)
        prev_carry = self._carry_steps.get(cid, 0)

        is_flick_pop = (
            prev_carry >= 2
            and (not player.on_ground or prev_act[5] > 0.5)
            and (player.ball_touched or horiz_dist < 215.0)
            and delta_ball_speed > 175.0
            and ball_vel[2] > 115.0
            and v_goal > 150.0
        )

        if is_flick_pop:
            flick_bonus = (4.2 + 0.24 * min(25, prev_carry)) * spec_flick
            reward += flick_bonus
            self._flick_state[cid] = {
                "active": True,
                "steps": 0,
                "carry_steps": prev_carry,
                "caught": False,
            }
            self._carry_steps[cid] = 0
            self._last_mechanic[cid] = "flick"
        elif ball_on_hood:
            self._carry_steps[cid] = prev_carry + 1
            c_steps = self._carry_steps[cid]
            # Claude section 5.6 anti-farming: full carry reward only if advancing toward opponent goal!
            advancing_factor = 1.0 if (v_goal > 140.0 and car_v_goal > 140.0) else 0.20
            carry_rew = (0.16 * min(4.2, 1.0 + c_steps / 14.0)) * advancing_factor
            if v_goal > 180.0:
                carry_rew += 0.14 * min(3.0, v_goal / 800.0)
            reward += carry_rew * spec_flick
            self._last_mechanic[cid] = "dribble"

            if fstate["active"] and fstate["steps"] >= 2 and not fstate["caught"] and player.on_ground:
                fstate["caught"] = True
                reward += 9.5 * spec_flick
                self._last_mechanic[cid] = "flick_catch"
        else:
            self._carry_steps[cid] = 0

        # =====================================================================
        # 3. GROUND MUSTY & AERIAL MUSTY (Strictly verified on ball touch/whip)
        # =====================================================================
        is_musty_posture = (forward[2] < -0.22) and (horiz_dist < 225.0) and (vert_diff > 10.0)
        if is_musty_posture and role == "musty":
            # Subtle posture shaping when right behind the ball with nose tilted down
            reward += 0.10 * spec_musty

        is_backflip_whip = (prev_act[2] > 0.35 or float(np.linalg.norm(car_ang_vel)) > 3.1) and (
            prev_act[5] > 0.5 or not player.has_flip
        )
        if is_musty_posture and is_backflip_whip and (player.ball_touched or delta_ball_speed > 250.0) and v_goal > 180.0:
            shot_power = min(1.5, delta_ball_speed / CAR_MAX_SPEED)
            if car_pos[2] > 280.0 and ball_pos[2] > 300.0:
                alt_bonus = min(6.5, (car_pos[2] - 280.0) / 180.0)
                reward += (14.0 + alt_bonus + 2.5 * shot_power) * spec_musty
                self._last_mechanic[cid] = "aerial_musty"
            else:
                reward += (7.0 + 1.8 * shot_power) * spec_musty
                self._last_mechanic[cid] = "ground_musty"

        # =====================================================================
        # 4. AERIAL TAKEOFF, BOUNCING CONTESTS, AIR DRIBBLE & FLIP RESET
        # =====================================================================
        # 4a. Ground-to-Air Aerial Takeoff & Climbing Approach toward bouncing or airborne ball
        vel_to_pred = 0.0
        if ball_pos[2] > 180.0 and dist < 2600.0:
            preds = predict_ball_multi_horizon(ball_pos, ball_vel, horizons=PREDICTION_HORIZONS)
            target_air = preds[0.15] if dist < 500.0 else (preds[0.25] if dist < 900.0 else (preds[0.5] if dist < 1600.0 else preds[0.6]))
            to_pred = target_air - car_pos
            to_pred_dist = float(np.linalg.norm(to_pred)) + 1e-6
            to_pred_dir = to_pred / to_pred_dist
            vel_to_pred = float(np.dot(car_vel, to_pred_dir))
            nose_to_pred = float(np.dot(forward, to_pred_dir))

            if is_in_air and car_vel[2] > 120.0 and vel_to_pred > 260.0 and nose_to_pred > 0.50:
                climb_bonus = min(1.0, car_vel[2] / 650.0) * min(1.0, vel_to_pred / 1200.0)
                reward += (0.16 + 0.18 * climb_bonus) * max(spec_air, spec_air_drib)
            elif is_in_air and prev_act[6] > 0.5 and vel_to_pred < -100.0:
                # Penalize wasting boost in the air while flying away from the high ball
                reward -= 0.04

        # 4b. Direct Aerial Hit on elevated or bouncing ball (z > 180)
        if player.ball_touched and (is_in_air or car_pos[2] > 120.0) and ball_pos[2] > 180.0:
            height_mult = min(3.5, max(1.0, ball_pos[2] / 240.0))
            shot_mult = 1.0 + min(1.5, max(0.0, v_goal / 600.0))
            reward += 2.5 * height_mult * shot_mult * max(spec_air, spec_air_drib)
            if self._last_mechanic.get(cid, "") not in ("flip_reset", "flip_reset_shot", "aerial_musty"):
                self._last_mechanic[cid] = "aerial_touch"

        # 4c. Wall Jump & Intercept (off the wall into the air)
        is_on_wall_area = (abs(car_pos[0]) > 3500.0 or abs(car_pos[1]) > 4650.0 or car_pos[2] > 220.0)
        if is_on_wall_area and ball_pos[2] > 200.0:
            if is_in_air and player.ball_touched:
                reward += 4.2 * max(spec_air, spec_air_drib)
                self._last_mechanic[cid] = "wall_aerial"
            elif is_in_air and vel_to_pred > 300.0:
                reward += 0.22 * max(spec_air, spec_air_drib)

        # 4d. Continuous Air Dribble Carry under the ball (Gemini section 3)
        # Exponential Air-Together Time: plus la voiture et le ballon restent en l'air ensemble, plus la récompense explose
        if is_in_air and ball_pos[2] > 200.0 and dist < 420.0:
            self._air_together_steps[cid] = self._air_together_steps.get(cid, 0) + 1
            together_cnt = self._air_together_steps[cid]
            air_together_mult = min(8.0, 1.0 + (together_cnt / 12.0) ** 1.3)
            reward += (0.45 if self.specialization == "aerial" else 0.15) * air_together_mult
            if v_goal > 100.0:
                reward += 0.35 * min(3.0, v_goal / 600.0)
        else:
            self._air_together_steps[cid] = 0

        is_under_ball_air = is_in_air and is_ball_high and (car_pos[2] < ball_pos[2]) and (dist < 350.0) and (horiz_dist < 220.0)
        if is_under_ball_air:
            self._air_dribble_steps[cid] = self._air_dribble_steps.get(cid, 0) + 1
            ad_steps = self._air_dribble_steps[cid]
            ad_mult = min(8.0, 1.0 + ad_steps / 10.0) if self.specialization == "aerial" else min(3.0, 0.5 + ad_steps / 18.0)
            ad_rew = (0.55 if self.specialization == "aerial" else 0.26) * ad_mult
            if v_goal > 150.0:
                ad_rew += 0.35 * min(3.0, v_goal / 700.0)
            reward += ad_rew * max(spec_air_drib, spec_air)
            if self._last_mechanic.get(cid, "") not in ("flip_reset", "flip_reset_shot", "aerial_musty"):
                self._last_mechanic[cid] = "air_dribble"
        elif not is_in_air:
            self._air_dribble_steps[cid] = 0

        # 4c. True Mid-Air Flip Reset (above ground z > 420 only)
        wheel_dir = -up
        dot_wheels = float(np.dot(wheel_dir, to_ball_dir))
        prev_flip = self._prev_has_flip.get(cid, bool(player.has_flip))

        # Anti-cheese: penalize trying to land wheels on top of a ground/low ball for a fake reset
        if is_in_air and ball_pos[2] < 320.0 and horiz_dist < 185.0 and dot_wheels > 0.45:
            reward -= 0.85

        # Subtle wheel-alignment approach shaping when airborne close to a high ball in air_reset role
        if (
            is_in_air
            and ball_pos[2] > 450.0
            and car_pos[2] > 380.0
            and dist < 270.0
            and dot_wheels > 0.40
            and not self._had_air_reset.get(cid, False)
        ):
            reward += 0.18 * dot_wheels * spec_air

        got_flip_back_in_air = is_in_air and (not prev_flip) and bool(player.has_flip)
        wheels_touch_in_air = (
            is_in_air
            and player.ball_touched
            and bool(player.has_flip)
            and dot_wheels > 0.42
            and horiz_dist < 185.0
            and car_pos[2] < ball_pos[2]
        )
        if ball_pos[2] > 420.0 and car_pos[2] > 380.0 and (got_flip_back_in_air or wheels_touch_in_air):
            if not self._had_air_reset.get(cid, False):
                height_scale = min(5.0, (ball_pos[2] - 420.0) / 220.0)
                reward += (12.0 + height_scale) * spec_air
                self._had_air_reset[cid] = True
                self._last_mechanic[cid] = "flip_reset"
        elif self._had_air_reset.get(cid, False) and player.ball_touched and is_in_air and car_pos[2] > 260.0 and v_goal > 180.0:
            reward += 12.5 * spec_air
            self._last_mechanic[cid] = "flip_reset_shot"

        self._prev_has_flip[cid] = bool(player.has_flip)

        # 4d. Continuous Air Keep-Up Streak without touching ground, walls, or ceiling
        in_opp_goal_mouth = (abs(ball_pos[0]) < 890.0 and ball_pos[2] < 640.0 and ball_pos[1] * forward_dir > 4950.0)
        ball_touched_ground = ball_pos[2] <= 125.0
        ball_touched_wall = (
            abs(ball_pos[0]) >= (SIDE_WALL_X - 135.0)
            or (abs(ball_pos[1]) >= (BACK_WALL_Y - 140.0) and not in_opp_goal_mouth)
        )
        ball_touched_ceiling = ball_pos[2] >= (CEILING_Z - 135.0)

        if ball_touched_ground or ball_touched_wall or ball_touched_ceiling:
            self._air_keepup_steps[cid] = 0
            self._air_keepup_touches[cid] = 0
            if ball_touched_ground:
                self._had_air_reset[cid] = False
        else:
            if player.ball_touched and is_in_air and ball_pos[2] > 260.0 and car_pos[2] > 200.0:
                self._air_keepup_touches[cid] = self._air_keepup_touches.get(cid, 0) + 1
                self._air_keepup_steps[cid] = self._air_keepup_steps.get(cid, 0) + 1
                touches = self._air_keepup_touches[cid]
                keepup_base = 4.8 if self.specialization == "aerial" else 2.8
                reward += (keepup_base * min(8.0, 1.0 + 0.75 * touches)) * max(spec_air, spec_air_drib)
                if self._last_mechanic.get(cid, "") not in ("flip_reset", "flip_reset_shot", "aerial_musty"):
                    self._last_mechanic[cid] = "air_keepup"
            elif self._air_keepup_touches.get(cid, 0) >= 1 and is_in_air and ball_pos[2] > 260.0 and dist < 480.0:
                self._air_keepup_steps[cid] = self._air_keepup_steps.get(cid, 0) + 1
                k_steps = self._air_keepup_steps[cid]
                keepup_rew = 0.15 * min(5.0, 1.0 + k_steps / 10.0)
                if v_goal > 200.0:
                    keepup_rew += 0.08
                reward += keepup_rew * max(spec_air, spec_air_drib)

        # =====================================================================
        # 5. STRICT CROSS-PENALTIES & ANTI-PASSIVITY (SOL VS AIR)
        # =====================================================================
        is_air_scenario = scn in ("air_keepup", "air_dribble", "air_reset", "musty_air", "air_takeoff", "air_cross", "air_backboard", "bouncing_ball", "wall") or role in ("air_dribble", "air_reset_keepup")
        is_ground_scenario = scn in ("dribble_flick", "musty_ground") or role == "flick_dribble"

        if self.specialization == "aerial" and player.on_ground and ball_pos[2] > 220.0 and self._episode_steps > 6:
            # Pénalité stricte anti-gazon lors de la spécialisation aérienne
            reward -= 0.35
        elif is_air_scenario and player.on_ground and ball_pos[2] > 320.0 and self._episode_steps > 8:
            # Penalize staying glued to the turf when the ball is high in an aerial scenario
            reward -= 0.16
        elif (190.0 < ball_pos[2] < 750.0) and dist < 550.0 and player.on_ground and prev_act[5] < 0.5 and self._episode_steps > 8:
            # Penalize waiting passively under a bouncing ball without jumping to contest!
            reward -= 0.16
        elif is_ground_scenario and is_in_air and ball_pos[2] < 190.0 and not fstate["active"] and prev_carry == 0:
            # Penalize jumping randomly into the air when the ball is on the ground in a ground-dribble scenario
            reward -= 0.18
        elif ball_pos[2] < 115.0 and dist < 260.0 and is_in_air and not fstate["active"] and prev_carry == 0 and prev_act[5] > 0.5:
            # Penalize systematically flipping/jumping into flat ground rollouts (preserves smooth ground shots)
            reward -= 0.12

        # =====================================================================
        # 5b. SPECIALISATION DÉPLACEMENTS, RECOVERY & TIRS
        # =====================================================================
        if self.specialization == "movement":
            if is_supersonic:
                reward += 0.35 if prev_act[6] < 0.5 else 0.12
            if (player.on_ground or is_on_wall_area) and abs(up[2]) > 0.82 and float(np.linalg.norm(car_ang_vel)) < 1.8:
                reward += 0.25
            if is_on_wall_area and car_speed > 1200.0:
                reward += 0.25
        elif self.specialization == "shooting":
            if v_goal > 150.0:
                power_scale = min(3.0, (delta_ball_speed / CAR_MAX_SPEED) * 2.5)
                reward += (2.0 + 1.5 * power_scale) * min(3.0, v_goal / 800.0)
            if ball_pos[2] > 180.0 and player.ball_touched and v_goal > 150.0:
                reward += 5.0 * min(2.5, v_goal / 700.0)
        elif self.specialization == "flick":
            if is_flick_pop:
                reward += 5.0
            if fstate["active"] and fstate["caught"]:
                reward += 8.0

        # =====================================================================
        # 6. GOAL SCORED REWARDS & MECHANIC GOAL JACKPOTS
        # =====================================================================
        blue_sc = int(getattr(state, "blue_score", self._prev_blue_score))
        orange_sc = int(getattr(state, "orange_score", self._prev_orange_score))
        team_scored = (
            (not is_orange and blue_sc > self._prev_blue_score)
            or (is_orange and orange_sc > self._prev_orange_score)
            or (ball_pos[1] * forward_dir > 5080.0 and abs(ball_pos[0]) < 880.0 and ball_pos[2] < 630.0)
        )
        if team_scored:
            reward += 12.0 + 4.0 * min(1.5, ball_speed / CAR_MAX_SPEED)
            last_mech = self._last_mechanic.get(cid, "")
            k_steps = self._air_keepup_steps.get(cid, 0)
            k_touches = self._air_keepup_touches.get(cid, 0)

            if k_touches >= 1 and k_steps > 0:
                reward += (16.0 + 0.50 * min(100, k_steps) + 4.0 * min(10, k_touches)) * max(spec_air, spec_air_drib)
            if self._had_air_reset.get(cid, False) or last_mech in ("flip_reset", "flip_reset_shot"):
                reward += 16.0 * spec_air
            if last_mech == "aerial_musty":
                reward += 18.0 * spec_musty
            elif last_mech == "ground_musty":
                reward += 12.0 * spec_musty
            elif last_mech in ("flick", "flick_catch", "dribble") or fstate.get("active", False):
                c_bonus = 0.30 * min(40, max(self._carry_steps.get(cid, 0), fstate.get("carry_steps", 0)))
                catch_bonus = 10.0 if fstate.get("caught", False) else 0.0
                reward += (12.0 + c_bonus + catch_bonus) * spec_flick

        self._prev_blue_score = blue_sc
        self._prev_orange_score = orange_sc

        # =====================================================================
        # 7. GENERAL SHOT POWER, PINCHES, BOOST CONSERVATION & DISCIPLINE
        # =====================================================================
        if player.ball_touched:
            if v_goal > 100.0:
                reward += min(1.5, (delta_ball_speed / CAR_MAX_SPEED) * 1.2)
                if (SIDE_WALL_X - abs(car_pos[0]) < 260.0 or BACK_WALL_Y - abs(car_pos[1]) < 280.0) and ball_speed > 1700.0:
                    reward += 2.5
            elif v_goal < -350.0:
                reward -= 0.8 * (abs(v_goal) / CAR_MAX_SPEED)

        if not self.infinite_boost and self.specialization != "aerial":
            curr_b = float(player.boost_amount)
            prev_b = self._prev_boost.get(cid, curr_b)
            if car_speed >= 2200.0 and curr_b < prev_b:
                reward -= 0.05
            self._prev_boost[cid] = curr_b

        if player.on_ground and car_speed < 80.0:
            reward -= 0.03

        self._prev_ball_vel = ball_vel.copy()
        return float(reward)


FIELD_NORM = np.array([4096.0, 5120.0, 2048.0], dtype=np.float32)
GOAL_Y = 5120.0


class SusuSimObsBuilder(ObsBuilder):
    """
    142-dimensional observation builder for rlgym-sim.
    Matches src/susu_obs.py:
    - 19 self car
    - 15 current ball state
    - 12 opponent car
    - 22 key boost pads + 12 multi-horizon ball predictions (0.1s, 0.5s, 1.0s, 2.0s) = 34
    - 14 specialized mechanics perception features
    - 12 fine-grained multi-horizon ball predictions (0.05s, 0.15s, 0.25s, 0.60s)
    - 36 high-resolution trajectory preds (12 extra horizons up to 2.50s)
    Total = 142 floats.
    """

    OBS_SIZE = 142

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        prev_act = np.asarray(previous_action, dtype=np.float32) if previous_action is not None else np.zeros(8, dtype=np.float32)
        if prev_act.ndim == 0 or len(prev_act) < 8:
            prev_act = np.zeros(8, dtype=np.float32)

        is_orange = (player.team_num == ORANGE_TEAM)
        car = player.inverted_car_data if is_orange else player.car_data
        car_pos = np.asarray(car.position, dtype=np.float32)
        car_vel = np.asarray(car.linear_velocity, dtype=np.float32)
        car_ang_vel = np.asarray(car.angular_velocity, dtype=np.float32)
        rot_mtx = car.rotation_mtx()
        forward = rot_mtx[:, 0]
        up = rot_mtx[:, 2]

        car_speed = float(np.linalg.norm(car_vel))
        is_supersonic = float(car_speed > 2200.0)
        on_ground = float(player.on_ground)
        has_flip = float(player.has_flip)
        boost_fraction = float(player.boost_amount)

        ball = state.inverted_ball if is_orange else state.ball
        ball_pos = np.asarray(ball.position, dtype=np.float32)
        ball_vel = np.asarray(ball.linear_velocity, dtype=np.float32)
        ball_ang_vel = np.asarray(ball.angular_velocity, dtype=np.float32)

        rel_ball_pos = ball_pos - car_pos
        rel_ball_vel = ball_vel - car_vel

        # 1. Self car vector (19)
        self_vec = np.concatenate([
            car_pos / FIELD_NORM,
            car_vel / 2300.0,
            forward,
            up,
            car_ang_vel / 5.5,
            [on_ground, has_flip, boost_fraction, is_supersonic],
        ]).astype(np.float32)

        # 2. Ball vector (15)
        ball_vec = np.concatenate([
            rel_ball_pos / FIELD_NORM,
            rel_ball_vel / 2300.0,
            ball_pos / FIELD_NORM,
            ball_vel / 2300.0,
            ball_ang_vel / 6.0,
        ]).astype(np.float32)

        # 3. Opponent vector (12)
        opponents = [p for p in state.players if p.team_num != player.team_num]
        if opponents:
            opp = opponents[0]
            opp_car = opp.inverted_car_data if is_orange else opp.car_data
            opp_pos = np.asarray(opp_car.position, dtype=np.float32)
            opp_vel = np.asarray(opp_car.linear_velocity, dtype=np.float32)
            opp_rot = opp_car.rotation_mtx()
            opp_fwd = opp_rot[:, 0]
            opp_vec = np.concatenate([
                (opp_pos - car_pos) / FIELD_NORM,
                (opp_vel - car_vel) / 2300.0,
                opp_fwd,
                [float(opp.on_ground), float(opp.has_flip), float(opp.boost_amount)],
            ]).astype(np.float32)
        else:
            opp_vec = np.zeros(12, dtype=np.float32)

        # 4. Key boost pads (22) + Multi-horizon ball predictions at 0.1s, 0.5s, 1.0s, 2.0s (12) = 34
        pads = state.inverted_boost_pads if is_orange else state.boost_pads
        if pads is not None and len(pads) >= 34:
            pads_arr = np.asarray(pads[:34], dtype=np.float32)
            key_pads = (pads_arr[KEY_BOOST_INDICES] > 0.5).astype(np.float32)
        else:
            key_pads = np.ones(22, dtype=np.float32)

        preds = predict_ball_multi_horizon(ball_pos, ball_vel, horizons=PREDICTION_HORIZONS_20)
        pred_ball_vec = np.concatenate([
            (preds[0.1] - car_pos) / FIELD_NORM,
            (preds[0.5] - car_pos) / FIELD_NORM,
            (preds[1.0] - car_pos) / FIELD_NORM,
            (preds[2.0] - car_pos) / FIELD_NORM,
        ]).astype(np.float32)
        sub_horizon_vec = np.concatenate([
            (preds[0.05] - car_pos) / FIELD_NORM,
            (preds[0.15] - car_pos) / FIELD_NORM,
            (preds[0.25] - car_pos) / FIELD_NORM,
            (preds[0.6] - car_pos) / FIELD_NORM,
        ]).astype(np.float32)
        extra_horizon_vec = np.concatenate([
            (preds[h] - car_pos) / FIELD_NORM for h in EXTRA_HORIZONS_12
        ]).astype(np.float32)
        boost_vec = np.concatenate([key_pads, pred_ball_vec]).astype(np.float32)

        # 5. Specialized Mechanics Perception Features (14)
        ox, oy, oz = float(rel_ball_pos[0]), float(rel_ball_pos[1]), float(rel_ball_pos[2])
        horiz_dist = math.sqrt(ox * ox + oy * oy)
        vert_diff = oz
        dist_3d = math.sqrt(ox * ox + oy * oy + oz * oz) + 1e-6

        ball_on_hood = float(horiz_dist < 165.0 and 60.0 < vert_diff < 200.0)
        musty_readiness = float(forward[2] < -0.25 and horiz_dist < 220.0 and vert_diff > 30.0)

        dot_wheels = float((-up[0] * ox - up[1] * oy - up[2] * oz) / dist_3d)
        flip_reset_readiness = float(
            not player.on_ground
            and ball_pos[2] > 420.0
            and car_pos[2] > 380.0
            and vert_diff < 0
            and dot_wheels > 0.45
            and horiz_dist < 185.0
        )
        under_ball = float(not player.on_ground and horiz_dist < 130.0 and car_pos[2] < ball_pos[2])

        approaching_backboard = (ball_pos[1] > 3500.0 and ball_vel[1] > 300.0)
        double_tap_read = float(approaching_backboard and ball_pos[2] > 600.0 and not player.on_ground)

        dist_to_wall = float(min(4096.0 - abs(float(car_pos[0])), 4096.0 - abs(float(ball_pos[0]))))
        wall_pinch_read = float(dist_to_wall < 350.0 and horiz_dist < 250.0)
        ground_pinch_read = float(car_vel[2] < -800.0 and ball_pos[2] < 220.0 and horiz_dist < 200.0)

        gx = -float(car_pos[0])
        gy = -GOAL_Y - float(car_pos[1])
        gz = -float(car_pos[2])
        g_norm = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-6
        defending_alignment = float((forward[0] * gx + forward[1] * gy + forward[2] * gz) / g_norm)

        bx_net = float(ball_pos[0])
        by_net = float(ball_pos[1]) + GOAL_Y
        bz_net = float(ball_pos[2])
        dist_ball_to_net = math.sqrt(bx_net * bx_net + by_net * by_net + bz_net * bz_net)
        threat_level = max(0.0, min(1.0, 1.0 - (dist_ball_to_net / 6000.0)))

        boost_wasting = float(car_speed > 2200.0 and prev_act[6] > 0.5)
        speedflip_curl = float(abs(float(prev_act[2])) > 0.5 and abs(float(prev_act[4])) > 0.5)
        ceiling_prox = max(0.0, min(1.0, (2048.0 - float(car_pos[2])) / 2048.0))
        wall_prox = max(0.0, min(1.0, (4096.0 - abs(float(car_pos[0]))) / 4096.0))
        bvx, bvy, bvz = float(ball_vel[0]), float(ball_vel[1]), float(ball_vel[2])
        kickoff_active = float(abs(float(ball_pos[0])) < 10.0 and abs(float(ball_pos[1])) < 10.0 and (bvx * bvx + bvy * bvy + bvz * bvz) < 2500.0)

        mechanic_vec = np.array([
            ball_on_hood,
            musty_readiness,
            flip_reset_readiness,
            under_ball,
            double_tap_read,
            wall_pinch_read,
            ground_pinch_read,
            defending_alignment,
            threat_level,
            boost_wasting,
            speedflip_curl,
            ceiling_prox,
            wall_prox,
            kickoff_active,
        ], dtype=np.float32)

        return np.concatenate([self_vec, ball_vec, opp_vec, boost_vec, mechanic_vec, sub_horizon_vec, extra_horizon_vec]).astype(np.float32)


def build_rocketsim_reward_fn(
    scenario_tracker: Dict[str, Any] = None,
    base_reward_scale: float = None,
    specialization: str = "mix",
) -> CombinedReward:
    """
    Configures the rlgym-sim CombinedReward tailored to the active specialization
    ('aerial', 'flick', 'movement', 'shooting', or 'mix') with optional `SUSU_BOT_BASE_REWARD_SCALE`.
    """
    spec = (specialization or os.environ.get("SUSU_BOT_SPECIALIZATION", "mix")).strip().lower()
    if base_reward_scale is None:
        try:
            base_reward_scale = float(os.environ.get("SUSU_BOT_BASE_REWARD_SCALE", "1.0"))
        except ValueError:
            base_reward_scale = 1.0
    s = float(base_reward_scale)

    if spec == "aerial":
        # 1. Spécialisation Aérienne & Air Dribble & Flip Resets
        # Récompenses massives pour le temps passé en l'air ensemble, boost illimité, zéro pénalité boost
        rew_funcs = (
            VelocityPlayerToBallReward(),
            FaceBallReward(),
            LiuDistancePlayerToBallReward(),
            TouchBallReward(aerial_weight=6.0),
            AlignBallGoal(defense=1.0, offense=1.0),
            VelocityBallToGoalReward(),
            LiuDistanceBallToGoalReward(),
            EventReward(
                team_goal=25.0,
                concede=-10.0,
                touch=2.5,
                shot=5.0,
                save=4.5,
                demo=0.5,
                boost_pickup=0.0,
            ),
            SaveBoostReward(),
            SusuMechanicsSimReward(scenario_tracker=scenario_tracker, specialization="aerial"),
        )
        rew_weights = (
            1.8 * s,   # VelocityPlayerToBallReward
            0.5 * s,   # FaceBallReward
            0.2 * s,   # LiuDistancePlayerToBallReward
            5.0 * s,   # TouchBallReward (aerial_weight=6.0)
            1.0 * s,   # AlignBallGoal
            2.2 * s,   # VelocityBallToGoalReward
            0.3 * s,   # LiuDistanceBallToGoalReward
            1.0,       # EventReward
            0.0,       # SaveBoostReward (aucun malus de boost en aérienne)
            1.5,       # SusuMechanicsSimReward (aériennes amplifiées)
        )
    elif spec == "flick":
        # 2. Spécialisation Flicks & Dribbles au Toit
        rew_funcs = (
            VelocityPlayerToBallReward(),
            FaceBallReward(),
            LiuDistancePlayerToBallReward(),
            TouchBallReward(aerial_weight=1.0),
            AlignBallGoal(defense=1.0, offense=1.5),
            VelocityBallToGoalReward(),
            LiuDistanceBallToGoalReward(),
            EventReward(
                team_goal=25.0,
                concede=-10.0,
                touch=1.2,
                shot=6.0,
                save=4.5,
                demo=0.5,
                boost_pickup=0.3,
            ),
            SaveBoostReward(),
            SusuMechanicsSimReward(scenario_tracker=scenario_tracker, specialization="flick"),
        )
        rew_weights = (
            1.2 * s,   # VelocityPlayerToBallReward
            0.3 * s,   # FaceBallReward
            0.2 * s,   # LiuDistancePlayerToBallReward
            2.5 * s,   # TouchBallReward
            1.5 * s,   # AlignBallGoal
            3.5 * s,   # VelocityBallToGoalReward
            0.3 * s,   # LiuDistanceBallToGoalReward
            1.0,       # EventReward
            0.08 * s,  # SaveBoostReward
            1.5,       # SusuMechanicsSimReward (flicks amplifiés)
        )
    elif spec == "movement":
        # 3. Spécialisation Déplacements, Recovery, Rotations & Vitesse
        rew_funcs = (
            VelocityPlayerToBallReward(),
            FaceBallReward(),
            LiuDistancePlayerToBallReward(),
            TouchBallReward(aerial_weight=1.5),
            AlignBallGoal(defense=1.0, offense=1.0),
            VelocityBallToGoalReward(),
            LiuDistanceBallToGoalReward(),
            EventReward(
                team_goal=25.0,
                concede=-10.0,
                touch=1.5,
                shot=5.0,
                save=5.0,
                demo=1.0,
                boost_pickup=0.5,
            ),
            SaveBoostReward(),
            SusuMechanicsSimReward(scenario_tracker=scenario_tracker, specialization="movement"),
        )
        rew_weights = (
            3.5 * s,   # VelocityPlayerToBallReward (vitesse vers la balle prioritaire)
            0.8 * s,   # FaceBallReward
            0.3 * s,   # LiuDistancePlayerToBallReward
            2.5 * s,   # TouchBallReward
            1.0 * s,   # AlignBallGoal
            2.0 * s,   # VelocityBallToGoalReward
            0.3 * s,   # LiuDistanceBallToGoalReward
            1.0,       # EventReward
            0.25 * s,  # SaveBoostReward (gestion rigoureuse du boost)
            1.5,       # SusuMechanicsSimReward (movement & recovery amplifiés)
        )
    elif spec == "shooting":
        # 4. Spécialisation Tirs, Précision & Cages (Finishing / Sniper)
        rew_funcs = (
            VelocityPlayerToBallReward(),
            FaceBallReward(),
            LiuDistancePlayerToBallReward(),
            TouchBallReward(aerial_weight=2.0),
            AlignBallGoal(defense=1.0, offense=3.0),
            VelocityBallToGoalReward(),
            LiuDistanceBallToGoalReward(),
            EventReward(
                team_goal=35.0,
                concede=-15.0,
                touch=1.5,
                shot=8.0,
                save=5.0,
                demo=0.5,
                boost_pickup=0.2,
            ),
            SaveBoostReward(),
            SusuMechanicsSimReward(scenario_tracker=scenario_tracker, specialization="shooting"),
        )
        rew_weights = (
            1.2 * s,   # VelocityPlayerToBallReward
            0.4 * s,   # FaceBallReward
            0.2 * s,   # LiuDistancePlayerToBallReward
            3.0 * s,   # TouchBallReward
            2.5 * s,   # AlignBallGoal (alignement chirurgical vers le but)
            5.0 * s,   # VelocityBallToGoalReward (vitesse maximale vers la cage adverse)
            0.3 * s,   # LiuDistanceBallToGoalReward
            1.2,       # EventReward (buts et tirs très récompensés)
            0.05 * s,  # SaveBoostReward
            1.5,       # SusuMechanicsSimReward (tirs et rebonds amplifiés)
        )
    else:
        # Mix équilibré (défaut)
        rew_funcs = (
            VelocityPlayerToBallReward(),
            FaceBallReward(),
            LiuDistancePlayerToBallReward(),
            TouchBallReward(aerial_weight=2.5),
            AlignBallGoal(defense=1.0, offense=1.0),
            VelocityBallToGoalReward(),
            LiuDistanceBallToGoalReward(),
            EventReward(
                team_goal=25.0,
                concede=-10.0,
                touch=1.2,
                shot=5.0,
                save=4.5,
                demo=0.8,
                boost_pickup=0.3,
            ),
            SaveBoostReward(),
            SusuMechanicsSimReward(scenario_tracker=scenario_tracker, specialization="mix"),
        )
        rew_weights = (
            1.5 * s,
            0.3 * s,
            0.2 * s,
            3.5 * s,
            1.2 * s,
            3.2 * s,
            0.3 * s,
            1.0,
            0.05 * s,
            1.0,
        )

    return CombinedReward(reward_functions=rew_funcs, reward_weights=rew_weights)


def build_rocketsim_env(
    tick_skip: int = 8,
    spawn_opponents: bool = True,
    team_size: int = 1,
    timeout_seconds: float = 300.0,
    no_touch_seconds: float = 30.0,
    end_on_goal: bool = True,
    is_scenario_arena: bool = True,
    specialization: str = None,
    infinite_boost: bool = False,
):
    """
    Constructs a C++ RocketSim environment via rlgym_sim.make() with shared `scenario_tracker`
    linking `VariedRandomStateSetter`, `SusuMechanicsSimReward`, and `ScenarioTerminal`.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = "1"
    os.environ["MALLOC_ARENA_MAX"] = "2"

    ensure_rocketsim_meshes_initialized()

    spec = (specialization or os.environ.get("SUSU_BOT_SPECIALIZATION", "mix")).strip().lower()
    is_inf_boost = bool(
        infinite_boost
        or os.environ.get("SUSU_BOT_INFINITE_BOOST", "0").strip().lower() in ("1", "true", "yes")
        or spec == "aerial"
    )

    fps = 120 // tick_skip  # 15 steps per second
    timeout_ticks = int(round(timeout_seconds * fps))
    no_touch_ticks = int(round(no_touch_seconds * fps))

    scenario_tracker: Dict[str, Any] = {"scenario": "mix" if is_scenario_arena else "normal", "attacker_idx": 0}
    terminal_conditions = [
        TimeoutCondition(timeout_ticks),
        NoTouchTimeoutCondition(no_touch_ticks),
    ]
    if is_scenario_arena:
        terminal_conditions.append(ScenarioTerminal(scenario_tracker=scenario_tracker))
    if end_on_goal:
        terminal_conditions.append(GoalScoredCondition())

    forced_mech = None if is_scenario_arena else "normal"
    base_scale = None if is_scenario_arena else 1.0

    env = rlgym_sim.make(
        tick_skip=tick_skip,
        spawn_opponents=spawn_opponents,
        team_size=team_size,
        copy_gamestate_every_step=False,
        terminal_conditions=terminal_conditions,
        reward_fn=build_rocketsim_reward_fn(scenario_tracker=scenario_tracker, base_reward_scale=base_scale, specialization=spec),
        obs_builder=SusuSimObsBuilder(),
        action_parser=LookupAction(),
        state_setter=VariedRandomStateSetter(
            scenario_tracker=scenario_tracker,
            forced_mechanic=forced_mech,
            specialization=spec,
            infinite_boost=is_inf_boost,
        ),
    )

    if is_inf_boost:
        orig_step = env.step
        def step_inf_boost(actions):
            res = orig_step(actions)
            try:
                for car in env._game.cars:
                    st = car.get_state()
                    if st.boost < 98.0:
                        st.boost = 100.0
                        car.set_state(st)
            except Exception:
                pass
            return res
        env.step = step_inf_boost

    return env


class MultiArenaSimEnv:
    """
    Vectorized Multi-Arena C++ RocketSim wrapper: runs `n_arenas` independent 1v1 matches
    inside a single CPU worker process.
    Following Claude section 3.6, `SUSU_BOT_SCENARIO_ARENAS` (default: 4 out of 6 arenas) run
    focused mechanics scenarios with `ScenarioTerminal`, while the remaining arenas (2 out of 6)
    play full normal matches to prevent catastrophic forgetting.
    Supports `specialization` ('aerial', 'flick', 'movement', 'shooting') and `infinite_boost`.
    """

    def __init__(
        self,
        n_arenas: int = 6,
        tick_skip: int = 8,
        spawn_opponents: bool = True,
        team_size: int = 1,
        timeout_seconds: float = 300.0,
        no_touch_seconds: float = 30.0,
        end_on_goal: bool = True,
        specialization: str = None,
        infinite_boost: bool = False,
    ):
        self.n_arenas = max(1, int(n_arenas))
        self.agents_per_arena = (2 if spawn_opponents else 1) * team_size
        self.total_agents = self.n_arenas * self.agents_per_arena
        self.specialization = (specialization or os.environ.get("SUSU_BOT_SPECIALIZATION", "mix")).strip().lower()
        self.infinite_boost = bool(
            infinite_boost
            or os.environ.get("SUSU_BOT_INFINITE_BOOST", "0").strip().lower() in ("1", "true", "yes")
            or self.specialization == "aerial"
        )

        default_scn = max(1, int(round(self.n_arenas * 0.67))) if self.n_arenas > 1 else 1
        try:
            n_scn = int(os.environ.get("SUSU_BOT_SCENARIO_ARENAS", str(default_scn)))
        except ValueError:
            n_scn = default_scn
        self.n_scenario_arenas = max(0, min(self.n_arenas, n_scn))

        self.envs = [
            build_rocketsim_env(
                tick_skip=tick_skip,
                spawn_opponents=spawn_opponents,
                team_size=team_size,
                timeout_seconds=timeout_seconds,
                no_touch_seconds=no_touch_seconds,
                end_on_goal=end_on_goal,
                is_scenario_arena=(idx < self.n_scenario_arenas),
                specialization=self.specialization,
                infinite_boost=self.infinite_boost,
            )
            for idx in range(self.n_arenas)
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._obs_buf = np.empty((self.total_agents, SusuSimObsBuilder.OBS_SIZE), dtype=np.float32)
        self._rew_buf = [0.0] * self.total_agents
        self._dones_buf = np.zeros(self.total_agents, dtype=np.float32)

    def reset(self):
        k = self.agents_per_arena
        for i, env in enumerate(self.envs):
            self._obs_buf[i * k : (i + 1) * k] = env.reset()
        self._dones_buf.fill(0.0)
        return self._obs_buf

    def step(self, actions):
        k = self.agents_per_arena
        for i, env in enumerate(self.envs):
            s = i * k
            e = s + k
            obs, rew, done, info = env.step(actions[s:e])
            if done:
                obs = env.reset()
                self._dones_buf[s:e] = 1.0
            else:
                self._dones_buf[s:e] = 0.0
            self._obs_buf[s:e] = obs
            if k == 2:
                self._rew_buf[s] = float(rew[0])
                self._rew_buf[s + 1] = float(rew[1])
            else:
                for j in range(k):
                    self._rew_buf[s + j] = float(rew[j])

            if self.infinite_boost:
                try:
                    for car in env._game.cars:
                        st = car.get_state()
                        if st.boost < 98.0:
                            st.boost = 100.0
                            car.set_state(st)
                except Exception:
                    pass

        return self._obs_buf, self._rew_buf, False, {"state": self._dones_buf}

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass


class RocketSimEnvFactory:
    """Top-level picklable callable for Windows/Linux multiprocessing workers."""

    def __init__(
        self,
        tick_skip: int = 8,
        spawn_opponents: bool = True,
        team_size: int = 1,
        timeout_seconds: float = 300.0,
        no_touch_seconds: float = 30.0,
        end_on_goal: bool = True,
        arenas_per_worker: int = 1,
        specialization: str = None,
        infinite_boost: bool = False,
        **kwargs,
    ):
        self.tick_skip = tick_skip
        self.spawn_opponents = spawn_opponents
        self.team_size = team_size
        self.timeout_seconds = timeout_seconds
        self.no_touch_seconds = no_touch_seconds
        self.end_on_goal = end_on_goal
        self.arenas_per_worker = max(1, int(arenas_per_worker))
        self.specialization = specialization
        self.infinite_boost = infinite_boost

    def __call__(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["TORCH_NUM_THREADS"] = "1"
        os.environ["MALLOC_ARENA_MAX"] = "2"
        if self.arenas_per_worker > 1:
            return MultiArenaSimEnv(
                n_arenas=self.arenas_per_worker,
                tick_skip=self.tick_skip,
                spawn_opponents=self.spawn_opponents,
                team_size=self.team_size,
                timeout_seconds=self.timeout_seconds,
                no_touch_seconds=self.no_touch_seconds,
                end_on_goal=self.end_on_goal,
                specialization=self.specialization,
                infinite_boost=self.infinite_boost,
            )
        return build_rocketsim_env(
            tick_skip=self.tick_skip,
            spawn_opponents=self.spawn_opponents,
            team_size=self.team_size,
            timeout_seconds=self.timeout_seconds,
            no_touch_seconds=self.no_touch_seconds,
            end_on_goal=self.end_on_goal,
            specialization=self.specialization,
            infinite_boost=self.infinite_boost,
        )


def get_effective_ram_gb() -> float:
    """
    Returns available physical RAM in GB, detecting both system physical RAM and
    Linux container cgroup v1/v2 memory limits (essential for Vast.ai docker containers).
    Handles both private cgroup namespaces and host/nested cgroup hierarchies from /proc/self/cgroup.
    """
    import psutil

    try:
        total_bytes = int(psutil.virtual_memory().total)
    except Exception:
        total_bytes = 32 * (1024 ** 3)

    cgroup_candidates = []

    # 1. Inspect /proc/self/cgroup to discover container-specific subpaths
    proc_cgroup = Path("/proc/self/cgroup")
    if proc_cgroup.is_file():
        try:
            for line in proc_cgroup.read_text().splitlines():
                parts = line.strip().split(":")
                if len(parts) == 3:
                    controller, path = parts[1], parts[2].lstrip("/")
                    subpaths = [path]
                    curr = path
                    while curr and "/" in curr:
                        curr = curr.rsplit("/", 1)[0]
                        subpaths.append(curr)

                    for sp in subpaths:
                        if not sp:
                            continue
                        if controller == "":  # cgroup v2
                            cgroup_candidates.append((Path(f"/sys/fs/cgroup/{sp}/memory.max"), 2))
                            cgroup_candidates.append((Path(f"/sys/fs/cgroup/{sp}/memory.high"), 2))
                        elif "memory" in controller.split(","):  # cgroup v1
                            cgroup_candidates.append((Path(f"/sys/fs/cgroup/memory/{sp}/memory.limit_in_bytes"), 1))
                            cgroup_candidates.append((Path(f"/sys/fs/cgroup/{sp}/memory.limit_in_bytes"), 1))
        except Exception:
            pass

    # 2. Add standard direct mount points for cgroup v2 and v1
    cgroup_candidates.extend([
        (Path("/sys/fs/cgroup/memory.max"), 2),
        (Path("/sys/fs/cgroup/memory.high"), 2),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), 1),
        (Path("/sys/fs/cgroup/memory.limit_in_bytes"), 1),
    ])

    for cand_path, version in cgroup_candidates:
        if cand_path.is_file():
            try:
                val = cand_path.read_text().strip()
                if val and val != "max":
                    limit = int(val)
                    if version == 2:
                        if 0 < limit < total_bytes:
                            total_bytes = limit
                    else:  # version == 1
                        if 0 < limit < min(total_bytes, 1024 ** 5):
                            total_bytes = limit
            except Exception:
                pass

    return max(0.5, total_bytes / (1024.0 ** 3))


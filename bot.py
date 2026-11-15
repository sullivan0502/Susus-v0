"""
susu_bot (Susu-v0): High-performance Rocket League bot for RLGym training and RLBot competition.
Combines SusuAgent (TorchScript neural policy inference susu-model.pt + Susu lookup table ball-aiming)
and SusuBot (multi-horizon 0.1s/0.5s/1.0s/2.0s trajectory rendering, speedflip/slow kickoffs, and mechanics).
"""

import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import torch
from torch.distributions import Categorical

# Ensure current bot directory is on sys.path
_cur_dir = str(Path(__file__).resolve().parent)
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

from mechanics_controller import MechanicsController, Vec3
from necto_brain import NectoBrain
from nexto_brain import NextoBrain
from susu_obs import (
    SusuObsBuilder,
    compute_obs_ball_aim_logits_bias,
    get_lookup_table,
)

# ==============================================================================
# NEXTO COMPETITIVE SPEEDFLIP KICKOFF TABLE (168 TICKS = 1.4s AT 120 FPS)
# ==============================================================================
KICKOFF_CONTROLS = (
    11 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 4 * 4 * [SimpleControllerState(throttle=1, boost=True, steer=-1)]
    + 2 * 4 * [SimpleControllerState(throttle=1, jump=True, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
    + 13 * 4 * [SimpleControllerState(throttle=1, pitch=1, boost=True)]
    + 10 * 4 * [SimpleControllerState(throttle=1, roll=1, pitch=0.5)]
)

KICKOFF_NUMPY = np.array([
    [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, float(scs.jump), float(scs.boost), float(scs.handbrake)]
    for scs in KICKOFF_CONTROLS
], dtype=np.float32)


class SusuAgent:
    """
    Loads TorchScript policy (susu-model.pt) and parses 90 discrete actions,
    combining neural policy logits with Susu ball-aiming lookup table guidance.
    """

    def __init__(self, model_path: Optional[str] = None, aim_assist_weight: float = 0.0):
        cur_dir = Path(__file__).resolve().parent
        model_file = cur_dir / "susu-model.pt" if model_path is None else Path(model_path)

        self.actor = None
        self.expected_in_dim = 142
        self.device = "cpu"
        self.aim_assist_weight = aim_assist_weight
        self._lookup_table = get_lookup_table()

        if model_file.exists():
            try:
                self.actor = torch.jit.load(str(model_file), map_location=self.device)
                self.actor.eval()
                torch.set_num_threads(1)
                for p in self.actor.parameters():
                    if p.ndim == 2:
                        self.expected_in_dim = int(p.shape[1])
                        break
                print(f"[susu_bot/Susu-v0] Modele TorchScript charge avec succes : {model_file.name} (in_dim={self.expected_in_dim})")
            except Exception as e:
                print(f"[susu_bot/Susu-v0] Avertissement: Impossible de charger {model_file}: {e}")
                self.actor = None
        else:
            print(f"[susu_bot/Susu-v0] Aucun checkpoint neural {model_file.name} trouve. Modele hybride Susu actif.")

    @property
    def is_ready(self) -> bool:
        return self.actor is not None

    def act(self, obs: np.ndarray, beta: float = 1.0) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        obs = np.asarray(obs, dtype=np.float32)
        aim_bias = compute_obs_ball_aim_logits_bias(obs, self._lookup_table)

        if self.actor is None:
            action_idx = int(np.argmax(aim_bias))
            return self._lookup_table[action_idx].copy(), None

        if obs.shape[-1] > self.expected_in_dim:
            obs_in = obs[: self.expected_in_dim]
        elif obs.shape[-1] < self.expected_in_dim:
            obs_in = np.pad(obs, (0, self.expected_in_dim - obs.shape[-1]))
        else:
            obs_in = obs
        obs_tensor = torch.from_numpy(obs_in).float().unsqueeze(0)

        with torch.no_grad():
            out = self.actor(obs_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]

        logits = out.squeeze(0)

        if self.aim_assist_weight > 0.0 and len(aim_bias) == logits.shape[-1]:
            logits_std = float(torch.std(logits).item())
            scale_factor = max(1.0, logits_std)
            bias_tensor = torch.from_numpy(aim_bias).float().to(logits.device)
            logits = logits + (self.aim_assist_weight * scale_factor) * bias_tensor

        if beta == 1.0:
            action_idx = int(torch.argmax(logits, dim=-1).item())
        elif beta == -1.0:
            action_idx = int(torch.argmin(logits, dim=-1).item())
        else:
            if beta == 0.0:
                logits[torch.isfinite(logits)] = 0.0
            else:
                scale = math.log((beta + 1.0) / (1.0 - beta), 3.0)
                logits = logits * scale
            dist = Categorical(logits=logits)
            action_idx = int(dist.sample().item())

        return self._lookup_table[action_idx].copy(), None


class SusuBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.tick_skip = 8  # 120 / 8 = 15 decisions/second
        self.ticks = 0
        self.prev_time = 0.0
        self.update_action = True

        self.game_state: GameState = None
        self.obs_builder: SusuObsBuilder = None
        self.agent: SusuAgent = None
        self.mechanics: MechanicsController = None

        self.action = np.zeros(8, dtype=np.float32)
        self.controls = SimpleControllerState()
        self.beta = 1.0
        self.hardcoded_kickoffs = True
        self.stochastic_kickoffs = False
        self.field_info = None
        self.brain_mode = "nexto"  # "nexto" (SSL EARL Transformer), "susu" (Susu-model + Aim-Assist), "necto"
        self.necto_brain = None
        self.nexto_brain = None
        self.kickoff_index = -1
        self.kickoff_strike_tick = 0
        self.kickoff_dodge_tick = 0
        self.prev_round_active = False
        self.mechanics_assist = False  # Laisse 100% du pilotage au réseau neural sans forçage heuristique

        print(f"[susu_bot/Susu-v0] Initialise avec succes - Index: {index}, Equipe: {team} (Mode: {self.brain_mode})")

    def initialize_agent(self):
        self.field_info = self.get_field_info()
        self.game_state = GameState(self.field_info)
        self.obs_builder = SusuObsBuilder()
        self.mechanics = MechanicsController(index=self.index, team=self.team)
        self.agent = SusuAgent(aim_assist_weight=0.6)  # Modèle Susu prêt avec assistance anti-blocage

        if self.brain_mode == "necto":
            self.necto_brain = NectoBrain(field_info=self.field_info)
        elif self.brain_mode == "nexto":
            self.nexto_brain = NextoBrain(field_info=self.field_info)

        self.ticks = self.tick_skip
        self.prev_time = 0.0
        self.action = np.zeros(8, dtype=np.float32)
        self.controls = SimpleControllerState()
        self.update_action = True
        self.kickoff_index = -1
        self.kickoff_strike_tick = 0
        self.kickoff_dodge_tick = 0
        self.prev_round_active = False

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = 1.0 / 120.0 if self.prev_time == 0.0 else (cur_time - self.prev_time)
        self.prev_time = cur_time

        ticks_elapsed = max(1, round(delta * 120))

        # Reset kickoff timing at the exact instant countdown ends (GO!)
        if not self.prev_round_active and packet.game_info.is_round_active:
            self.kickoff_index = 0
            ticks_elapsed = 1
        self.prev_round_active = bool(packet.game_info.is_round_active)

        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        self.mechanics.update_flip_tracking(packet)

        # 1. Action inference at tick_skip rate
        if self.update_action and len(self.game_state.players) > self.index:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            self.game_state.players = [player] + teammates + opponents

            beta = self.beta
            if packet.game_info.is_match_ended:
                beta = 0.0
            elif self.stochastic_kickoffs and packet.game_info.is_kickoff_pause:
                beta = 0.5

            if self.brain_mode == "necto" and self.necto_brain is not None and self.necto_brain.is_ready:
                necto_act, _ = self.necto_brain.get_action_and_weights(player, self.game_state, self.action, beta=beta)
                if necto_act is not None:
                    self.action = necto_act
                else:
                    obs = self.obs_builder.build_obs(player, self.game_state, self.action)
                    self.action, _ = self.agent.act(obs, beta=beta)
            elif self.brain_mode == "nexto" and self.nexto_brain is not None and self.nexto_brain.is_ready:
                nexto_act, _ = self.nexto_brain.get_action_and_logits(player, self.game_state, self.action, beta=beta)
                if nexto_act is not None:
                    self.action = nexto_act
                else:
                    obs = self.obs_builder.build_obs(player, self.game_state, self.action)
                    self.action, _ = self.agent.act(obs, beta=beta)
            else:
                obs = self.obs_builder.build_obs(player, self.game_state, self.action)
                self.action, _ = self.agent.act(obs, beta=beta)

            self._apply_action_to_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True
            self._apply_action_to_controls(self.action)

        # 2. Hardcoded Speedflip Kickoff (Necto algorithm)
        if self.hardcoded_kickoffs:
            self.maybe_do_kickoff(packet, ticks_elapsed)

        self._render_hud(packet)
        return self.controls

    def is_kickoff_situation(self, packet: GameTickPacket) -> bool:
        ball = packet.game_ball.physics
        ball_dist_center = math.sqrt(ball.location.x ** 2 + ball.location.y ** 2)
        ball_speed = math.sqrt(ball.velocity.x ** 2 + ball.velocity.y ** 2 + ball.velocity.z ** 2)
        ball_at_center = (ball_dist_center < 35.0 and ball_speed < 120.0)
        in_strike = (0 < self.kickoff_strike_tick < 18)
        in_speedflip = (0 <= self.kickoff_index < len(KICKOFF_NUMPY))
        return bool(packet.game_info.is_kickoff_pause or ball_at_center or in_strike or in_speedflip)

    def _is_designated_taker(self, packet: GameTickPacket) -> bool:
        num_cars = max(1, getattr(packet, "num_cars", self.index + 1))
        teammates = [i for i in range(num_cars) if packet.game_cars[i].team == self.team and i != self.index]

        # In 1v1 (solo on team): ALWAYS True! Never yield kickoff to opponent or go for boost!
        if len(teammates) == 0:
            return True

        # In 2v2/3v3: only compare against teammates
        my_car = packet.game_cars[self.index]
        ball = packet.game_ball.physics
        my_pos = np.array([my_car.physics.location.x, my_car.physics.location.y])
        ball_pos = np.array([ball.location.x, ball.location.y])
        my_dist = np.linalg.norm(my_pos - ball_pos)

        for mate_idx in teammates:
            mate_car = packet.game_cars[mate_idx]
            mate_pos = np.array([mate_car.physics.location.x, mate_car.physics.location.y])
            mate_dist = np.linalg.norm(mate_pos - ball_pos)
            if mate_dist < my_dist - 20.0:
                return False
            elif abs(mate_dist - my_dist) <= 20.0:
                # Tied: left goes
                if self.team == 0:
                    if mate_pos[0] < my_pos[0]:
                        return False
                else:
                    if mate_pos[0] > my_pos[0]:
                        return False
        return True

    def maybe_do_kickoff(self, packet: GameTickPacket, ticks_elapsed: int):
        if not self.is_kickoff_situation(packet):
            self.kickoff_index = -1
            self.kickoff_strike_tick = 0
            self.kickoff_dodge_tick = 0
            return

        # Determine taker if not set
        if self.kickoff_index == -1:
            if self._is_designated_taker(packet):
                self.kickoff_index = 0
                self.kickoff_strike_tick = 0
                self.kickoff_dodge_tick = 0
            else:
                self.kickoff_index = -2

        if self.kickoff_index < 0:
            return

        my_car = packet.game_cars[self.index]
        ball = packet.game_ball.physics
        ball_target = Vec3(ball.location.x, ball.location.y, 93.0)
        car_pos = Vec3(my_car.physics.location)
        dist_to_ball = car_pos.dist(ball_target)

        # During countdown (3-2-1): hold full throttle and boost, lock kickoff_index at 0
        if not packet.game_info.is_round_active:
            self.kickoff_index = 0
            self.kickoff_strike_tick = 0
            self.kickoff_dodge_tick = 0
            self.controls = SimpleControllerState(throttle=1.0, boost=True, steer=0.0)
            return

        # Determine if car started on the left side of its perspective (needs mirrored roll/yaw/steer)
        is_car_left = (my_car.physics.location.x < 0) if self.team == 0 else (my_car.physics.location.x > 0)

        # 1. High-speed Nexto Speedflip Execution (0 to 167 ticks = 1.4s at 120 FPS)
        if 0 <= self.kickoff_index < len(KICKOFF_NUMPY):
            act = KICKOFF_NUMPY[self.kickoff_index].copy()
            if is_car_left:
                act[1] = -act[1]  # steer
                act[3] = -act[3]  # yaw
                act[4] = -act[4]  # roll
            self.action = act
            self._apply_action_to_controls(act)
            self.kickoff_index += max(1, round(ticks_elapsed))
            return

        # 2. Advance ticks beyond speedflip table
        self.kickoff_index += max(1, round(ticks_elapsed))

        # Continuous laser-steer directly toward the center of the ball
        steer_to_ball = self.mechanics._steer_toward(my_car, ball_target)

        # 3. 50/50 Block Trigger at contact:
        if dist_to_ball < 420.0 or self.kickoff_strike_tick > 0:
            if self.kickoff_strike_tick == 0:
                self.kickoff_strike_tick = 1

            self.controls = SimpleControllerState()
            self.controls.throttle = 1.0
            self.controls.steer = steer_to_ball
            self.controls.boost = bool(dist_to_ball > 100.0 and my_car.boost > 0)

            if self.kickoff_strike_tick <= 3:
                # Pop up to ball equator
                self.controls.jump = True
                self.controls.pitch = 0.0
                self.controls.yaw = steer_to_ball
            elif self.kickoff_strike_tick <= 6:
                # Release jump
                self.controls.jump = False
                self.controls.pitch = 0.0
                self.controls.yaw = steer_to_ball
            else:
                # Forward dodge squarely through ball center
                self.controls.jump = True
                self.controls.pitch = -1.0
                self.controls.yaw = float(np.clip(steer_to_ball * 0.7, -0.5, 0.5))

            self.kickoff_strike_tick += max(1, round(ticks_elapsed))
            if self.kickoff_strike_tick >= 18:
                # Strike completed, exit kickoff for this round
                self.kickoff_index = -2
                self.kickoff_strike_tick = 0
            return

        # 4. Post-speedflip supersonic pursuit into ball:
        self.controls = SimpleControllerState()
        self.controls.throttle = 1.0
        self.controls.steer = steer_to_ball
        self.controls.boost = bool(my_car.boost > 0)

        if not my_car.has_wheel_contact and car_pos.z < 100.0:
            # Level wheels to regain full ground traction immediately
            self.controls.pitch = float(np.clip(-my_car.physics.rotation.pitch * 2.0, -1.0, 1.0))
            self.controls.roll = float(np.clip(-my_car.physics.rotation.roll * 2.0, -1.0, 1.0))
            self.controls.yaw = steer_to_ball

    def _apply_action_to_controls(self, action: np.ndarray):
        self.controls.throttle = float(action[0])
        self.controls.steer = float(action[1])
        self.controls.pitch = float(action[2])
        self.controls.yaw = float(action[3])
        self.controls.roll = float(action[4])
        self.controls.jump = bool(action[5] > 0.5)
        self.controls.boost = bool(action[6] > 0.5)
        self.controls.handbrake = bool(action[7] > 0.5)

    def _render_hud(self, packet: GameTickPacket, status: Optional[str] = None):
        """Renders multi-horizon ball trajectory preview (0.1s, 0.5s, 1.0s, 2.0s) and telemetry in RLBot."""
        if self.renderer is None:
            return

        if status is None:
            status = f"Mode {self.brain_mode.upper()} (Nexto Speedflip Kickoff + SSL EARL Policy)"

        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)

        ball_prediction = None
        try:
            ball_prediction = self.get_ball_prediction_struct()
        except Exception:
            ball_prediction = None

        horizon_vecs, target_location, chosen_h = self.mechanics.get_multi_horizon_predictions(
            packet, ball_prediction_struct=ball_prediction
        )

        prev_pt = ball_location
        for h in self.mechanics.HORIZONS:
            pt = horizon_vecs[h]
            self.renderer.draw_line_3d(prev_pt, pt, self.renderer.cyan())
            self.renderer.draw_rect_3d(pt, 5, 5, True, self.renderer.cyan(), centered=True)
            prev_pt = pt

        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_string_3d(
            car_location,
            1,
            1,
            f"Speed: {car_velocity.length():.1f} | Anticipation: {chosen_h:.2f}s (max 2.0s) | {status}",
            self.renderer.white(),
        )
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)

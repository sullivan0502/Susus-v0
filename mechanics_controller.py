"""
Unified Hybrid Susu + Mechanics Controller & 3D Math Utilities for susu_bot (Susu-v0).
Integrates:
1. Vec3, Orientation, relative_location, and find_slice_at_time (replaces src/util/*).
2. Clean, deterministic Speedflip Kickoff and Slow Kickoff (prevents random/erratic flicking on kickoff).
3. Multi-horizon ball trajectory anticipation at 0.1s, 0.5s, 1.0s, and 2.0s.
4. Specialized mechanics: roof dribble carry, flick + catch-after-flick, ground & aerial musty flicks,
   true mid-air flip resets (above ground only, z > 420), continuous air-dribble keep-up, and boost conservation.
"""

import math
import random
from typing import Dict, Optional, Tuple, Union
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

try:
    from susu_obs import (
        KICKOFF_CONTROLS,
        KICKOFF_NUMPY,
        PREDICTION_HORIZONS_20,
        PREDICTION_HORIZONS,
        compute_ball_aim_action,
        predict_ball_multi_horizon,
        select_optimal_ball_intercept,
    )
except ImportError:
    from src.susu_obs import (
        KICKOFF_CONTROLS,
        KICKOFF_NUMPY,
        PREDICTION_HORIZONS_20,
        PREDICTION_HORIZONS,
        compute_ball_aim_action,
        predict_ball_multi_horizon,
        select_optimal_ball_intercept,
    )

GOAL_Y = 5120.0
SIDE_WALL_X = 4096.0


# ==============================================================================
# 1. 3D VECTOR, ORIENTATION & BALL PREDICTION UTILITIES
# ==============================================================================
class Vec3:
    __slots__ = ["x", "y", "z"]

    def __init__(self, x: Union[float, "Vec3"] = 0.0, y: float = 0.0, z: float = 0.0):
        if hasattr(x, "x"):
            self.x = float(x.x)
            self.y = float(x.y) if hasattr(x, "y") else 0.0
            self.z = float(x.z) if hasattr(x, "z") else 0.0
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    def __getitem__(self, item: int):
        return (self.x, self.y, self.z)[item]

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, scale: float) -> "Vec3":
        return Vec3(self.x * scale, self.y * scale, self.z * scale)

    def __rmul__(self, scale: float) -> "Vec3":
        return self * scale

    def __truediv__(self, scale: float) -> "Vec3":
        return self * (1.0 / float(scale))

    def __str__(self) -> str:
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def __repr__(self) -> str:
        return self.__str__()

    def flat(self) -> "Vec3":
        return Vec3(self.x, self.y, 0.0)

    def length(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def dist(self, other: "Vec3") -> float:
        return (self - other).length()

    def normalized(self) -> "Vec3":
        l = self.length()
        return self / l if l > 1e-6 else Vec3(0.0, 0.0, 0.0)

    def rescale(self, new_len: float) -> "Vec3":
        return new_len * self.normalized()

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def ang_to(self, ideal: "Vec3") -> float:
        denom = self.length() * ideal.length()
        if denom <= 1e-6:
            return 0.0
        cos_ang = max(-1.0, min(1.0, self.dot(ideal) / denom))
        return math.acos(cos_ang)


class Orientation:
    def __init__(self, rotation):
        self.yaw = float(rotation.yaw)
        self.roll = float(rotation.roll)
        self.pitch = float(rotation.pitch)

        cr = math.cos(self.roll)
        sr = math.sin(self.roll)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        self.forward = Vec3(cp * cy, cp * sy, sp)
        self.right = Vec3(cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr)
        self.up = Vec3(-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr)


def relative_location(center: Vec3, ori: Orientation, target: Vec3) -> Vec3:
    diff = target - center
    return Vec3(diff.dot(ori.forward), diff.dot(ori.right), diff.dot(ori.up))


def find_slice_at_time(ball_prediction, game_time: float):
    if ball_prediction is None or getattr(ball_prediction, "num_slices", 0) <= 0:
        return None
    start_time = ball_prediction.slices[0].game_seconds
    approx_index = int((game_time - start_time) * 60)
    if 0 <= approx_index < ball_prediction.num_slices:
        return ball_prediction.slices[approx_index]
    return None


# ==============================================================================
# 2. MECHANICS CONTROLLER (Kickoffs, Multi-Horizon Intercepts & Advanced Plays)
# ==============================================================================
class MechanicsController:
    FLIP_TIMEOUT = 1.5
    HORIZONS = PREDICTION_HORIZONS_20

    def __init__(self, index: int, team: int):
        self.index = index
        self.team = team
        self.state_tick = 0
        self.active_mechanic: Optional[str] = None
        self.mechanic_timer = 0
        self.kickoff_strategy = "speedflip"
        self.kickoff_index = -1
        self.kickoff_strike_tick = 0
        self.has_reset = False
        self.reset_tick = 0
        self.latest_wheel_contact_time = 0.0
        self.dribble_ticks = 0
        self.post_flick_catch_timer = 0

    def update_flip_tracking(self, packet: GameTickPacket) -> bool:
        my_car = packet.game_cars[self.index]
        cur_time = packet.game_info.seconds_elapsed

        if my_car.has_wheel_contact:
            self.latest_wheel_contact_time = cur_time

        time_since_contact = cur_time - self.latest_wheel_contact_time
        jump_expired = (time_since_contact > self.FLIP_TIMEOUT) and my_car.jumped
        return not my_car.double_jumped and not jump_expired

    def get_multi_horizon_predictions(
        self, packet: GameTickPacket, ball_prediction_struct=None
    ) -> Tuple[Dict[float, Vec3], Vec3, float]:
        my_car = packet.game_cars[self.index]
        car_pos = np.array(
            [my_car.physics.location.x, my_car.physics.location.y, my_car.physics.location.z],
            dtype=np.float32,
        )
        car_vel = np.array(
            [my_car.physics.velocity.x, my_car.physics.velocity.y, my_car.physics.velocity.z],
            dtype=np.float32,
        )
        ball = packet.game_ball.physics
        ball_pos = np.array([ball.location.x, ball.location.y, ball.location.z], dtype=np.float32)
        ball_vel = np.array([ball.velocity.x, ball.velocity.y, ball.velocity.z], dtype=np.float32)

        intercept_np, chosen_h, preds_np = select_optimal_ball_intercept(
            car_pos=car_pos,
            car_vel=car_vel,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            horizons=self.HORIZONS,
        )

        cur_time = packet.game_info.seconds_elapsed
        horizon_vecs: Dict[float, Vec3] = {}
        for h in self.HORIZONS:
            vec = Vec3(float(preds_np[h][0]), float(preds_np[h][1]), float(preds_np[h][2]))
            if ball_prediction_struct is not None:
                try:
                    sl = find_slice_at_time(ball_prediction_struct, cur_time + h)
                    if sl is not None:
                        vec = Vec3(sl.physics.location)
                except Exception:
                    pass
            horizon_vecs[h] = vec

        target_vec = horizon_vecs.get(chosen_h, Vec3(float(intercept_np[0]), float(intercept_np[1]), float(intercept_np[2])))
        if chosen_h == self.HORIZONS[0] and Vec3(my_car.physics.location).dist(Vec3(ball.location)) < 380.0:
            target_vec = Vec3(ball.location)

        return horizon_vecs, target_vec, chosen_h

    def reset_kickoff_strategy(self, opponent_distance: float = 3000.0):
        self.kickoff_strike_tick = 0
        self.kickoff_strategy = "direct_5050"

    def is_kickoff_active(self, packet: GameTickPacket) -> bool:
        if not packet.game_info.is_round_active:
            return False
        if packet.game_info.is_kickoff_pause:
            return True
        if self.kickoff_strike_tick >= 16:
            return False

        # Safety check: if car has driven past the midfield ball line, exit kickoff immediately
        my_car = packet.game_cars[self.index]
        if self.team == 0 and my_car.physics.location.y > 120.0:
            return False
        if self.team == 1 and my_car.physics.location.y < -120.0:
            return False

        ball = packet.game_ball.physics
        ball_dist_center = math.sqrt(ball.location.x ** 2 + ball.location.y ** 2)
        ball_speed = math.sqrt(ball.velocity.x ** 2 + ball.velocity.y ** 2 + ball.velocity.z ** 2)

        # Kickoff remains active while ball is stationary near center
        if ball_dist_center < 150.0 and ball_speed < 120.0:
            return True
        # Or while completing the 50/50 contact strike
        if 0 < self.kickoff_strike_tick < 16:
            return True
        return False

    def _is_designated_kickoff_taker(self, packet: GameTickPacket) -> bool:
        ball_pos = np.array([
            packet.game_ball.physics.location.x,
            packet.game_ball.physics.location.y,
        ])
        num_cars = max(1, getattr(packet, "num_cars", self.index + 1))
        teammates = [i for i in range(num_cars) if packet.game_cars[i].team == self.team]

        # In 1v1 (or solo team), our bot is ALWAYS the designated taker:
        if len(teammates) <= 1:
            return True

        teammate_positions = np.array([
            [packet.game_cars[i].physics.location.x, packet.game_cars[i].physics.location.y]
            for i in teammates
        ])
        distances = np.linalg.norm(teammate_positions - ball_pos, axis=1)
        my_sub_idx = teammates.index(self.index)
        min_dist = distances.min()
        if abs(min_dist - distances[my_sub_idx]) > 10.0:
            return False

        # Left-goes competitive rule among tied teammates
        for sub_idx, car_idx in enumerate(teammates):
            if abs(distances[sub_idx] - distances[my_sub_idx]) <= 10.0 and car_idx != self.index:
                other_is_left = (
                    teammate_positions[sub_idx, 0] < teammate_positions[my_sub_idx, 0]
                    if self.team == 0
                    else teammate_positions[sub_idx, 0] > teammate_positions[my_sub_idx, 0]
                )
                if other_is_left:
                    return False
        return True

    def handle_kickoff(
        self, packet: GameTickPacket, ticks_elapsed: int
    ) -> Tuple[Optional[SimpleControllerState], bool]:
        if not packet.game_info.is_round_active:
            self.kickoff_index = -1
            self.kickoff_strike_tick = 0
            return None, False

        my_car = packet.game_cars[self.index]
        car_pos = Vec3(my_car.physics.location)
        ball_pos = Vec3(packet.game_ball.physics.location)
        dist = car_pos.dist(ball_pos)
        step_ticks = max(1, round(ticks_elapsed))

        # Detect opponent car for adaptive kickoff
        opp_car = None
        for i in range(packet.num_cars):
            if packet.game_cars[i].team != self.team:
                opp_car = packet.game_cars[i]
                break
        opp_dist = Vec3(opp_car.physics.location).dist(ball_pos) if opp_car is not None else 3500.0

        if packet.game_info.is_kickoff_pause:
            # During countdown (3-2-1): car controls are locked by game engine.
            # Prime kickoff indices and hold throttle + boost for instantaneous launch at GO!
            if self._is_designated_kickoff_taker(packet):
                self.kickoff_index = 0
                self.kickoff_strike_tick = 0
                self.reset_kickoff_strategy(opponent_distance=opp_dist)
            else:
                self.kickoff_index = -2
            controls = SimpleControllerState(throttle=1.0, boost=True)
            return controls, True

        if not self.is_kickoff_active(packet):
            self.kickoff_index = -1
            self.kickoff_strike_tick = 0
            return None, False

        if self.kickoff_index == -1:
            if self._is_designated_kickoff_taker(packet):
                self.kickoff_index = 0
                self.reset_kickoff_strategy(opponent_distance=opp_dist)
            else:
                self.kickoff_index = -2

        if self.kickoff_index < 0:
            return None, False

        steer_to_ball = self._steer_toward(my_car, Vec3(0.0, 0.0, 93.0))

        # 50/50 Block Trigger at contact:
        if dist < 340.0 and my_car.has_wheel_contact and self.kickoff_strike_tick == 0:
            self.kickoff_strike_tick = 1

        if self.kickoff_strike_tick > 0:
            controls = SimpleControllerState()
            controls.throttle = 1.0
            controls.steer = steer_to_ball
            controls.boost = (dist > 150.0 and my_car.boost > 0)
            if self.kickoff_strike_tick <= 3:
                # Initial pop jump into the air squarely facing ball
                controls.jump = True
                controls.pitch = 0.0
                controls.yaw = steer_to_ball
            elif self.kickoff_strike_tick <= 6:
                # Release jump button to allow second jump / dodge
                controls.jump = False
                controls.pitch = 0.0
                controls.yaw = steer_to_ball
            else:
                # Solid forward dodge right through the middle of the ball
                controls.jump = True
                controls.pitch = -1.0
                controls.yaw = float(np.clip(steer_to_ball * 0.7, -0.5, 0.5))

            self.kickoff_strike_tick += step_ticks
            return controls, True

        # Pure continuous high-speed pursuit directly towards ball center (0, 0, 93)
        self.kickoff_index += step_ticks
        controls = SimpleControllerState()
        controls.throttle = 1.0
        controls.steer = steer_to_ball
        controls.boost = True
        controls.jump = False

        # In-flight recovery in case of accidental bump or ground disconnect
        if not my_car.has_wheel_contact:
            controls.pitch = float(np.clip(-my_car.physics.rotation.pitch * 3.0, -1.0, 1.0))
            controls.roll = float(np.clip(-my_car.physics.rotation.roll * 3.0, -1.0, 1.0))
            controls.yaw = steer_to_ball

        return controls, True

    def compute_ball_aim_controls(self, packet: GameTickPacket) -> Tuple[SimpleControllerState, float]:
        my_car = packet.game_cars[self.index]
        ori = Orientation(my_car.physics.rotation)
        car_pos = np.array([my_car.physics.location.x, my_car.physics.location.y, my_car.physics.location.z], dtype=np.float32)
        car_vel = np.array([my_car.physics.velocity.x, my_car.physics.velocity.y, my_car.physics.velocity.z], dtype=np.float32)
        forward = np.array([ori.forward.x, ori.forward.y, ori.forward.z], dtype=np.float32)
        up = np.array([ori.up.x, ori.up.y, ori.up.z], dtype=np.float32)

        ball = packet.game_ball.physics
        ball_pos = np.array([ball.location.x, ball.location.y, ball.location.z], dtype=np.float32)
        ball_vel = np.array([ball.velocity.x, ball.velocity.y, ball.velocity.z], dtype=np.float32)

        target_goal_y = GOAL_Y if self.team == 0 else -GOAL_Y
        action, aim_angle = compute_ball_aim_action(
            car_pos=car_pos,
            car_vel=car_vel,
            forward=forward,
            up=up,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            on_ground=bool(my_car.has_wheel_contact),
            boost_amount=float(my_car.boost),
            target_goal_y=target_goal_y,
        )

        controls = SimpleControllerState()
        controls.throttle = float(action[0])
        controls.steer = float(action[1])
        controls.pitch = float(action[2])
        controls.yaw = float(action[3])
        controls.roll = float(action[4])
        controls.jump = bool(action[5] > 0.5)
        controls.boost = bool(action[6] > 0.5)
        controls.handbrake = bool(action[7] > 0.5)
        return controls, aim_angle

    def check_mechanic_trigger(
        self, packet: GameTickPacket, prev_action: np.ndarray
    ) -> Optional[SimpleControllerState]:
        my_car = packet.game_cars[self.index]
        car_pos = Vec3(my_car.physics.location)
        car_vel = Vec3(my_car.physics.velocity)
        ball_pos = Vec3(packet.game_ball.physics.location)
        ball_vel = Vec3(packet.game_ball.physics.velocity)
        dist = car_pos.dist(ball_pos)

        forward_dir = 1.0 if self.team == 0 else -1.0
        own_goal_y = -GOAL_Y if self.team == 0 else GOAL_Y
        opp_goal_y = GOAL_Y if self.team == 0 else -GOAL_Y
        opp_goal = Vec3(0.0, opp_goal_y, 100.0)

        if self.active_mechanic is not None and self.mechanic_timer > 0:
            self.mechanic_timer -= 1
            controls = SimpleControllerState()
            controls.throttle = 1.0
            controls.steer = self._steer_toward(my_car, opp_goal)
            if self.active_mechanic == "hood_flick":
                if self.mechanic_timer >= 6:
                    controls.jump = True
                    controls.pitch = -0.2
                elif self.mechanic_timer >= 4:
                    controls.jump = False
                else:
                    controls.jump = True
                    controls.pitch = -1.0
                    controls.yaw = controls.steer * 0.4
                if self.mechanic_timer == 0:
                    self.active_mechanic = None
                    self.post_flick_catch_timer = 35
                return controls

            elif self.active_mechanic in ("ground_musty", "aerial_musty"):
                if self.mechanic_timer >= 8:
                    controls.jump = True
                    controls.pitch = -1.0
                    controls.boost = True
                elif self.mechanic_timer >= 5:
                    controls.jump = False
                    controls.pitch = -1.0
                else:
                    controls.jump = True
                    controls.pitch = 1.0
                    controls.boost = True
                if self.mechanic_timer == 0:
                    self.active_mechanic = None
                return controls

        # 1. Critical Goal Line Defense / Save
        ball_heading_to_own_goal = (ball_vel.y * (-forward_dir) > 500.0)
        dist_to_own_goal = abs(ball_pos.y - own_goal_y)
        if ball_heading_to_own_goal and dist_to_own_goal < 2500.0 and dist < 1200.0:
            controls = SimpleControllerState()
            controls.throttle = 1.0
            controls.boost = (car_vel.length() < 2200.0 and my_car.boost > 0)
            controls.steer = self._steer_toward(my_car, ball_pos)
            if dist < 450.0 and not my_car.has_wheel_contact:
                controls.jump = True
                controls.pitch = -1.0
            elif dist < 350.0 and my_car.has_wheel_contact and ball_pos.z > 150.0:
                controls.jump = True
            return controls

        # 2. Roof Dribble Carry + Flick + Catch-After-Flick + Ground/Aerial Musty
        horiz_dist = math.sqrt((ball_pos.x - car_pos.x) ** 2 + (ball_pos.y - car_pos.y) ** 2)
        vert_diff = ball_pos.z - car_pos.z
        car_pitch = my_car.physics.rotation.pitch

        if self.post_flick_catch_timer > 0:
            self.post_flick_catch_timer -= 1
            if ball_pos.z > 130.0 and horiz_dist < 600.0:
                preds = predict_ball_multi_horizon(
                    np.array([ball_pos.x, ball_pos.y, ball_pos.z], dtype=np.float32),
                    np.array([ball_vel.x, ball_vel.y, ball_vel.z], dtype=np.float32),
                    horizons=self.HORIZONS,
                )
                catch_spot = Vec3(float(preds[0.5][0]), float(preds[0.5][1]), 17.0)
                controls = SimpleControllerState()
                controls.steer = self._steer_toward(my_car, catch_spot)
                dist_to_spot = car_pos.dist(catch_spot)
                controls.throttle = float(np.clip(dist_to_spot / 220.0, 0.2, 1.0))
                controls.boost = (dist_to_spot > 240.0 and car_vel.length() < 1800.0 and my_car.boost > 10)
                return controls

        if horiz_dist < 165.0 and 65.0 < vert_diff < 195.0 and my_car.has_wheel_contact:
            self.dribble_ticks += 1
            dist_to_opp_goal = abs(ball_pos.y - opp_goal_y)
            if self.dribble_ticks >= 18 and dist_to_opp_goal < 2600.0:
                self.dribble_ticks = 0
                if car_pitch < -0.2 or (my_car.boost > 25 and random.random() < 0.35):
                    self.active_mechanic = "ground_musty"
                    self.mechanic_timer = 11
                else:
                    self.active_mechanic = "hood_flick"
                    self.mechanic_timer = 9
                return self.check_mechanic_trigger(packet, prev_action)

            carry_spot = Vec3(
                ball_pos.x * 0.85,
                ball_pos.y - forward_dir * 25.0,
                car_pos.z,
            )
            controls = SimpleControllerState()
            controls.steer = self._steer_toward(my_car, carry_spot)
            speed_diff = ball_vel.length() - car_vel.length()
            controls.throttle = float(np.clip(0.85 + speed_diff / 400.0, 0.1, 1.0))
            controls.boost = (speed_diff > 140.0 and car_vel.length() < 2000.0 and my_car.boost > 5)
            return controls
        else:
            self.dribble_ticks = max(0, self.dribble_ticks - 1)

        if (
            not my_car.has_wheel_contact
            and car_pos.z > 320.0
            and ball_pos.z > 350.0
            and horiz_dist < 190.0
            and 10.0 < vert_diff < 160.0
            and car_pitch < -0.30
            and self.update_flip_tracking(packet)
        ):
            self.active_mechanic = "aerial_musty"
            self.mechanic_timer = 6
            return self.check_mechanic_trigger(packet, prev_action)

        # 3. Wall Jump / Wall Pop Assist (off side and back walls)
        is_on_wall = my_car.has_wheel_contact and (abs(car_pos.x) > 3600.0 or abs(car_pos.y) > 4700.0)
        if is_on_wall and dist < 850.0:
            ori = Orientation(my_car.physics.rotation)
            wall_out = ori.up
            ball_out_dist = (ball_pos - car_pos).dot(wall_out)
            if ball_out_dist > 50.0 or abs(ball_pos.x) < 3700.0:
                controls = SimpleControllerState()
                controls.throttle = 1.0
                controls.jump = True
                controls.steer = self._steer_toward(my_car, ball_pos)
                controls.pitch = float(np.clip(-my_car.physics.rotation.pitch, -0.8, 0.8))
                controls.boost = (my_car.boost > 10 and dist > 200.0)
                return controls

        # 4. Hybrid Multi-Horizon Ball-Aiming & Attack Guidance
        aim_controls, aim_angle = self.compute_ball_aim_controls(packet)
        neural_throttle = float(prev_action[0])
        neural_steer = float(prev_action[1])

        # Bouncing Ball Contest: if ball is bouncing (z in 190..700) and closing in, jump up to meet it!
        if (
            my_car.has_wheel_contact
            and not is_on_wall
            and 190.0 < ball_pos.z < 700.0
            and dist < 650.0
            and abs(aim_angle) < 0.50
        ):
            controls = SimpleControllerState()
            controls.throttle = 1.0
            controls.steer = aim_controls.steer
            controls.jump = True
            vert_lead = ball_pos.z - car_pos.z
            controls.pitch = float(np.clip(-vert_lead / 320.0, -1.0, -0.30))
            controls.boost = (my_car.boost > 15 and vert_lead > 130.0)
            return controls

        # 4b. 3D In-Flight Aerial Guidance & Intercept Alignment (up to 2000 uu)
        if not my_car.has_wheel_contact and not is_on_wall and ball_pos.z > 260.0 and dist < 2500.0:
            ori = Orientation(my_car.physics.rotation)
            local_target = relative_location(car_pos, ori, target_vec)
            yaw_err = math.atan2(local_target.y, max(1.0, local_target.x))
            pitch_err = math.atan2(local_target.z, max(1.0, local_target.x))

            controls = SimpleControllerState()
            controls.throttle = 1.0
            controls.pitch = float(np.clip(-pitch_err * 2.8, -1.0, 1.0))
            controls.yaw = float(np.clip(yaw_err * 2.8, -1.0, 1.0))
            controls.roll = float(np.clip(-my_car.physics.rotation.roll * 1.5, -0.6, 0.6))

            is_aligned = abs(yaw_err) < 0.35 and abs(pitch_err) < 0.35
            controls.boost = bool(is_aligned and my_car.boost > 0 and dist > 180.0)

            # Dodge strike into ball when at contact point
            if dist < 220.0 and self.update_flip_tracking(packet):
                controls.jump = True
                controls.pitch = -1.0
            return controls

        stuck_near_ball = (neural_throttle <= 0.0 and car_vel.length() < 120.0 and dist < 1200.0)
        if my_car.has_wheel_contact and stuck_near_ball:
            return aim_controls

        # 5. Boost Management: shut off boost if already supersonic (> 2200 uu/s)
        if car_vel.length() >= 2200.0 and prev_action[6] > 0.5:
            controls = SimpleControllerState()
            controls.throttle = 1.0
            controls.steer = float(aim_controls.steer if abs(aim_angle) > 0.15 else prev_action[1])
            controls.pitch = float(prev_action[2])
            controls.yaw = float(prev_action[3])
            controls.roll = float(prev_action[4])
            controls.jump = bool(prev_action[5] > 0.5)
            controls.boost = False
            controls.handbrake = bool(prev_action[7] > 0.5)
            return controls

        return None

    @staticmethod
    def _steer_toward(car, target: Vec3) -> float:
        car_pos = Vec3(car.physics.location)
        ori = Orientation(car.physics.rotation)
        rel = relative_location(car_pos, ori, target)
        angle = math.atan2(rel.y, rel.x)
        return float(np.clip(angle * 4.5, -1.0, 1.0))

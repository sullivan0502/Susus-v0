"""
Unified Susu Observation Builder, 90-Action Lookup Table, Kickoff Sequence,
and Multi-Horizon Ball Physics Predictor (0.1s, 0.5s, 1.0s, 2.0s) for susu_bot (Susu-v0).
"""

import math
from typing import Any, Optional, Tuple
import numpy as np

try:
    from rlbot.agents.base_agent import SimpleControllerState
except ImportError:
    class SimpleControllerState:
        def __init__(
            self,
            steer=0.0,
            throttle=0.0,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            jump=False,
            boost=False,
            handbrake=False,
            use_item=False,
        ):
            self.steer = steer
            self.throttle = throttle
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll
            self.jump = jump
            self.boost = boost
            self.handbrake = handbrake
            self.use_item = use_item

# ==============================================================================
# 1. DISCRETE 90-ACTION LOOKUP TABLE (Susu Compatible)
# ==============================================================================
_LOOKUP_TABLE = None


def make_lookup_table() -> np.ndarray:
    actions = []
    # Ground actions
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
                    actions.append([throttle or boost, steer, 0.0, steer, 0.0, 0.0, float(boost), float(handbrake)])

    # Aerial actions
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = float(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        actions.append([float(boost), float(yaw), float(pitch), float(yaw), float(roll), float(jump), float(boost), handbrake])

    return np.array(actions, dtype=np.float32)


def get_lookup_table() -> np.ndarray:
    global _LOOKUP_TABLE
    if _LOOKUP_TABLE is None:
        _LOOKUP_TABLE = make_lookup_table()
    return _LOOKUP_TABLE


def parse_action_index(action_index: int) -> np.ndarray:
    table = get_lookup_table()
    idx = int(action_index) % len(table)
    return table[idx]


# ==============================================================================
# 2. FIELD CONSTANTS, BOOST PADS & SPEEDFLIP KICKOFF SEQUENCE
# ==============================================================================
KEY_BOOST_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 29, 30, 33],
    dtype=np.int32,
)

BOOST_LOCATIONS = np.array([
    [0.0, -4240.0, 70.0],
    [-1792.0, -4184.0, 70.0],
    [1792.0, -4184.0, 70.0],
    [-3072.0, -4096.0, 73.0],
    [3072.0, -4096.0, 73.0],
    [-940.0, -3308.0, 70.0],
    [940.0, -3308.0, 70.0],
    [0.0, -2816.0, 70.0],
    [-3584.0, -2484.0, 70.0],
    [3584.0, -2484.0, 70.0],
    [-1788.0, -2300.0, 70.0],
    [1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0],
    [0.0, -1024.0, 70.0],
    [2048.0, -1036.0, 70.0],
    [-3584.0, 0.0, 73.0],
    [-1024.0, 0.0, 70.0],
    [1024.0, 0.0, 70.0],
    [3584.0, 0.0, 73.0],
    [-2048.0, 1036.0, 70.0],
    [0.0, 1024.0, 70.0],
    [2048.0, 1036.0, 70.0],
    [-1788.0, 2300.0, 70.0],
    [1788.0, 2300.0, 70.0],
    [-3584.0, 2484.0, 70.0],
    [3584.0, 2484.0, 70.0],
    [0.0, 2816.0, 70.0],
    [-940.0, 3310.0, 70.0],
    [940.0, 3308.0, 70.0],
    [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0],
    [-1792.0, 4184.0, 70.0],
    [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
], dtype=np.float32)

KICKOFF_CONTROLS = (
    11 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 4 * 4 * [SimpleControllerState(throttle=1, boost=True, steer=-1)]
    + 2 * 4 * [SimpleControllerState(throttle=1, jump=True, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, boost=True)]
    + 1 * 4 * [SimpleControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
    + 13 * 4 * [SimpleControllerState(throttle=1, pitch=1, boost=True)]
    + 10 * 4 * [SimpleControllerState(throttle=1, roll=1, pitch=0.5)]
)

KICKOFF_NUMPY = np.array(
    [
        [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, scs.jump, scs.boost, scs.handbrake]
        for scs in KICKOFF_CONTROLS
    ],
    dtype=np.float32,
)

FIELD_NORM = np.array([4096.0, 5120.0, 2048.0], dtype=np.float32)
GOAL_Y = 5120.0
PREDICTION_HORIZONS = (0.05, 0.1, 0.15, 0.25, 0.5, 0.6, 1.0, 2.0)
LEGACY_HORIZONS = (0.1, 0.5, 1.0, 2.0)
SUB_HORIZONS = (0.05, 0.15, 0.25, 0.6)
PREDICTION_HORIZONS_20 = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50,
)
EXTRA_HORIZONS_12 = (
    0.20, 0.30, 0.35, 0.40, 0.45, 0.70, 0.80, 0.90, 1.25, 1.50, 1.75, 2.50,
)
BALL_RADIUS = 93.15
GRAVITY_Z = -650.0


# ==============================================================================
# 3. MULTI-HORIZON BALL TRAJECTORY PREDICTION (20 Horizons: 0.05s to 2.50s)
# ==============================================================================
_DEFAULT_TARGET_STEPS_20 = {
    int(round(h / 0.05)): h for h in PREDICTION_HORIZONS_20
}
_DEFAULT_TARGET_STEPS = _DEFAULT_TARGET_STEPS_20


def predict_ball_multi_horizon(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    horizons: Tuple[float, ...] = PREDICTION_HORIZONS_20,
) -> dict:
    """
    Simulates Rocket League ball physics (gravity, air drag, floor/ceiling/wall/backboard bounces)
    and returns predicted 3D ball positions at each horizon (default: 0.1s, 0.5s, 1.0s, 2.0s).
    Optimized with pure scalar arithmetic to eliminate per-step numpy overhead in tight loops.
    """
    px = float(ball_pos[0])
    py = float(ball_pos[1])
    pz = float(ball_pos[2])
    vx = float(ball_vel[0])
    vy = float(ball_vel[1])
    vz = float(ball_vel[2])

    dt = 0.05
    drag = 1.0 - 0.0305 * dt  # 0.998475
    grav_dt = GRAVITY_Z * dt  # -32.5
    ceil_z = 2044.0 - BALL_RADIUS
    wall_x = 4096.0 - BALL_RADIUS
    wall_y = 5120.0 - BALL_RADIUS

    if horizons == PREDICTION_HORIZONS:
        n_steps = 40
        target_steps = _DEFAULT_TARGET_STEPS
    else:
        max_h = max(horizons) if horizons else 2.0
        n_steps = int(round(max_h / dt))
        target_steps = {int(round(h / dt)): h for h in horizons}

    results = {}
    if 0 in target_steps:
        results[target_steps[0]] = np.array([px, py, pz], dtype=np.float32)

    for step in range(1, n_steps + 1):
        if pz > 95.15 or (vz > 25.0 or vz < -25.0):
            vz += grav_dt
        else:
            pz = BALL_RADIUS
            vz = 0.0

        vx *= drag
        vy *= drag
        vz *= drag
        px += vx * dt
        py += vy * dt
        pz += vz * dt

        if pz < BALL_RADIUS:
            pz = BALL_RADIUS
            if vz < 0.0:
                vz = -vz * 0.60
                vx *= 0.985
                vy *= 0.985
                if -35.0 < vz < 35.0:
                    vz = 0.0

        if pz > ceil_z:
            pz = ceil_z
            if vz > 0.0:
                vz = -vz * 0.60

        if px > wall_x:
            px = wall_x
            vx = -vx * 0.60
        elif px < -wall_x:
            px = -wall_x
            vx = -vx * 0.60

        if py > wall_y:
            if not (-850.0 < px < 850.0 and pz < 600.0):
                py = wall_y
                vy = -vy * 0.60
        elif py < -wall_y:
            if not (-850.0 < px < 850.0 and pz < 600.0):
                py = -wall_y
                vy = -vy * 0.60

        if step in target_steps:
            results[target_steps[step]] = np.array([px, py, pz], dtype=np.float32)

    for h in horizons:
        if h not in results:
            results[h] = np.array([px, py, pz], dtype=np.float32)
    return results


def select_optimal_ball_intercept(
    car_pos: np.ndarray,
    car_vel: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    horizons: Tuple[float, ...] = PREDICTION_HORIZONS,
) -> Tuple[np.ndarray, float, dict]:
    """
    Uses multi-horizon ball predictions (0.1s, 0.5s, 1.0s, 2.0s) to select the earliest
    horizon where the car can intercept the ball, returning (intercept_pos, chosen_horizon, preds_dict).
    """
    preds = predict_ball_multi_horizon(ball_pos, ball_vel, horizons=horizons)
    car_speed = float(np.linalg.norm(car_vel))
    effective_speed = max(900.0, min(2300.0, car_speed + 550.0))

    chosen_h = horizons[0]
    chosen_pos = preds[chosen_h]

    prev_t = 0.0
    prev_p = np.asarray(ball_pos, dtype=np.float32)

    for h in horizons:
        p = preds[h]
        dist_to_p = float(np.linalg.norm(p - car_pos))
        height_extra = max(0.0, float(p[2]) - 200.0) * 1.35
        eta = (dist_to_p + height_extra) / effective_speed
        if eta <= h * 1.15:
            span = max(1e-4, h - prev_t)
            alpha = float(np.clip((eta - prev_t) / span, 0.0, 1.0))
            chosen_pos = prev_p + alpha * (p - prev_p)
            chosen_h = h
            return chosen_pos.astype(np.float32), chosen_h, preds
        prev_t = h
        prev_p = p

    chosen_h = horizons[-1]
    return preds[chosen_h].astype(np.float32), chosen_h, preds


def compute_ball_aim_action(
    car_pos: np.ndarray,
    car_vel: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    on_ground: bool = True,
    boost_amount: float = 33.3,
    target_goal_y: float = GOAL_Y,
) -> Tuple[np.ndarray, float]:
    car_pos = np.asarray(car_pos, dtype=np.float32)
    car_vel = np.asarray(car_vel, dtype=np.float32)
    forward = np.asarray(forward, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    ball_pos = np.asarray(ball_pos, dtype=np.float32)
    ball_vel = np.asarray(ball_vel, dtype=np.float32)

    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm > 1e-6:
        right = right / right_norm
    else:
        right = np.array([forward[1], -forward[0], 0.0], dtype=np.float32)

    dist = float(np.linalg.norm(ball_pos - car_pos))
    car_speed = float(np.linalg.norm(car_vel))

    predicted_ball, chosen_h, _ = select_optimal_ball_intercept(
        car_pos=car_pos,
        car_vel=car_vel,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        horizons=PREDICTION_HORIZONS_20,
    )

    # Goal targeting: strike behind the ball aligned with the opponent goal
    goal_pos = np.array([0.0, target_goal_y, 100.0], dtype=np.float32)
    ball_to_goal = goal_pos - predicted_ball
    b2g_dist = float(np.linalg.norm(ball_to_goal))
    if b2g_dist > 1e-6:
        shot_dir = ball_to_goal / b2g_dist
        # Strike offset: 68 uu behind the ball on the goal axis
        # Smoothly interpolates from distance so car curves in behind the ball
        offset_len = min(68.0, max(22.0, dist * 0.16))
        aim_target = predicted_ball - shot_dir * offset_len
    else:
        aim_target = predicted_ball

    rel = aim_target - car_pos
    local_x = float(np.dot(rel, forward))
    local_y = float(np.dot(rel, right))
    local_z = float(np.dot(rel, up))

    aim_angle = math.atan2(local_y, local_x)
    steer = float(np.clip(aim_angle * 4.0, -1.0, 1.0))
    throttle = 1.0

    handbrake = float(on_ground and abs(aim_angle) > 1.35 and car_speed > 450.0)
    use_boost = float(abs(aim_angle) < 0.40 and car_speed < 2200.0 and boost_amount > 0.0 and not handbrake)

    pitch = 0.0
    yaw = steer
    roll = 0.0
    jump = 0.0

    # Wall detection: car is on side or back wall if z > 140 and car up vector is perpendicular to ground
    is_on_wall = on_ground and (abs(car_pos[0]) > 3600.0 or abs(car_pos[1]) > 4700.0 or abs(up[2]) < 0.82)

    if not on_ground:
        elev_angle = math.atan2(local_z, max(1.0, abs(local_x)))
        pitch = float(np.clip(elev_angle * 3.0, -1.0, 1.0))
        yaw = float(np.clip(aim_angle * 3.0, -1.0, 1.0))
        if local_z > 25.0 and abs(elev_angle) < 0.50 and abs(aim_angle) < 0.50:
            use_boost = 1.0
    elif is_on_wall:
        # Car is driving on the wall!
        # If ball is floating off the wall into the arena:
        dist_to_pred = float(np.linalg.norm(predicted_ball - car_pos))
        ball_off_wall = float(np.dot(predicted_ball - car_pos, up))
        if dist_to_pred < 850.0 and (ball_off_wall > 60.0 or abs(predicted_ball[0]) < 3700.0):
            # WALL POP / WALL JUMP: Jump off the wall to intercept floating ball!
            jump = 1.0
            elev_angle = math.atan2(local_z, max(1.0, abs(local_x)))
            pitch = float(np.clip(elev_angle * 2.5, -1.0, 1.0))
            yaw = float(np.clip(aim_angle * 2.5, -1.0, 1.0))
            if ball_off_wall > 150.0:
                use_boost = 1.0
    else:
        # Car is on the flat ground
        is_ball_bouncing_or_high = (predicted_ball[2] > 180.0) or (ball_pos[2] > 180.0)
        vert_lead = max(0.0, float(predicted_ball[2] - car_pos[2]))

        if is_ball_bouncing_or_high and dist < 750.0 and abs(aim_angle) < 0.55:
            # FAST AERIAL / BOUNCING BALL INTERCEPT:
            # Ball is bouncing or in the air! Do NOT wait on the turf! Jump up!
            jump = 1.0
            pitch = float(np.clip(-vert_lead / 320.0, -1.0, -0.30))
            if vert_lead > 130.0 and abs(aim_angle) < 0.40:
                use_boost = 1.0
        elif dist < 170.0 and abs(aim_angle) < 0.22 and car_speed > 1650.0 and ball_pos[2] < 140.0:
            # Supersonic power shot only when already fast, close, and cleanly lined up with goal
            jump = 1.0
            pitch = -1.0
        else:
            # Clean ground drive / push / hook shot: NO blind jump! Keep wheels on ground for precision!
            jump = 0.0
            pitch = 0.0

    action = np.array([throttle, steer, pitch, yaw, roll, jump, use_boost, handbrake], dtype=np.float32)
    return action, aim_angle


def compute_obs_ball_aim_logits_bias(obs: np.ndarray, lookup_table: np.ndarray) -> np.ndarray:
    if obs.shape[-1] < 34:
        return np.zeros(len(lookup_table), dtype=np.float32)

    car_pos = obs[0:3] * FIELD_NORM
    car_vel = obs[3:6] * 2300.0
    forward = obs[6:9]
    up = obs[9:12]
    on_ground = bool(obs[15] > 0.5)
    boost_fraction = float(obs[17])
    is_supersonic = bool(obs[18] > 0.5)

    ball_pos = obs[25:28] * FIELD_NORM
    ball_vel = obs[28:31] * 2300.0

    ideal_action, _ = compute_ball_aim_action(
        car_pos=car_pos,
        car_vel=car_vel,
        forward=forward,
        up=up,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        on_ground=on_ground,
        boost_amount=boost_fraction * 100.0,
        target_goal_y=GOAL_Y,
    )

    throttles = lookup_table[:, 0]
    steers = lookup_table[:, 1]
    jumps = lookup_table[:, 5]
    boosts = lookup_table[:, 6]
    handbrakes = lookup_table[:, 7]

    throttle_score = throttles * ideal_action[0]
    steer_diff = np.abs(steers - ideal_action[1])
    steer_score = 1.0 - steer_diff

    if is_supersonic:
        boost_score = -0.5 * boosts
    else:
        boost_score = 0.35 * boosts * ideal_action[6]

    hb_score = 0.25 * (1.0 - np.abs(handbrakes - ideal_action[7]))

    dist = float(np.linalg.norm(ball_pos - car_pos))
    if ideal_action[5] > 0.5:
        # Aerial takeoff, bounce jump, or wall pop is desired!
        jump_score = 0.50 * jumps
    elif on_ground and dist > 300.0:
        jump_score = -0.35 * jumps
    else:
        # On ground: reward clean ground contact unless specified
        jump_score = 0.25 * (1.0 - np.abs(jumps - ideal_action[5]))

    total_bias = 0.45 * throttle_score + 0.65 * steer_score + boost_score + hb_score + jump_score
    return total_bias.astype(np.float32)


# ==============================================================================
# 4. 142-FLOAT OBSERVATION BUILDER (SusuObsBuilder)
# ==============================================================================
class SusuObsBuilder:
    """
    Susu-v0 observation builder:
    - 19 self car
    - 15 ball
    - 12 opponent
    - 34 (22 boost pads + 12 ball preds at 0.1s, 0.5s, 1.0s, 2.0s)
    - 14 mechanics perception features
    - 12 fine-grained ball trajectory preds at 0.05s, 0.15s, 0.25s, 0.60s
    - 36 high-resolution trajectory preds (12 extra horizons up to 2.50s)
    Total = 142 floats (indices 0..105 preserve 100% compatibility with 106-float checkpoints).
    """

    OBS_SIZE = 142

    def __init__(self):
        self._prev_action = np.zeros(8, dtype=np.float32)

    def reset(self, initial_state: Any = None):
        self._prev_action = np.zeros(8, dtype=np.float32)

    def build_obs(self, player: Any, state: Any, previous_action: Optional[np.ndarray] = None) -> np.ndarray:
        if previous_action is not None:
            self._prev_action = np.asarray(previous_action, dtype=np.float32)

        is_orange = (player.team_num == 1)

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

        # 4. Boost pad states (22) + Multi-horizon ball predictions at 0.1s, 0.5s, 1.0s, 2.0s (12) = 34
        if hasattr(state, "boost_pads") and len(state.boost_pads) >= 34:
            pads = np.asarray(state.boost_pads[:34], dtype=np.float32)
            if is_orange:
                pads = pads[::-1]
            key_pads = (pads[KEY_BOOST_INDICES] > 0.5).astype(np.float32)
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
        offset = rel_ball_pos
        horiz_dist = float(np.linalg.norm(offset[:2]))
        vert_diff = float(offset[2])

        ball_on_hood = float(horiz_dist < 165.0 and 60.0 < vert_diff < 200.0)
        musty_readiness = float(forward[2] < -0.25 and horiz_dist < 220.0 and vert_diff > 30.0)

        wheel_dir = -up
        dot_wheels = float(np.dot(wheel_dir, offset / (np.linalg.norm(offset) + 1e-6)))
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

        dist_to_wall = float(min(4096.0 - abs(car_pos[0]), 4096.0 - abs(ball_pos[0])))
        wall_pinch_read = float(dist_to_wall < 350.0 and horiz_dist < 250.0)
        ground_pinch_read = float(car_vel[2] < -800.0 and ball_pos[2] < 220.0 and horiz_dist < 200.0)

        own_goal = np.array([0.0, -GOAL_Y, 0.0], dtype=np.float32)
        to_own_goal = own_goal - car_pos
        to_own_goal_dir = to_own_goal / (np.linalg.norm(to_own_goal) + 1e-6)
        defending_alignment = float(np.dot(forward, to_own_goal_dir))

        dist_ball_to_net = float(np.linalg.norm(ball_pos - own_goal))
        base_threat = float(np.clip(1.0 - (dist_ball_to_net / 6000.0), 0.0, 1.0))

        # Dynamic Opponent Attention Gating (Sullivan's Vision Architecture):
        # 1. If opponent is airborne (aerial challenge) -> Attention active across whole pitch!
        # 2. If opponent is on the ground -> Attention active only when within 1800 uu of ball.
        # 3. If opponent is far away on the turf -> 0 threat, focus 100% on precision open-net shot.
        if opponents:
            opp_dist_to_ball = float(np.linalg.norm(opp_pos - ball_pos))
            opp_airborne = float(not opp.on_ground or opp_pos[2] > 220.0)
            if opp_airborne > 0.5:
                opp_threat_gate = 1.0
            elif opp_dist_to_ball < 1800.0:
                opp_threat_gate = float(np.clip(1.0 - (opp_dist_to_ball - 400.0) / 1400.0, 0.0, 1.0))
            else:
                opp_threat_gate = 0.0
            threat_level = float(np.clip(0.5 * base_threat + 0.5 * opp_threat_gate, 0.0, 1.0))
        else:
            threat_level = base_threat

        boost_wasting = float(car_speed > 2200.0 and self._prev_action[6] > 0.5)
        speedflip_curl = float(abs(self._prev_action[2]) > 0.5 and abs(self._prev_action[4]) > 0.5)
        ceiling_prox = float(np.clip((2048.0 - car_pos[2]) / 2048.0, 0.0, 1.0))
        wall_prox = float(np.clip((4096.0 - abs(car_pos[0])) / 4096.0, 0.0, 1.0))
        kickoff_active = float(abs(ball_pos[0]) < 10.0 and abs(ball_pos[1]) < 10.0 and np.linalg.norm(ball_vel) < 50.0)

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

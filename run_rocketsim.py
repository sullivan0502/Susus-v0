#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VISUALISEUR & SIMULATEUR ROCKETSIM 3D — SUSU_BOT
===============================================
Lance susu_bot directement dans le moteur physique C++ RocketSim avec affichage 3D
en temps réel (120 FPS) via la fenêtre graphique RLViser.

Modes disponibles via le menu interactif ou en ligne de commande :
  [1] Match 1v1 Normal     : susu_bot contre un adversaire en conditions réelles
  [2] Démo Aérienne        : Balles hautes en cloche, air dribbles, sorties de mur
  [3] Démo Flick           : Portage capot, déclenchement de flick puissant
  [4] Démo Tir Précis      : Frappes puissantes et cadrées sur balles rebondissantes
  [5] Démo Déplacement     : Supersonique, rotations rapides, rétablissement 4 roues
  [6] Test Kick-off        : Démonstration en boucle du Speedflip Nexto (5 positions)

Usage:
  python run_rocketsim.py                 (Ouvre le menu interactif)
  python run_rocketsim.py --mode match    (Lance directement le match 1v1)
  python run_rocketsim.py --mode aerial   (Lance l'entraînement aérien)
  python run_rocketsim.py --no_vis        (Mode sans fenêtre, benchmark pur)
"""

import argparse
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# Fix Windows console UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import RocketSim as rsim
try:
    import rlviser_py as rlviser
    RLVISER_AVAILABLE = True
except ImportError:
    RLVISER_AVAILABLE = False

from bot import SusuBot
from nexto_obs import BOOST_LOCATIONS
from rlbot.utils.structures.game_data_struct import GameTickPacket, FieldInfoPacket

# Constantes du terrain Rocket League
BALL_RADIUS = 93.15
GOAL_Y = 5120.0
TICK_RATE = 120.0
DT = 1.0 / TICK_RATE

# 5 positions de coup d'envoi officielles (Blue team)
KICKOFF_SPAWNS_BLUE = [
    ("Diag Droit", rsim.Vec(2048, -2560, 17), np.arctan2(2560, -2048)),
    ("Diag Gauche", rsim.Vec(-2048, -2560, 17), np.arctan2(2560, 2048)),
    ("Off-Center Droit", rsim.Vec(256, -3840, 17), np.pi / 2),
    ("Off-Center Gauche", rsim.Vec(-256, -3840, 17), np.pi / 2),
    ("Centre Droit", rsim.Vec(0, -4608, 17), np.pi / 2),
]

KICKOFF_SPAWNS_ORANGE = [
    ("Diag Gauche", rsim.Vec(2048, 2560, 17), np.arctan2(-2560, -2048)),
    ("Diag Droit", rsim.Vec(-2048, 2560, 17), np.arctan2(-2560, 2048)),
    ("Off-Center Gauche", rsim.Vec(256, 3840, 17), -np.pi / 2),
    ("Off-Center Droit", rsim.Vec(-256, 3840, 17), -np.pi / 2),
    ("Centre Droit", rsim.Vec(0, 4608, 17), -np.pi / 2),
]


class RocketSimBridge:
    """Convertit l'état physique RocketSim en GameTickPacket pour l'agent RLBot."""

    def __init__(self):
        self.packet = GameTickPacket()
        self.field_info = FieldInfoPacket()
        self.field_info.num_boosts = len(BOOST_LOCATIONS)
        for i, loc in enumerate(BOOST_LOCATIONS):
            self.field_info.boost_pads[i].location.x = float(loc[0])
            self.field_info.boost_pads[i].location.y = float(loc[1])
            self.field_info.boost_pads[i].location.z = float(loc[2])
            self.field_info.boost_pads[i].is_full_boost = bool(loc[2] > 72)

    def to_packet(self, arena: rsim.Arena, is_round_active: bool = True) -> GameTickPacket:
        p = self.packet
        p.game_info.seconds_elapsed = float(arena.tick_count * DT)
        p.game_info.is_round_active = bool(is_round_active)

        b = arena.ball.get_state()
        ball_dist_center = math.sqrt(b.pos.x ** 2 + b.pos.y ** 2)
        ball_speed = math.sqrt(b.vel.x ** 2 + b.vel.y ** 2 + b.vel.z ** 2)
        p.game_info.is_kickoff_pause = bool(ball_dist_center < 35.0 and ball_speed < 120.0)

        p.game_ball.physics.location.x = float(b.pos.x)
        p.game_ball.physics.location.y = float(b.pos.y)
        p.game_ball.physics.location.z = float(b.pos.z)
        p.game_ball.physics.velocity.x = float(b.vel.x)
        p.game_ball.physics.velocity.y = float(b.vel.y)
        p.game_ball.physics.velocity.z = float(b.vel.z)

        cars = arena.get_cars()
        p.num_cars = len(cars)
        for i, c in enumerate(cars):
            cs = c.get_state()
            p.game_cars[i].physics.location.x = float(cs.pos.x)
            p.game_cars[i].physics.location.y = float(cs.pos.y)
            p.game_cars[i].physics.location.z = float(cs.pos.z)
            p.game_cars[i].physics.velocity.x = float(cs.vel.x)
            p.game_cars[i].physics.velocity.y = float(cs.vel.y)
            p.game_cars[i].physics.velocity.z = float(cs.vel.z)

            ang = cs.rot_mat.as_angle()
            p.game_cars[i].physics.rotation.pitch = float(ang.pitch)
            p.game_cars[i].physics.rotation.yaw = float(ang.yaw)
            p.game_cars[i].physics.rotation.roll = float(ang.roll)
            p.game_cars[i].boost = int(round(cs.boost))
            p.game_cars[i].has_wheel_contact = bool(cs.is_on_ground)
            p.game_cars[i].team = int(c.team)

        return p


class RocketSimVisualizer:
    def __init__(
        self,
        mode: str = "match",
        enable_vis: bool = True,
        speed_mult: float = 1.0,
    ):
        self.mode = mode.lower().strip()
        self.enable_vis = enable_vis and RLVISER_AVAILABLE
        self.speed_mult = max(0.1, min(10.0, speed_mult))
        self.running = True

        # Initialisation du moteur RocketSim
        meshes_dir = _ROOT / "collision_meshes"
        try:
            rsim.init(str(meshes_dir) if meshes_dir.exists() else "./collision_meshes")
        except RuntimeError as re:
            if "already inited" not in str(re).lower():
                raise re

        self.arena = rsim.Arena(rsim.GameMode.SOCCAR)
        self.bridge = RocketSimBridge()
        self.pads = self.arena.get_boost_pads()

        # Agents
        self.car_blue = self.arena.add_car(rsim.Team.BLUE)
        self.bot_blue = SusuBot("susu_bot", team=0, index=0)
        self.bot_blue.get_field_info = lambda: self.bridge.field_info
        self.bot_blue.initialize_agent()

        # Adversaire / Gardien Orange pour les matchs et entraînements réalistes (Flick & Tir)
        self.car_orange = None
        self.bot_orange = None
        if self.mode in ("match", "flick", "shooting"):
            self.car_orange = self.arena.add_car(rsim.Team.ORANGE)
            if self.mode == "match":
                try:
                    import importlib.util
                    naze_bot_file = _ROOT.parent / "naze_bot" / "bot.py"
                    if not naze_bot_file.exists():
                        naze_bot_file = _ROOT.parent / "naze bot2" / "bot.py"
                    if naze_bot_file.exists():
                        spec = importlib.util.spec_from_file_location("naze_bot_module", str(naze_bot_file))
                        naze_mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(naze_mod)
                        self.bot_orange = naze_mod.NazeBot("naze_bot", team=1, index=1)
                except Exception:
                    pass

            if self.bot_orange is None:
                self.bot_orange = SusuBot("opponent", team=1, index=1)

            self.bot_orange.get_field_info = lambda: self.bridge.field_info
            self.bot_orange.initialize_agent()

        # Métriques & Statistiques avancées par mode
        self.score_blue = 0
        self.score_orange = 0
        self.scenario_timer = 0
        self.scenario_index = 0
        self.scenario_name = ""
        self.aerial_touches = 0
        self.aerial_fails = 0
        self.flick_successes = 0
        self.flick_fails = 0
        self.flick_floor_ticks = 0
        self.power_shots_count = 0
        self.max_shot_kmh = 0.0
        self.recoveries_count = 0
        self.touches_count = 0

        # Lancement de la fenêtre RLViser
        if self.enable_vis:
            try:
                os.chdir(str(_ROOT))
                rlviser.set_boost_pad_locations(BOOST_LOCATIONS)
                rlviser.launch()
                print("🎮 Fenêtre 3D RLViser lancée avec succès (120 FPS) !")
            except Exception as e:
                print(f"⚠️ Avertissement RLViser : {e}. Poursuite en mode console.")
                self.enable_vis = False

        self._reset_scenario()

    def _reset_scenario(self):
        """Configure des situations de jeu riches, variées et ultra-réalistes selon le mode."""
        self.scenario_timer = 0
        self.flick_floor_ticks = 0

        if self.mode == "match":
            # 1v1 Normal avec position de coup d'envoi aléatoire officielle
            idx = random.randint(0, 4)
            name, pos_b, yaw_b = KICKOFF_SPAWNS_BLUE[idx]
            _, pos_o, yaw_o = KICKOFF_SPAWNS_ORANGE[idx]
            self.scenario_name = f"Kickoff {name}"

            sb = rsim.CarState()
            sb.pos = pos_b
            sb.rot_mat = rsim.Angle(yaw_b, 0, 0).as_rot_mat()
            sb.boost = 33.3
            sb.is_on_ground = True
            self.car_blue.set_state(sb)

            so = rsim.CarState()
            so.pos = pos_o
            so.rot_mat = rsim.Angle(yaw_o, 0, 0).as_rot_mat()
            so.boost = 33.3
            so.is_on_ground = True
            self.car_orange.set_state(so)

            ball = rsim.BallState()
            ball.pos = rsim.Vec(0, 0, BALL_RADIUS)
            ball.vel = rsim.Vec(0, 0, 0)
            self.arena.ball.set_state(ball)

        elif self.mode == "aerial":
            # AÉRIENNE PURE : balles hautes flottantes/ascendantes, STRICTEMENT sans rebond au sol !
            archetype = random.choice(["high_cross", "backboard_rebound", "climbing_lob", "air_dribble"])

            if archetype == "high_cross":
                is_left = random.random() < 0.5
                sign_x = -1.0 if is_left else 1.0
                self.scenario_name = f"Aérienne : Centre Aérien ({'Flanc Gauche' if is_left else 'Flanc Droit'})"
                bx = sign_x * random.uniform(2500, 3100)
                by = random.uniform(800, 2600)
                bz = random.uniform(950, 1500)
                bvx = -sign_x * random.uniform(900, 1450)
                bvy = random.uniform(-100, 250)
                bvz = random.uniform(180, 420)  # Flottant/ascendant

                car_x = random.uniform(-600, 600)
                car_y = random.uniform(-1600, 200)
                car_yaw = math.atan2(by - car_y, (bx + bvx * 1.0) - car_x)
                car_speed = random.uniform(800, 1300)

                sb = rsim.CarState()
                sb.pos = rsim.Vec(car_x, car_y, 17)
                sb.vel = rsim.Vec(car_speed * math.cos(car_yaw), car_speed * math.sin(car_yaw), 0)
                sb.rot_mat = rsim.Angle(car_yaw, 0, 0).as_rot_mat()
                sb.boost = 100.0
                sb.is_on_ground = True
                self.car_blue.set_state(sb)

            elif archetype == "backboard_rebound":
                self.scenario_name = "Aérienne : Rebond Haut Backboard Orange"
                bx = random.uniform(-1200, 1200)
                by = random.uniform(4500, 4950)
                bz = random.uniform(1150, 1750)
                bvx = random.uniform(-300, 300)
                bvy = random.uniform(-650, -1100)  # Rebondit en sortant du mur vers le milieu
                bvz = random.uniform(150, 400)

                car_x = random.uniform(-800, 800)
                car_y = random.uniform(-200, 1400)
                car_yaw = math.atan2(by - car_y, bx - car_x)
                car_speed = random.uniform(850, 1400)

                sb = rsim.CarState()
                sb.pos = rsim.Vec(car_x, car_y, 17)
                sb.vel = rsim.Vec(car_speed * math.cos(car_yaw), car_speed * math.sin(car_yaw), 0)
                sb.rot_mat = rsim.Angle(car_yaw, 0, 0).as_rot_mat()
                sb.boost = 100.0
                sb.is_on_ground = True
                self.car_blue.set_state(sb)

            elif archetype == "climbing_lob":
                self.scenario_name = "Aérienne : Cloche Montante Fast-Aerial"
                bx = random.uniform(-900, 900)
                by = random.uniform(0, 1800)
                bz = random.uniform(750, 1300)
                bvx = random.uniform(-250, 250)
                bvy = random.uniform(500, 1000)
                bvz = random.uniform(350, 700)  # Ascendant rapide !

                car_x = random.uniform(-700, 700)
                car_y = random.uniform(-2400, -1400)
                car_yaw = math.atan2(by - car_y, bx - car_x)
                car_speed = random.uniform(800, 1400)

                sb = rsim.CarState()
                sb.pos = rsim.Vec(car_x, car_y, 17)
                sb.vel = rsim.Vec(car_speed * math.cos(car_yaw), car_speed * math.sin(car_yaw), 0)
                sb.rot_mat = rsim.Angle(car_yaw, 0, 0).as_rot_mat()
                sb.boost = 100.0
                sb.is_on_ground = True
                self.car_blue.set_state(sb)

            else:
                self.scenario_name = "Aérienne : Prise d'Air & Air Dribble"
                car_x = random.uniform(-1000, 1000)
                car_y = random.uniform(-800, 800)
                car_z = random.uniform(650, 1100)
                car_yaw = math.atan2(5120 - car_y, -car_x * 0.4)
                car_pitch = random.uniform(0.35, 0.65)
                car_speed = random.uniform(800, 1200)

                fwd_x = math.cos(car_pitch) * math.cos(car_yaw)
                fwd_y = math.cos(car_pitch) * math.sin(car_yaw)
                fwd_z = math.sin(car_pitch)

                sb = rsim.CarState()
                sb.pos = rsim.Vec(car_x, car_y, car_z)
                sb.vel = rsim.Vec(fwd_x * car_speed, fwd_y * car_speed, fwd_z * car_speed + 150)
                sb.rot_mat = rsim.Angle(car_yaw, car_pitch, 0).as_rot_mat()
                sb.boost = 100.0
                sb.is_on_ground = False
                self.car_blue.set_state(sb)

                bx = car_x + fwd_x * 160.0
                by = car_y + fwd_y * 160.0
                bz = car_z + fwd_z * 160.0 + 50.0
                bvx = fwd_x * (car_speed + 50.0)
                bvy = fwd_y * (car_speed + 50.0)
                bvz = fwd_z * car_speed + 220.0

            ball = rsim.BallState()
            ball.pos = rsim.Vec(bx, by, bz)
            ball.vel = rsim.Vec(bvx, bvy, bvz)
            self.arena.ball.set_state(ball)

        elif self.mode == "flick":
            # FLICK AVANCÉ : Balle posée sur le toit avec angles d'attaque variés et gardien en face !
            approach = random.choice(["center", "left_wing", "right_wing"])
            if approach == "center":
                self.scenario_name = "Flick : Dribble Plein Axe -> Pop de Flick"
                car_x = random.uniform(-350, 350)
                car_y = random.uniform(-3000, -1800)
                yaw = math.pi / 2 + random.uniform(-0.12, 0.12)
            elif approach == "left_wing":
                self.scenario_name = "Flick : Débordement Aile Gauche -> Flick Cadré"
                car_x = random.uniform(-1600, -800)
                car_y = random.uniform(-2800, -1600)
                yaw = math.atan2(4900 - car_y, -car_x * 0.7)
            else:
                self.scenario_name = "Flick : Débordement Aile Droite -> Flick Cadré"
                car_x = random.uniform(800, 1600)
                car_y = random.uniform(-2800, -1600)
                yaw = math.atan2(4900 - car_y, -car_x * 0.7)

            car_speed = random.uniform(850, 1300)
            vx = car_speed * math.cos(yaw)
            vy = car_speed * math.sin(yaw)

            sb = rsim.CarState()
            sb.pos = rsim.Vec(car_x, car_y, 17)
            sb.vel = rsim.Vec(vx, vy, 0)
            sb.rot_mat = rsim.Angle(yaw, 0, 0).as_rot_mat()
            sb.boost = 75.0
            sb.is_on_ground = True
            self.car_blue.set_state(sb)

            # Balle équilibrée sur le toit (17 + 36 + BALL_RADIUS)
            ball = rsim.BallState()
            ball.pos = rsim.Vec(car_x + math.cos(yaw) * 15.0, car_y + math.sin(yaw) * 15.0, 17 + 36.0 + BALL_RADIUS)
            ball.vel = rsim.Vec(vx, vy, 0)
            self.arena.ball.set_state(ball)

            # Gardien Orange dans les cages
            if self.car_orange is not None:
                so = rsim.CarState()
                so.pos = rsim.Vec(random.uniform(-500, 500), random.uniform(4900, 5080), 17)
                so.vel = rsim.Vec(random.uniform(-350, 350), 0, 0)
                so.rot_mat = rsim.Angle(-math.pi / 2, 0, 0).as_rot_mat()
                so.boost = 100.0
                so.is_on_ground = True
                self.car_orange.set_state(so)

        elif self.mode == "shooting":
            # TIR PRÉCIS : Demi-volées sur rebond au sol, reprises de centre, frappes puissantes > 90 km/h !
            archetype = random.choice(["half_volley", "corner_cross", "side_rebound"])
            if archetype == "half_volley":
                self.scenario_name = "Tir Précis : Demi-Volée Puissante au Rebond"
                bx = random.uniform(-1000, 1000)
                by = random.uniform(1000, 2400)
                bz = random.uniform(700, 1200)
                bvx = random.uniform(-200, 200)
                bvy = random.uniform(-200, 200)
                bvz = random.uniform(-600, -350)  # Tombe rapidement pour frapper au rebond

                dist_back = random.uniform(1500, 2200)
                car_x = bx + random.uniform(-300, 300)
                car_y = by - dist_back
                yaw = math.atan2(by - car_y, bx - car_x)
                speed = random.uniform(1200, 1700)
                vx = speed * math.cos(yaw)
                vy = speed * math.sin(yaw)

            elif archetype == "corner_cross":
                is_left = random.random() < 0.5
                sign_x = -1.0 if is_left else 1.0
                self.scenario_name = f"Tir Précis : Reprise de Centre ({'Gauche' if is_left else 'Droit'})"
                bx = sign_x * random.uniform(2600, 3200)
                by = random.uniform(2200, 3600)
                bz = random.uniform(120, 280)
                bvx = -sign_x * random.uniform(1100, 1700)
                bvy = random.uniform(-250, 100)
                bvz = random.uniform(120, 300)

                car_x = random.uniform(-600, 600)
                car_y = random.uniform(400, 1600)
                yaw = math.atan2(by - car_y, (bx + bvx * 0.8) - car_x)
                speed = random.uniform(1100, 1600)
                vx = speed * math.cos(yaw)
                vy = speed * math.sin(yaw)

            else:
                is_left = random.random() < 0.5
                sign_x = -1.0 if is_left else 1.0
                self.scenario_name = f"Tir Précis : Frappe sur Sortie de Mur ({'Gauche' if is_left else 'Droit'})"
                bx = sign_x * 3400.0
                by = random.uniform(1200, 2600)
                bz = random.uniform(250, 600)
                bvx = -sign_x * random.uniform(1000, 1500)
                bvy = random.uniform(200, 600)
                bvz = random.uniform(50, 250)

                car_x = random.uniform(-800, 800)
                car_y = random.uniform(-400, 800)
                yaw = math.atan2(by - car_y, bx - car_x)
                speed = random.uniform(1000, 1500)
                vx = speed * math.cos(yaw)
                vy = speed * math.sin(yaw)

            sb = rsim.CarState()
            sb.pos = rsim.Vec(car_x, car_y, 17)
            sb.vel = rsim.Vec(vx, vy, 0)
            sb.rot_mat = rsim.Angle(yaw, 0, 0).as_rot_mat()
            sb.boost = 90.0
            sb.is_on_ground = True
            self.car_blue.set_state(sb)

            ball = rsim.BallState()
            ball.pos = rsim.Vec(bx, by, bz)
            ball.vel = rsim.Vec(bvx, bvy, bvz)
            self.arena.ball.set_state(ball)

            if self.car_orange is not None:
                so = rsim.CarState()
                so.pos = rsim.Vec(random.uniform(-650, 650), random.uniform(4900, 5080), 17)
                so.vel = rsim.Vec(random.uniform(-350, 350), 0, 0)
                so.rot_mat = rsim.Angle(-math.pi / 2, 0, 0).as_rot_mat()
                so.boost = 100.0
                so.is_on_ground = True
                self.car_orange.set_state(so)

        elif self.mode == "movement":
            # DÉPLACEMENT & RÉTABLISSEMENT : Sorties de mur, wave-dashes, réceptions 4 roues, half-flips
            archetype = random.choice(["wall_recovery", "airborne_tumble", "half_flip_retreat"])
            if archetype == "wall_recovery":
                is_left = random.random() < 0.5
                sign_x = -1.0 if is_left else 1.0
                self.scenario_name = f"Déplacement : Descente Mur {'Gauche' if is_left else 'Droit'} & Wave-Dash"
                car_x = sign_x * (4096.0 - 18.0)
                car_y = random.uniform(-2500, 2500)
                car_z = random.uniform(800, 1500)
                roll = -sign_x * math.pi / 2.0
                yaw = math.pi / 2.0 if random.random() < 0.5 else -math.pi / 2.0

                sb = rsim.CarState()
                sb.pos = rsim.Vec(car_x, car_y, car_z)
                sb.vel = rsim.Vec(0, random.uniform(-600, 600), random.uniform(-250, 50))
                sb.rot_mat = rsim.Angle(yaw, 0, roll).as_rot_mat()
                sb.boost = 40.0
                sb.is_on_ground = True
                self.car_blue.set_state(sb)

                ball = rsim.BallState()
                ball.pos = rsim.Vec(0, 0, BALL_RADIUS)
                ball.vel = rsim.Vec(0, 0, 0)
                self.arena.ball.set_state(ball)

            elif archetype == "airborne_tumble":
                self.scenario_name = "Déplacement : Projection Aérienne & Rétablissement 4 Roues"
                car_x = random.uniform(-1400, 1400)
                car_y = random.uniform(-2000, 2000)
                car_z = random.uniform(600, 1100)
                sb = rsim.CarState()
                sb.pos = rsim.Vec(car_x, car_y, car_z)
                sb.vel = rsim.Vec(random.uniform(-1000, 1000), random.uniform(-1000, 1000), random.uniform(150, 450))
                sb.rot_mat = rsim.Angle(random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)).as_rot_mat()
                sb.boost = 50.0
                sb.is_on_ground = False
                self.car_blue.set_state(sb)

                ball = rsim.BallState()
                ball.pos = rsim.Vec(0, 0, BALL_RADIUS)
                ball.vel = rsim.Vec(0, 0, 0)
                self.arena.ball.set_state(ball)

            else:
                self.scenario_name = "Déplacement : Demi-Tour Rapide (Half-Flip) en Repli Défensif"
                car_x = random.uniform(-600, 600)
                car_y = random.uniform(400, 1800)
                sb = rsim.CarState()
                sb.pos = rsim.Vec(car_x, car_y, 17)
                sb.vel = rsim.Vec(0, random.uniform(200, 500), 0)
                sb.rot_mat = rsim.Angle(math.pi / 2, 0, 0).as_rot_mat()  # Face vers le camp adverse !
                sb.boost = 60.0
                sb.is_on_ground = True
                self.car_blue.set_state(sb)

                # Balle dégagée au-dessus de sa tête vers ses propres cages
                ball = rsim.BallState()
                ball.pos = rsim.Vec(random.uniform(-400, 400), random.uniform(-2200, -3800), random.uniform(350, 800))
                ball.vel = rsim.Vec(random.uniform(-200, 200), random.uniform(-1200, -1800), random.uniform(100, 300))
                self.arena.ball.set_state(ball)

        elif self.mode == "kickoff":
            # TEST KICK-OFF : Démonstration du Speedflip Nexto officiel sur les 5 positions
            spawn_name, pos, yaw = KICKOFF_SPAWNS_BLUE[self.scenario_index % len(KICKOFF_SPAWNS_BLUE)]
            self.scenario_index += 1
            self.scenario_name = f"Kickoff Nexto Speedflip ({spawn_name})"

            sb = rsim.CarState()
            sb.pos = pos
            sb.rot_mat = rsim.Angle(yaw, 0, 0).as_rot_mat()
            sb.boost = 33.3
            sb.is_on_ground = True
            self.car_blue.set_state(sb)

            ball = rsim.BallState()
            ball.pos = rsim.Vec(0, 0, BALL_RADIUS)
            ball.vel = rsim.Vec(0, 0, 0)
            self.arena.ball.set_state(ball)

    def _apply_controls(self, car: rsim.Car, ctrl):
        cc = rsim.CarControls()
        cc.throttle = float(ctrl.throttle)
        cc.steer = float(ctrl.steer)
        cc.pitch = float(ctrl.pitch)
        cc.yaw = float(ctrl.yaw)
        cc.roll = float(ctrl.roll)
        cc.jump = bool(ctrl.jump)
        cc.boost = bool(ctrl.boost)
        cc.handbrake = bool(ctrl.handbrake)
        car.set_controls(cc)

    def step(self):
        packet = self.bridge.to_packet(self.arena)

        # 1. Contrôles Blue (susu_bot)
        ctrl_blue = self.bot_blue.get_output(packet)
        self._apply_controls(self.car_blue, ctrl_blue)

        # 2. Contrôles Orange (adversaire en match ou gardien de but)
        if self.car_orange is not None and self.bot_orange is not None:
            ctrl_orange = self.bot_orange.get_output(packet)
            self._apply_controls(self.car_orange, ctrl_orange)

        # 3. Avancement de la simulation RocketSim
        self.arena.step(1)
        self.scenario_timer += 1

        # 4. Analyse physique du ballon et des voitures
        ball = self.arena.ball.get_state()
        ball_pos = ball.pos
        ball_vel = ball.vel
        ball_speed = math.sqrt(ball_vel.x ** 2 + ball_vel.y ** 2 + ball_vel.z ** 2)
        shot_kmh = ball_speed * 0.036
        if shot_kmh > self.max_shot_kmh:
            self.max_shot_kmh = shot_kmh

        is_goal_blue = ball_pos.y > GOAL_Y and abs(ball_pos.x) < 890.0 and ball_pos.z < 640.0
        is_goal_orange = ball_pos.y < -GOAL_Y and abs(ball_pos.x) < 890.0 and ball_pos.z < 640.0

        if is_goal_blue:
            self.score_blue += 1
            if shot_kmh > 85.0:
                self.power_shots_count += 1
            self._reset_scenario()
            return
        elif is_goal_orange:
            self.score_orange += 1
            self._reset_scenario()
            return

        # 5. Validation stricte des mécaniques selon le mode
        if self.mode == "aerial":
            # ÉCHEC STRICT si le ballon touche le sol (après 15 ticks = 0.12s de grace)
            if self.scenario_timer > 15 and ball_pos.z <= (BALL_RADIUS + 8.0):
                self.aerial_fails += 1
                self._reset_scenario()
                return

            # Détection de contact aérien propre
            cs = self.car_blue.get_state()
            dist = math.sqrt((cs.pos.x - ball_pos.x) ** 2 + (cs.pos.y - ball_pos.y) ** 2 + (cs.pos.z - ball_pos.z) ** 2)
            if dist < 175.0 and ball_pos.z > 350.0 and cs.pos.z > 200.0:
                self.aerial_touches += 1

            if self.scenario_timer > 600:  # 5.0s max par cloche aérienne
                self._reset_scenario()
                return

        elif self.mode == "flick":
            # ÉCHEC FLICK : Balle tombée du toit et roulant au sol plus de 35 ticks (~0.3s)
            if ball_pos.z <= (BALL_RADIUS + 5.0):
                self.flick_floor_ticks += 1
                if self.flick_floor_ticks > 35:
                    self.flick_fails += 1
                    self._reset_scenario()
                    return
            else:
                self.flick_floor_ticks = 0

            # Détection d'éjection puissante de flick
            if ball_vel.z > 350.0 and ball_pos.z > 220.0 and ball_speed > 1300.0:
                self.flick_successes += 1

            if self.scenario_timer > 480:  # 4.0s max
                self._reset_scenario()
                return

        elif self.mode == "shooting":
            if shot_kmh > 85.0 and ball_vel.y > 1000.0:
                self.power_shots_count += 1
            if self.scenario_timer > 480 or ball_pos.y > GOAL_Y:
                self._reset_scenario()
                return

        elif self.mode == "movement":
            if self.scenario_timer > 360:  # Nouveau rétablissement toutes les 3 secondes
                self.recoveries_count += 1
                self._reset_scenario()
                return

        elif self.mode == "kickoff":
            if ball_speed > 150.0 or self.scenario_timer > 300:
                self.touches_count += 1
                self._reset_scenario()
                return

    def render(self):
        if not self.enable_vis:
            return

        boost_states = [p.get_state().is_active for p in self.pads]
        ball = self.arena.ball.get_state()
        cars = [(c.id, c.team, c.get_config(), c.get_state()) for c in self.arena.get_cars()]

        rlviser.render(
            tick_count=self.arena.tick_count,
            tick_rate=TICK_RATE,
            game_mode=self.arena.game_mode,
            boost_pad_states=boost_states,
            ball=ball,
            cars=cars,
        )

    def run(self):
        print(f"\n🚀 Simulation RocketSim démarrée : Mode [{self.mode.upper()}] à {TICK_RATE:.0f} FPS...")
        print("💡 Appuyez sur [Ctrl + C] dans cette console pour arrêter ou changer de mode.\n")

        clock_target = time.perf_counter()
        report_time = time.perf_counter()
        ticks_count = 0

        try:
            while self.running:
                # Synchronisation 120 FPS temps réel
                now = time.perf_counter()
                elapsed = now - clock_target
                delay = (DT / self.speed_mult) - elapsed

                if delay > 0.001:
                    time.sleep(delay)
                clock_target = time.perf_counter()

                # Vérifie l'état de pause ou de vitesse de RLViser
                if self.enable_vis:
                    try:
                        if rlviser.get_game_paused():
                            time.sleep(0.05)
                            continue
                        user_speed = rlviser.get_game_speed()
                        if 0.1 <= user_speed <= 10.0:
                            self.speed_mult = user_speed
                    except Exception:
                        pass

                self.step()
                self.render()
                ticks_count += 1

                # Télémétrie en direct dans la console (mise à jour toutes les 0.2s)
                if now - report_time >= 0.2:
                    cs = self.car_blue.get_state()
                    car_speed = math.sqrt(cs.vel.x ** 2 + cs.vel.y ** 2 + cs.vel.z ** 2)
                    car_speed_kmh = car_speed * 0.036
                    b = self.arena.ball.get_state()
                    fps_real = ticks_count / max(0.001, (now - report_time))
                    ticks_count = 0
                    report_time = now

                    if self.mode == "aerial":
                        details = f"Air touches: {self.aerial_touches} | Fails sol: {self.aerial_fails} | Alt: {b.pos.z:.0f}"
                    elif self.mode == "flick":
                        details = f"Flicks: {self.flick_successes} | Fails sol: {self.flick_fails} | Alt: {b.pos.z:.0f}"
                    elif self.mode == "shooting":
                        details = f"Buts: {self.score_blue} | Missiles >85km/h: {self.power_shots_count} | Max: {self.max_shot_kmh:.0f}km/h"
                    elif self.mode == "movement":
                        details = f"Drills: {self.recoveries_count} | Sol: {cs.is_on_ground} | Alt: {cs.pos.z:.0f}"
                    elif self.mode == "match":
                        details = f"{self.bot_blue.name} {self.score_blue} - {self.score_orange} {self.bot_orange.name} | Boost: {cs.boost:.0f}%"
                    else:
                        details = f"Kickoffs: {self.touches_count} | Vitesse: {car_speed_kmh:.0f}km/h"

                    status_line = (
                        f"\r[{self.scenario_name[:36]:<36}] {details} | "
                        f"Vitesse: {car_speed_kmh:.0f} km/h | {fps_real:.0f} FPS "
                    )
                    sys.stdout.write(status_line)
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\n⏹️ Arrêt demandé par l'utilisateur.")
        finally:
            if self.enable_vis:
                try:
                    rlviser.quit()
                except Exception:
                    pass
            print("✅ Simulation RocketSim fermée proprement.")


def interactive_menu() -> str:
    print("=" * 80)
    print("🚀 SUSU_BOT — SIMULATION & VISUALISATION 3D ROCKETSIM (RLViser)")
    print("=" * 80)
    print("Choisissez le mode de simulation 3D à observer :")
    print("  [1] Match 1v1 Normal     (susu_bot VS naze_bot en duel direct 3D 120 FPS)")
    print("  [2] Démo Aérienne        (Balles hautes en cloche, air dribbles, sorties de mur)")
    print("  [3] Démo Flick           (Contrôle de balle sur le toit, pop de flick puissant)")
    print("  [4] Démo Tir Précis      (Balles rebondissantes, frappes supersoniques cadrées)")
    print("  [5] Démo Déplacement     (Rotations, wave-dashes, rétablissements 4 roues)")
    print("  [6] Test Kick-off        (Test en boucle du Speedflip Nexto sur les 5 positions)")
    print("  [0] Quitter")
    print("=" * 80)

    try:
        choice = input("Votre choix (1-6, ou Entrée pour Match 1v1) : ").strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"

    mapping = {
        "1": "match",
        "2": "aerial",
        "3": "flick",
        "4": "shooting",
        "5": "movement",
        "6": "kickoff",
        "": "match",
        "0": "quit",
        "q": "quit",
    }
    return mapping.get(choice, "match")


def main():
    parser = argparse.ArgumentParser(description="Lanceur RocketSim 3D avec RLViser pour susu_bot")
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        choices=["match", "aerial", "flick", "shooting", "movement", "kickoff", ""],
        help="Mode de simulation ('match', 'aerial', 'flick', 'shooting', 'movement', 'kickoff')",
    )
    parser.add_argument("--no_vis", action="store_true", help="Désactive la fenêtre 3D RLViser")
    parser.add_argument("--speed", type=float, default=1.0, help="Multiplicateur de vitesse (défaut: 1.0)")
    args = parser.parse_args()

    mode = args.mode
    if not mode:
        mode = interactive_menu()

    if mode == "quit":
        print("Fermeture.")
        return

    sim = RocketSimVisualizer(
        mode=mode,
        enable_vis=not args.no_vis,
        speed_mult=args.speed,
    )
    sim.run()


if __name__ == "__main__":
    main()

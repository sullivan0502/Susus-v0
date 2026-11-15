#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SUSU_BOT CLUSTER COORDINATOR — Orchestrateur Multi-GPU / Multi-Spécialiste pour RocketSim C++
========================================================================================
Fonctionnalités :
1. Détection automatique des GPUs CUDA disponibles (1x, 2x, 4x, 8x RTX 3050 / 3090 / 4090 / 5090).
2. Répartition intelligente des rôles spécialisés par carte :
   - GPU 0 : AÉRIENNE & AIR DRIBBLE & FLIP RESETS (Boost Illimité, récompense exponentielle en l'air)
   - GPU 1 : DRIBBLE AU SOL, FLICKS & CATCH (Portage capot, pop de flick, rattrapage post-flick)
   - GPU 2 : DÉPLACEMENTS, RECOVERY, ROTATIONS & VITESSE (Supersonic, atterrissages 4 roues, wave-dash)
   - GPU 3 : TIRS, PRÉCISION & VISÉE CAGES (Sniping chirurgical, balles rebondissantes, jackpots buts)
3. Exploitation totale du CPU et de la RAM :
   - Divise les threads CPU (ex: 256 threads EPYC) équitablement entre les GPUs.
   - Maximise les arènes par worker (jusqu'à 6 arènes = 960+ matchs simultanés dans 258 Go de RAM).
4. Tableau de bord en direct (Live Dashboard) affichant les métriques par GPU et la vitesse cumulée.
5. Export automatique et synchronisation du meilleur modèle champion vers `src/susu-model.pt`.
6. Arrêt propre (Ctrl+C) : chaque instance sauvegarde son checkpoint et exporte son modèle.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple
import psutil
import torch

# Fix console encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rocketsim_env import get_effective_ram_gb

ROLE_CONFIGS = {
    "aerial": {
        "label": "AÉRIENNES & RESETS",
        "desc": "Air Dribbles, Fast Aerials, Flip Resets, Keep-Up",
        "infinite_boost": True,
        "default_role": "air_dribble",
    },
    "flick": {
        "label": "FLICKS & DRIBBLE",
        "desc": "Dribble Carry, Pop de Flick, Post-Flick Catch",
        "infinite_boost": False,
        "default_role": "flick_dribble",
    },
    "movement": {
        "label": "DÉPLACEMENTS & ROT",
        "desc": "Vitesse Supersonic, Recovery 4 roues, Wave-dash",
        "infinite_boost": False,
        "default_role": "all_rounder",
    },
    "shooting": {
        "label": "TIRS & PRÉCISION",
        "desc": "Tirs Cages, Rebond Intercept, Sniping Puissant",
        "infinite_boost": False,
        "default_role": "all_rounder",
    },
    "mix": {
        "label": "MIX COMPLET",
        "desc": "Matchs 1v1 complets et polyvalence globale",
        "infinite_boost": False,
        "default_role": "all_rounder",
    },
}

DEFAULT_4GPU_ROLES = ["aerial", "flick", "movement", "shooting"]
DEFAULT_2GPU_ROLES = ["aerial", "shooting"]
DEFAULT_1GPU_ROLES = ["mix"]


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def parse_args():
    parser = argparse.ArgumentParser(
        description="SUSU_BOT Multi-GPU Cluster Orchestrator (RocketSim C++ + PyTorch CUDA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="GPUs to use: 'auto' (all available), or comma-separated list like '0,1,2,3'",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="",
        help="Comma-separated roles per GPU, e.g. 'aerial,flick,movement,shooting'",
    )
    parser.add_argument(
        "--total_workers",
        type=int,
        default=0,
        help="Total CPU workers across all GPUs (0 = auto-scale based on detected CPU cores)",
    )
    parser.add_argument(
        "--arenas_per_worker",
        type=int,
        default=0,
        help="Arenas per worker (0 = auto-scale based on RAM: 4-6 on servers)",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=0,
        help="PPO batch size per GPU (0 = auto-scale based on GPU VRAM)",
    )
    parser.add_argument(
        "--minibatch_size_per_gpu",
        type=int,
        default=0,
        help="PPO minibatch size per GPU",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100_000,
        help="Save checkpoint every N steps per GPU",
    )
    parser.add_argument(
        "--sync_interval",
        type=int,
        default=30,
        help="Interval in seconds between dashboard updates and champion model sync",
    )
    parser.add_argument(
        "--cost_per_hour",
        type=float,
        default=float(os.environ.get("SUSU_BOT_COST_PER_HOUR", "1.876")),
        help="Server rental cost in USD per hour (for live spending tracker)",
    )
    return parser.parse_args()


class GPUInstance:
    """Manages an individual training subprocess for a dedicated GPU and specialization."""

    def __init__(
        self,
        gpu_id: int,
        specialization: str,
        n_proc: int,
        arenas_per_worker: int,
        batch_size: int,
        minibatch_size: int,
        save_every: int,
        python_exe: str,
        base_checkpoint_dir: Path,
    ):
        self.gpu_id = gpu_id
        self.specialization = specialization
        self.cfg = ROLE_CONFIGS.get(specialization, ROLE_CONFIGS["mix"])
        self.n_proc = n_proc
        self.arenas_per_worker = arenas_per_worker
        self.total_matches = n_proc * arenas_per_worker
        self.total_cars = self.total_matches * 2
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.save_every = save_every
        self.python_exe = python_exe

        self.save_dir = _ROOT / f"training/checkpoints_gpu{gpu_id}_{specialization}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = _ROOT / f"src/susu-model-{specialization}.pt"
        self.status_file = _ROOT / f"training/status_gpu{gpu_id}_{specialization}.json"

        # Initialize checkpoint folder if empty by seeding from existing main checkpoint
        if not any(self.save_dir.glob("**/PPO_POLICY.pt")):
            main_chk_dir = _ROOT / "training/checkpoints"
            candidates = sorted(list(main_chk_dir.glob("**/PPO_POLICY.pt")), key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else 0, reverse=True)
            if candidates:
                src_folder = candidates[0].parent
                dst_folder = self.save_dir / src_folder.name
                dst_folder.mkdir(parents=True, exist_ok=True)
                for f in src_folder.glob("*"):
                    if f.is_file():
                        try:
                            shutil.copy2(f, dst_folder / f.name)
                        except Exception:
                            pass
                bkv = src_folder.parent.parent / "BOOK_KEEPING_VARS.json"
                if bkv.is_file():
                    try:
                        shutil.copy2(bkv, self.save_dir / "BOOK_KEEPING_VARS.json")
                    except Exception:
                        pass
                print(f"  [GPU {gpu_id}] Checkpoint initial copié depuis {src_folder.name} vers {self.save_dir.name}")

        self.process: Optional[subprocess.Popen] = None
        self.log_file = _ROOT / f"training/gpu{gpu_id}_{specialization}.log"
        self._log_fh = None

        # Metrics for dashboard
        self.current_timesteps = 0
        self.current_steps_per_sec = 0.0
        self.current_reward = 0.0
        self.last_checkpoint = "-"
        self.restarts = 0
        self.start_time = time.time()

    def start(self):
        env_copy = os.environ.copy()

        # Isolate child process strictly to its allocated physical GPU via CUDA_VISIBLE_DEVICES.
        # Inside the child process, the isolated GPU will always be indexed as 'cuda:0'.
        if torch.cuda.is_available() and self.gpu_id >= 0:
            original_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if original_visible is not None:
                visible_list = [v.strip() for v in original_visible.split(",") if v.strip()]
                if 0 <= self.gpu_id < len(visible_list):
                    env_copy["CUDA_VISIBLE_DEVICES"] = visible_list[self.gpu_id]
                else:
                    env_copy["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            else:
                env_copy["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            device_arg = "cuda:0"
        else:
            device_arg = "cpu"
            env_copy.pop("CUDA_VISIBLE_DEVICES", None)

        cmd = [
            self.python_exe,
            "-B",
            "-u",
            str(_ROOT / "train.py"),
            "--continuous",
            "--engine", "rocketsim",
            "--device", device_arg,
            "--n_proc", str(self.n_proc),
            "--arenas_per_worker", str(self.arenas_per_worker),
            "--specialization", self.specialization,
            "--save_dir", str(self.save_dir),
            "--model_path", str(self.model_path),
            "--status_file", str(self.status_file),
            "--save_every", str(self.save_every),
        ]
        if self.batch_size > 0:
            cmd.extend(["--batch_size", str(self.batch_size)])
        if self.minibatch_size > 0:
            cmd.extend(["--minibatch_size", str(self.minibatch_size)])
        if self.cfg["infinite_boost"]:
            cmd.append("--infinite_boost")

        env_copy["SUSU_BOT_SPECIALIZATION"] = self.specialization
        env_copy["SUSU_BOT_INFINITE_BOOST"] = "1" if self.cfg["infinite_boost"] else "0"
        env_copy["SUSU_BOT_PHYSICAL_GPU"] = str(self.gpu_id)

        self._log_fh = open(self.log_file, "a", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env_copy,
        )
        self.start_time = time.time()

    def poll(self) -> Optional[int]:
        if self.process is None:
            return -1
        return self.process.poll()

    def restart_if_needed(self):
        ret = self.poll()
        if ret is not None and ret != 0:
            self.restarts += 1
            time.sleep(2.0)
            self.start()

    def update_metrics_from_disk(self):
        if self.status_file.exists():
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.current_timesteps = int(data.get("cumulative_timesteps", self.current_timesteps))
                self.current_steps_per_sec = float(data.get("collected_steps_per_sec", self.current_steps_per_sec))
                self.current_reward = float(data.get("mean_reward", self.current_reward))
                chk = data.get("latest_checkpoint", "")
                if chk:
                    self.last_checkpoint = Path(chk).name
            except Exception:
                pass

    def stop(self, timeout: float = 15.0):
        if self.process is not None and self.process.poll() is None:
            try:
                if sys.platform == "win32":
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=timeout)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        if self._log_fh is not None and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


def main():
    args = parse_args()

    # Determine Python executable
    python_exe = sys.executable
    if sys.platform != "win32":
        candidates = [
            _ROOT / ".venv/bin/python",
            _ROOT / "venv/bin/python",
            Path("/opt/venv/bin/python"),
        ]
        for c in candidates:
            if c.exists() and os.access(c, os.X_OK):
                python_exe = str(c)
                break

    # Detect GPUs
    num_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.gpus == "auto":
        target_gpus = list(range(max(1, num_cuda))) if num_cuda > 0 else [0]
    elif args.gpus in ("all", "0,1,2,3"):
        target_gpus = list(range(num_cuda)) if num_cuda > 0 else [0]
    else:
        target_gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip().isdigit()]

    n_gpus = len(target_gpus)

    # Determine specializations
    if args.roles:
        roles = [r.strip().lower() for r in args.roles.split(",") if r.strip()]
    elif n_gpus >= 4:
        roles = DEFAULT_4GPU_ROLES
    elif n_gpus == 2:
        roles = DEFAULT_2GPU_ROLES
    else:
        roles = DEFAULT_1GPU_ROLES

    # Match roles length to GPU count
    assigned_roles = []
    for i in range(n_gpus):
        assigned_roles.append(roles[i % len(roles)])

    # Compute CPU and RAM distribution
    total_cores = os.cpu_count() or 12
    total_ram_gb = get_effective_ram_gb()
    
    # Reserve cores for OS and IPC, distribute remainder across active GPUs
    if total_cores >= 64:
        usable_cores = max(4, total_cores - 12)
    elif total_cores >= 32:
        usable_cores = max(4, total_cores - 4)
    else:
        usable_cores = total_cores

    if args.total_workers > 0:
        total_workers = args.total_workers
    else:
        total_workers = usable_cores

    workers_per_gpu = max(1, total_workers // n_gpus)

    # Arenas per worker: maximize based on available RAM
    if args.arenas_per_worker > 0:
        arenas_per_worker = args.arenas_per_worker
    else:
        if total_ram_gb >= 180.0 and total_cores >= 64:
            arenas_per_worker = 6  # 258 GB RAM server: 6 arenas per worker
        elif total_ram_gb >= 60.0:
            arenas_per_worker = 4
        elif total_ram_gb >= 24.0:
            arenas_per_worker = 3
        else:
            arenas_per_worker = 2

    # Batch sizes per GPU based on VRAM
    vram_per_gpu = 4.0
    if num_cuda > 0:
        try:
            vram_per_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            vram_per_gpu = 4.0

    if args.batch_size_per_gpu > 0:
        batch_size = args.batch_size_per_gpu
    else:
        if vram_per_gpu >= 28.0:
            batch_size = 200_000 if workers_per_gpu >= 40 else 150_000
        elif vram_per_gpu >= 20.0:
            batch_size = 150_000
        elif vram_per_gpu >= 14.0:
            batch_size = 100_000
        else:
            batch_size = 30_000

    minibatch_size = args.minibatch_size_per_gpu if args.minibatch_size_per_gpu > 0 else min(batch_size, 30_000 if vram_per_gpu >= 20.0 else 15_000)

    total_sim_matches = n_gpus * workers_per_gpu * arenas_per_worker
    total_cars_in_sim = total_sim_matches * 2

    print("=" * 92)
    print("🚀 SUSU_BOT CLUSTER MULTI-GPU & MULTI-SPÉCIALISTE (C++ ROCKETSIM + PYTORCH CUDA)")
    print("=" * 92)
    print(f"  [Matériel Détecté] : {total_cores} threads CPU | {total_ram_gb:.1f} Go RAM | {num_cuda}x GPU CUDA ({vram_per_gpu:.1f} Go VRAM/carte)")
    print(f"  [GPUs Mobilisés]   : {n_gpus} carte(s) active(s) : {target_gpus}")
    print(f"  [Workers C++]      : {workers_per_gpu} workers par GPU × {n_gpus} GPUs = {workers_per_gpu * n_gpus} workers au total")
    print(f"  [Arènes Simultanées: {arenas_per_worker} arènes/worker -> {total_sim_matches} MATCHS / {total_cars_in_sim} VOITURES EN SIMULTANÉ !")
    print(f"  [PPO Batch GPU]    : {batch_size:,} steps par itération / GPU ({minibatch_size:,} minibatch)")
    print(f"  [Coût Horaire]     : {args.cost_per_hour:.3f} $/h")
    print("=" * 92)
    print("📋 RÉPARTITION DES SPÉCIALISATIONS PAR CARTE GRAPHIQUE :")
    for i, (gid, spec) in enumerate(zip(target_gpus, assigned_roles)):
        cfg = ROLE_CONFIGS.get(spec, ROLE_CONFIGS["mix"])
        inf_str = "BOOST ILLIMITÉ" if cfg["infinite_boost"] else "Boost Standard"
        print(f"  • GPU {gid} : {cfg['label']:<20} [{inf_str}] -> {cfg['desc']}")
    print("=" * 92)
    print("Démarrage des instances...")

    instances: List[GPUInstance] = []
    for gid, spec in zip(target_gpus, assigned_roles):
        inst = GPUInstance(
            gpu_id=gid,
            specialization=spec,
            n_proc=workers_per_gpu,
            arenas_per_worker=arenas_per_worker,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            save_every=args.save_every,
            python_exe=python_exe,
            base_checkpoint_dir=_ROOT / "training",
        )
        inst.start()
        instances.append(inst)
        time.sleep(1.0)

    print("✅ Toutes les instances sont lancées ! Surveillance active...")
    print("Appuyez sur Ctrl+C pour arrêter le cluster et sauvegarder l'ensemble des modèles.")
    print("")

    start_wall = time.time()
    last_sync = time.time()

    def handle_exit(sig, frame):
        print("\n\n🛑 Arrêt du cluster demandé (Ctrl+C). Sauvegarde de toutes les instances...")
        for inst in instances:
            inst.stop(timeout=10.0)
        print("✅ Toutes les instances ont sauvegardé leurs données avec succès.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handle_exit)

    try:
        while True:
            time.sleep(max(2.0, float(args.sync_interval)))

            # 1. Update metrics from all GPUs
            total_steps = 0
            total_rate = 0.0
            best_inst = None
            max_ts = -1

            for inst in instances:
                inst.restart_if_needed()
                inst.update_metrics_from_disk()
                total_steps += inst.current_timesteps
                total_rate += inst.current_steps_per_sec
                if inst.current_timesteps > max_ts:
                    max_ts = inst.current_timesteps
                    best_inst = inst

            # 2. Champion Model Sync: copy highest-progress model to primary susu-model.pt
            if best_inst and best_inst.model_path.exists():
                champion_dst = _ROOT / "src/susu-model.pt"
                try:
                    shutil.copy2(best_inst.model_path, champion_dst)
                except Exception:
                    pass

            # 3. Render Live Dashboard
            elapsed = time.time() - start_wall
            elapsed_str = format_duration(elapsed)
            hours_elapsed = elapsed / 3600.0
            spent_usd = hours_elapsed * args.cost_per_hour
            sim_years = total_steps / (365.25 * 86400.0 * 15.0)

            os.system("cls" if sys.platform == "win32" else "clear")
            print("=" * 96)
            print("🚀 SUSU_BOT CLUSTER MULTI-GPU (RocketSim C++ & PyTorch CUDA) — TABLEAU DE BORD EN DIRECT")
            print("=" * 96)
            print(f" {'GPU':<7} | {'Spécialité':<20} | {'Boost':<11} | {'Vitesse (steps/s)':<17} | {'Timesteps':<15} | {'Dernier Checkpoint'}")
            print("-" * 96)
            for inst in instances:
                b_str = "ILLIMITÉ" if inst.cfg["infinite_boost"] else "Standard"
                ts_str = f"{inst.current_timesteps:,}"
                rate_str = f"{inst.current_steps_per_sec:,.0f} st/s" if inst.current_steps_per_sec > 0 else "Initialisation..."
                print(f" GPU {inst.gpu_id:<3} | {inst.cfg['label']:<20} | {b_str:<11} | {rate_str:<17} | {ts_str:<15} | #{inst.last_checkpoint}")
            print("-" * 96)
            total_rate_str = f"{total_rate:,.0f} steps/s"
            total_ts_str = f"{total_steps:,}"
            print(f" TOTAL   | {len(instances)}x GPUs ({total_sim_matches} MATCHS) | Global      | {total_rate_str:<17} | {total_ts_str:<15} | Modèle: susu-model.pt")
            print("=" * 96)
            print(f" ⏱️  Temps écoulé : {elapsed_str} | ⚡ Vitesse Combinée : {total_rate_str} | 💰 Coût estimé : {spent_usd:.2f} $")
            print(f" 🎮 Expérience totale accumulée : ~{sim_years:.3f} ans de matchs simulés | 🔄 Relances auto : {sum(i.restarts for i in instances)}")
            print("=" * 96)
            print(" (Ctrl+C pour arrêter proprement et sauvegarder l'ensemble des modèles)")

    except (KeyboardInterrupt, SystemExit):
        handle_exit(None, None)


if __name__ == "__main__":
    main()

"""
Optimized C++ RocketSim + rlgym-sim + CUDA GPU Training Script for susu_bot (Susu-v0).
Features:
1. C++ RocketSim & rlgym-sim simulation running in parallel across the maximum available CPU cores.
2. VariedRandomStateSetter: Randomizes ball and car positions on the Ground (30%), in the Air (30%),
   on the Walls/Backboard (25%), and RandomState/Kickoffs (15%) instead of repetitive kickoffs.
3. LookupAction: 90-action discrete Susu-style LookupAction space.
4. CombinedReward: VelocityPlayerToBallReward, FaceBallReward, TouchBallReward, AlignBallGoal,
   VelocityBallToGoalReward, EventReward, and SusuMechanicsSimReward.
5. Dedicated NVIDIA GPU acceleration via PyTorch CUDA (cuda:0) verified for all PPO policy & value updates.
6. Persistent all-time cumulative training duration tracking with clean Ctrl+C summary.

Usage:
    python train.py
    python train.py --engine rocketsim --n_proc 12 --device cuda:0
"""

import argparse
from datetime import datetime
import dis
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import numpy as np
import psutil

# Fix Windows cp1252 console encoding issues and ensure instant line flushing
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True, write_through=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True, write_through=True)

# Prevent OpenMP/MKL thread explosion & IPC deadlocks across multiprocessing workers
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

# Disable core dumps on Linux (protects 30 GB disk on Vast.ai)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except Exception:
    pass

import torch
torch.set_num_threads(1)

# ==============================================================================
# ROBUST TORCH.LOAD & PPOLearner DESERIALIZATION MONKEY-PATCH
# Automatically handles cross-device mapping (CUDA -> CPU or CUDA:1 -> CUDA:0)
# to completely eliminate "Attempting to deserialize object on a CUDA device" crashes.
# ==============================================================================
_orig_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if "map_location" not in kwargs and not torch.cuda.is_available():
        kwargs["map_location"] = torch.device("cpu")
    try:
        return _orig_torch_load(*args, **kwargs)
    except RuntimeError as re:
        err_msg = str(re).lower()
        if "cuda" in err_msg or "deserialize" in err_msg or "device" in err_msg:
            kwargs["map_location"] = torch.device("cpu")
            return _orig_torch_load(*args, **kwargs)
        raise

torch.load = safe_torch_load

try:
    from rlgym_ppo.ppo.ppo_learner import PPOLearner

    def _safe_load_from(self, folder_path):
        assert os.path.exists(folder_path), f"PPO LEARNER CANNOT FIND FOLDER {folder_path}"
        dev = getattr(self, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_POLICY.pt"), map_location=dev)
        )
        self.value_net.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_VALUE_NET.pt"), map_location=dev)
        )
        self.policy_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"), map_location=dev)
        )
        self.value_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"), map_location=dev)
        )

    PPOLearner.load_from = _safe_load_from
except Exception:
    pass

_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.rocketsim_env import ensure_rocketsim_meshes_initialized

# Initialize RocketSim collision_meshes dynamically BEFORE importing rlgym_sim
ensure_rocketsim_meshes_initialized()

import RocketSim as rsim
import rlgym_sim
from rlgym_ppo import Learner
from rlgym_ppo.util import reporting, KBHit

import torch.nn as nn
from src.rocketsim_env import RocketSimEnvFactory, ensure_rocketsim_meshes_initialized


class SusuPolicyNetwork(nn.Module):
    def __init__(self, in_dim: int = 142, out_dim: int = 90):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def upgrade_checkpoint_obs_dim(checkpoint_folder: Path, target_in_dim: int = 142) -> None:
    """
    Losslessly upgrades a 94-dim or 106-dim checkpoint (policy, value_net, Adam optimizers, and BOOK_KEEPING_VARS.json)
    to `target_in_dim` (142 floats: +36 floats for 12 extra multi-horizon ball trajectory predictions up to 2.50s)
    by zero-padding the new input columns of `model.0.weight` so 100% of existing learned weights and
    outputs are preserved without any discontinuity.
    Also repairs inflated `reward_running_stats` variance if affected by multi-arena unsplit episodes.
    """
    if checkpoint_folder is None or not checkpoint_folder.exists():
        return

    pol_path = checkpoint_folder / "PPO_POLICY.pt"
    val_path = checkpoint_folder / "PPO_VALUE_NET.pt"
    pol_opt_path = checkpoint_folder / "PPO_POLICY_OPTIMIZER.pt"
    val_opt_path = checkpoint_folder / "PPO_VALUE_NET_OPTIMIZER.pt"
    bkv_path = checkpoint_folder / "BOOK_KEEPING_VARS.json"

    for net_path, opt_path in ((pol_path, pol_opt_path), (val_path, val_opt_path)):
        if not net_path.exists():
            continue
        sd = torch.load(str(net_path), map_location="cpu")
        if "model.0.weight" in sd and sd["model.0.weight"].shape[1] < target_in_dim:
            old_dim = int(sd["model.0.weight"].shape[1])
            diff = target_in_dim - old_dim
            out_features = int(sd["model.0.weight"].shape[0])
            pad_w = torch.zeros((out_features, diff), dtype=sd["model.0.weight"].dtype)
            sd["model.0.weight"] = torch.cat([sd["model.0.weight"], pad_w], dim=1)
            torch.save(sd, str(net_path))
            print(f"[susu_bot Upgrade] {net_path.name} mis à jour sans perte : {old_dim} -> {target_in_dim} entrées (+{diff} horizons balle)")

            if opt_path.exists():
                try:
                    opt_sd = torch.load(str(opt_path), map_location="cpu")
                    if "state" in opt_sd and 0 in opt_sd["state"]:
                        st0 = opt_sd["state"][0]
                        if "exp_avg" in st0 and st0["exp_avg"].shape[1] == old_dim:
                            st0["exp_avg"] = torch.cat(
                                [st0["exp_avg"], torch.zeros((out_features, diff), dtype=st0["exp_avg"].dtype)],
                                dim=1,
                            )
                        if "exp_avg_sq" in st0 and st0["exp_avg_sq"].shape[1] == old_dim:
                            mean_sq = float(st0["exp_avg_sq"].mean().item()) if st0["exp_avg_sq"].numel() > 0 else 1e-6
                            st0["exp_avg_sq"] = torch.cat(
                                [st0["exp_avg_sq"], torch.full((out_features, diff), mean_sq, dtype=st0["exp_avg_sq"].dtype)],
                                dim=1,
                            )
                        torch.save(opt_sd, str(opt_path))
                except Exception as e:
                    print(f"[susu_bot Upgrade] Note optimiseur {opt_path.name}: {e}")

    if bkv_path.exists():
        try:
            with open(bkv_path, "r", encoding="utf-8") as f:
                bkv = json.load(f)
            changed = False
            obs_st = bkv.get("obs_running_stats")
            if isinstance(obs_st, dict) and "mean" in obs_st and "var" in obs_st:
                cur_len = len(obs_st["mean"])
                if cur_len < target_in_dim:
                    diff = target_in_dim - cur_len
                    # Use existing ball prediction running stats (indices 68..79) as prior for new horizons
                    ref_means = obs_st["mean"][68:80] if cur_len >= 80 else [0.0] * 12
                    ref_vars = obs_st["var"][68:80] if cur_len >= 80 else [1.0] * 12
                    for i in range(diff):
                        obs_st["mean"].append(float(ref_means[i % len(ref_means)]))
                        obs_st["var"].append(float(ref_vars[i % len(ref_vars)]))
                    obs_st["shape"] = [target_in_dim]
                    changed = True

            rew_st = bkv.get("reward_running_stats")
            if isinstance(rew_st, dict) and "var" in rew_st and "count" in rew_st:
                cnt = max(1, int(rew_st["count"]))
                var_val = float(rew_st["var"][0]) if rew_st["var"] else 1.0
                ratio = var_val / cnt
                if ratio > 6000.0 or cnt > 250_000:
                    # Repair inflated return variance and cap count at 200k so reward normalization adapts quickly (Claude 2.3)
                    clean_ratio = 5116.0 if ratio > 6000.0 else ratio
                    new_cnt = min(cnt, 200_000)
                    rew_st["mean"] = [53.68]
                    rew_st["var"] = [float(new_cnt * clean_ratio)]
                    rew_st["count"] = int(new_cnt)
                    changed = True

            if changed:
                with open(bkv_path, "w", encoding="utf-8") as f:
                    json.dump(bkv, f, indent=4)
        except Exception as e:
            print(f"[susu_bot Upgrade] Note BOOK_KEEPING_VARS.json: {e}")


def find_latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    candidates = []
    if checkpoints_dir.exists():
        candidates.extend(checkpoints_dir.glob("**/PPO_POLICY.pt"))
    if checkpoints_dir.parent.exists():
        for sibling in checkpoints_dir.parent.glob("checkpoints*"):
            if sibling.is_dir():
                candidates.extend(sibling.glob("**/PPO_POLICY.pt"))
    if not candidates:
        return None

    def _sort_key(p: Path):
        folder_name = p.parent.name
        if folder_name.isdigit():
            return (1, int(folder_name), p.stat().st_mtime)
        return (0, 0, p.stat().st_mtime)

    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    in_dim: int = 142,
    out_dim: int = 90,
) -> Path:
    print(f"Loading checkpoint state from: {checkpoint_path}")
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    if "model.0.weight" in state_dict:
        in_dim = state_dict["model.0.weight"].shape[1]
        out_dim = state_dict["model.6.weight"].shape[0]
    model = SusuPolicyNetwork(in_dim=in_dim, out_dim=out_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    dummy_input = torch.zeros(1, in_dim, dtype=torch.float32)
    traced_model = torch.jit.trace(model, dummy_input)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))
    size_kb = output_path.stat().st_size / 1024.0
    print(f"Successfully exported TorchScript model to: {output_path} (size: {size_kb:.1f} KB)")
    return output_path


def make_env_create_fn(
    initial_stage: int = -1,
    spawn_opponents: bool = True,
    team_size: int = 1,
    tick_skip: int = 8,
    timeout_seconds: float = 300.0,
    no_touch_timeout: float = 30.0,
    end_on_goal: bool = True,
    engine_type: str = "rocketsim",
    arenas_per_worker: int = 1,
    specialization: str = None,
    infinite_boost: bool = False,
):
    return RocketSimEnvFactory(
        tick_skip=tick_skip,
        spawn_opponents=spawn_opponents,
        team_size=team_size,
        timeout_seconds=timeout_seconds,
        no_touch_seconds=no_touch_timeout,
        end_on_goal=end_on_goal,
        arenas_per_worker=arenas_per_worker,
        specialization=specialization,
        infinite_boost=infinite_boost,
    )



def _to_rel_path(path: Optional[Path], root_dir: Path = _ROOT_DIR) -> Optional[str]:
    """
    Converts an absolute Path inside the project to a portable forward-slash relative path
    so JSON status files contain zero hardcoded machine-specific paths.
    """
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(root_dir.resolve()).as_posix()
    except Exception:
        return Path(path).as_posix()


def format_duration_seconds(total_seconds: float) -> Tuple[str, dict]:
    """
    Formats a duration in seconds into human-readable French string (ans, mois, j, HHhMMmSSs)
    and returns a dictionary of summary metrics.
    """
    total_seconds = max(0.0, float(total_seconds))
    total_hours = total_seconds / 3600.0
    total_days = total_seconds / 86400.0
    total_years = total_seconds / (365.25 * 86400.0)
    total_matches_5min = total_seconds / 300.0

    years = int(total_seconds // (365.25 * 86400.0))
    rem = total_seconds % (365.25 * 86400.0)
    months = int(rem // (30.4375 * 86400.0))
    rem = rem % (30.4375 * 86400.0)
    days = int(rem // 86400.0)
    rem = rem % 86400.0
    hours = int(rem // 3600.0)
    rem = rem % 3600.0
    minutes = int(rem // 60.0)
    seconds = int(rem % 60.0)

    parts = []
    if years > 0:
        suffix = "s" if years > 1 else ""
        parts.append(f"{years} an{suffix}")
    if months > 0:
        parts.append(f"{months} mois")
    if days > 0:
        parts.append(f"{days} j")
    parts.append(f"{hours:02d}h{minutes:02d}m{seconds:02d}s")

    exp_dict = {
        "years": total_years,
        "total_days": total_days,
        "total_hours": total_hours,
        "total_seconds": total_seconds,
        "total_matches_5min": total_matches_5min,
    }
    return ", ".join(parts), exp_dict


def format_in_game_time(cumulative_timesteps: int, tick_skip: int = 8) -> Tuple[str, dict]:
    """
    Computes cumulative simulated match time from agent timesteps.
    In Rocket League, tick rate is 120 ticks/s. With tick_skip=8,
    each step represents 8/120 = 1/15th of a second of actual match play.
    """
    total_seconds = float(cumulative_timesteps) * (tick_skip / 120.0)
    return format_duration_seconds(total_seconds)


def load_persisted_wall_time(status_file: Path, save_dir: Path, loaded_steps: int = 0) -> float:
    """
    Loads the cumulative real wall-clock training time (in seconds) across all historical sessions.
    """
    if status_file is not None and status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "cumulative_wall_time_seconds" in data and float(data["cumulative_wall_time_seconds"]) > 0:
                return float(data["cumulative_wall_time_seconds"])
        except Exception:
            pass

    latest_chk = find_latest_checkpoint(save_dir) if save_dir is not None else None
    if latest_chk is not None:
        bkv_path = latest_chk.parent / "BOOK_KEEPING_VARS.json"
        if bkv_path.exists():
            try:
                with open(bkv_path, "r", encoding="utf-8") as f:
                    bkv = json.load(f)
                if "cumulative_wall_time_seconds" in bkv and float(bkv["cumulative_wall_time_seconds"]) > 0:
                    return float(bkv["cumulative_wall_time_seconds"])
            except Exception:
                pass

    if loaded_steps > 0:
        return round(float(loaded_steps) / 1724.0, 2)
    return 0.0


def get_effective_ram_gb() -> float:
    """
    Returns available physical RAM in GB, detecting both system physical RAM and
    Linux container cgroup v1/v2 memory limits (essential for Vast.ai docker containers).
    Handles both private cgroup namespaces and host/nested cgroup hierarchies from /proc/self/cgroup.
    """
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
                        # Unconstrained cgroup v1 is typically 0x7FFFFFFFFFFFF000 (>= 9 * 10^18)
                        if 0 < limit < min(total_bytes, 1024 ** 5):
                            total_bytes = limit
            except Exception:
                pass

    return max(0.5, total_bytes / (1024.0 ** 3))


# Backward-compatible alias
get_system_ram_gb = get_effective_ram_gb


def calculate_safe_worker_count(
    detected_cores: int,
    total_ram_gb: float,
    requested_n_proc: int = 0,
    ram_per_worker_gb: float = 0.40,
    max_ram_ratio: float = 0.90,
    max_default_ceiling: int = 256,
    hard_max_ceiling: int = 256,
) -> int:
    """
    Calculates the parallel worker count (n_proc / num_envs) scaled to the hardware.
    On high-end servers (e.g. 64-128 physical cores / 128-256 threads EPYC with 258 GB RAM),
    mobilizes the maximum available CPU threads while reserving headroom for OS and IPC.
    """
    if requested_n_proc > 0:
        return max(1, requested_n_proc)

    max_ram_budget_gb = total_ram_gb * max_ram_ratio
    ram_worker_cap = max(1, min(hard_max_ceiling, int(max_ram_budget_gb / ram_per_worker_gb)))
    if detected_cores >= 64:
        target_workers = max(1, detected_cores - 8)
    elif detected_cores >= 32:
        target_workers = max(1, detected_cores - 4)
    elif detected_cores >= 16:
        target_workers = int(round(detected_cores * 1.5))
    else:
        target_workers = detected_cores
    selected = min(target_workers, ram_worker_cap, max_default_ceiling)
    return max(1, selected)


auto_scale_workers = calculate_safe_worker_count


def verify_hardware_and_cuda_gpu(
    n_proc: int, requested_device: str, total_ram_gb: Optional[float] = None
) -> Tuple[str, float, float]:
    """
    Verifies and logs:
    1. CPU physical & logical core counts, system RAM (physical and cgroup v1/v2 container limits), and parallel worker allocation.
    2. C++ RocketSim binary & collision_meshes + rlgym-sim installation.
    3. PyTorch CUDA GPU availability, device name (RTX 3050 / 3090 / 4090 / 5090), VRAM, Ampere/Ada/Blackwell TF32 TensorCore support,
       and live CUDA tensor verification on the requested GPU device.
    Returns (verified_device, total_ram_gb, total_vram_gb).
    """
    phys_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = os.cpu_count() or phys_cores
    if total_ram_gb is None or total_ram_gb <= 0:
        total_ram_gb = get_effective_ram_gb()
    max_ram_budget_gb = total_ram_gb * 0.85
    mesh_dir = ensure_rocketsim_meshes_initialized()
    total_vram_gb = 0.0

    print("=" * 84)
    print("🔍 VÉRIFICATION MATÉRIELLE & MOTEUR C++ ROCKETSIM / CUDA GPU (AUTO-SCALING)")
    print("=" * 84)
    print(f"  [CPU] Cœurs physiques / logiques : {phys_cores} cœurs physiques / {logical_cores} threads logiques")
    print(f"  [RAM] Mémoire vive disponible    : {total_ram_gb:.1f} Go RAM (Budget sécurité 85% : {max_ram_budget_gb:.1f} Go max, 0 Go swap)")
    print(f"  [CPU] Workers RocketSim C++      : {n_proc} instances simultanées (Plafond RAM/Cgroup respecté)")
    print(f"  [C++] Extension RocketSim        : {rsim.__file__}")
    print(f"  [C++] Collision Meshes (dynamique): {_to_rel_path(Path(mesh_dir)) if mesh_dir else mesh_dir}")
    print(f"  [SIM] Bibliothèque rlgym-sim     : {rlgym_sim.__file__}")

    if torch.cuda.is_available() and requested_device != "cpu":
        device = requested_device if str(requested_device).startswith("cuda") else "cuda:0"
        dev_idx = int(device.split(":")[1]) if ":" in device else 0
        dev_idx = min(dev_idx, max(0, torch.cuda.device_count() - 1))
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(dev_idx)
        cuda_ver = torch.version.cuda
        props = torch.cuda.get_device_properties(dev_idx)
        total_vram_mb = props.total_memory / (1024 * 1024)
        total_vram_gb = total_vram_mb / 1024.0
        torch.backends.cudnn.benchmark = True

        # Enable Ampere (RTX 30xx), Ada Lovelace (RTX 40xx) & Blackwell (RTX 50xx) TF32 TensorCore acceleration
        tf32_enabled = False
        if props.major >= 8:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
                tf32_enabled = True
            except Exception:
                pass

        # Live CUDA tensor warmup & verification on target device
        test_a = torch.randn((256, 256), device=device)
        test_b = torch.randn((256, 256), device=device)
        test_c = torch.matmul(test_a, test_b)
        torch.cuda.synchronize(device)
        assert test_c.is_cuda, "Échec de vérification du tenseur sur CUDA !"
        del test_a, test_b, test_c

        print(f"  [GPU] PyTorch CUDA disponible    : OUI ({gpu_count}x GPU détectés | PyTorch {torch.__version__} | CUDA {cuda_ver})")
        print(f"  [GPU] Carte graphique assignée   : {device} -> {gpu_name} ({total_vram_mb:.0f} MB / {total_vram_gb:.1f} Go VRAM)")
        print(f"  [GPU] Accélération TensorCore    : {'TF32 + cuDNN benchmark ACTIVÉS (Ampere/Ada/Blackwell)' if tf32_enabled else 'cuDNN benchmark activé'}")
    else:
        device = "cpu"
        print(f"  [GPU] Avertissement              : CUDA non actif (device={device})")
    print("=" * 84)
    return device, total_ram_gb, total_vram_gb


class ScaledHyperparams:
    """
    Container for auto-scaled training hyperparameters.
    Supports attribute access, dict-like indexing, and flexible tuple unpacking
    (3-tuple for backwards compatibility, 4-tuple, or 5-tuple).
    """

    def __init__(
        self,
        batch_size: int,
        minibatch_size: int,
        exp_buffer_size: int,
        keep_checkpoints: int,
        profile_name: str,
    ):
        self.batch_size = int(batch_size)
        self.minibatch_size = int(minibatch_size)
        self.exp_buffer_size = int(exp_buffer_size)
        self.keep_checkpoints = int(keep_checkpoints)
        self.profile_name = str(profile_name)
        self.hw_profile = self.profile_name

    def _expected_unpack_count(self) -> Optional[int]:
        try:
            frame = inspect.currentframe().f_back.f_back
            for inst in dis.get_instructions(frame.f_code):
                if inst.offset == frame.f_lasti:
                    if inst.opname == "UNPACK_SEQUENCE":
                        return inst.argval
                    elif inst.opname == "UNPACK_EX":
                        return 5
        except Exception:
            pass
        return None

    def __len__(self) -> int:
        n = self._expected_unpack_count()
        if n in (3, 4, 5):
            return n
        return 5

    def __iter__(self):
        n = self._expected_unpack_count()
        if n == 3:
            return iter((self.batch_size, self.minibatch_size, self.profile_name))
        elif n == 4:
            return iter((self.batch_size, self.minibatch_size, self.exp_buffer_size, self.profile_name))
        return iter((
            self.batch_size,
            self.minibatch_size,
            self.exp_buffer_size,
            self.keep_checkpoints,
            self.profile_name,
        ))

    def __getitem__(self, item):
        raw = (
            self.batch_size,
            self.minibatch_size,
            self.exp_buffer_size,
            self.keep_checkpoints,
            self.profile_name,
        )
        if isinstance(item, str):
            try:
                return getattr(self, item)
            except AttributeError:
                raise KeyError(item) from None
        return raw[item]

    def keys(self):
        return ("batch_size", "minibatch_size", "exp_buffer_size", "keep_checkpoints", "profile_name")

    def values(self):
        return (self.batch_size, self.minibatch_size, self.exp_buffer_size, self.keep_checkpoints, self.profile_name)

    def items(self):
        return zip(self.keys(), self.values())

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self):
        return dict(self.items())

    def __eq__(self, other):
        if isinstance(other, ScaledHyperparams):
            return (
                self.batch_size,
                self.minibatch_size,
                self.exp_buffer_size,
                self.keep_checkpoints,
                self.profile_name,
            ) == (
                other.batch_size,
                other.minibatch_size,
                other.exp_buffer_size,
                other.keep_checkpoints,
                other.profile_name,
            )
        if isinstance(other, (tuple, list)):
            if len(other) == 3:
                return (self.batch_size, self.minibatch_size, self.profile_name) == tuple(other)
            elif len(other) == 4:
                return (self.batch_size, self.minibatch_size, self.exp_buffer_size, self.profile_name) == tuple(other)
            elif len(other) == 5:
                return (self.batch_size, self.minibatch_size, self.exp_buffer_size, self.keep_checkpoints, self.profile_name) == tuple(other)
        return False

    def __repr__(self) -> str:
        return (
            f"ScaledHyperparams(batch_size={self.batch_size}, minibatch_size={self.minibatch_size}, "
            f"exp_buffer_size={self.exp_buffer_size}, keep_checkpoints={self.keep_checkpoints}, "
            f"profile_name='{self.profile_name}')"
        )


def auto_scale_server_hyperparams(
    n_proc: int,
    requested_batch_size: int,
    requested_minibatch_size: int,
    total_ram_gb: float,
    total_vram_gb: float,
) -> ScaledHyperparams:
    """
    Dynamically scales PPO batch_size, minibatch_size, ExperienceBuffer, and checkpoint rotation
    for Vast.ai server (30 CPU cores, 32 GB RAM, 16 GB GPU VRAM, 30 GB disk) and high-end rigs.

    Rules:
    - 16 GB VRAM GPU (14.0 - 20.0 GB VRAM): batch_size = 120_000, minibatch_size = 30_000 (100k-150k range).
    - 24 GB+ VRAM GPU (>= 20.0 GB VRAM, RTX 3090/4090): batch_size = 150_000 to 200_000, minibatch_size = 50_000.
    - Low VRAM or CPU: batch_size = 50_000, minibatch_size = 25_000.
    - RAM <= 36 GB: exp_buffer_size = batch_size * 2 (limits RAM consumption to guarantee 0 GB swap).
    - RAM > 36 GB: exp_buffer_size = batch_size * 3.
    - Disk protection (30 GB disk): keep_checkpoints = 1.
    """
    is_high_end_server = (n_proc >= 20) or (total_vram_gb >= 14.0) or (total_ram_gb >= 60.0)

    # 1. Determine batch_size
    if requested_batch_size > 0:
        batch_size = requested_batch_size
    else:
        if total_vram_gb >= 28.0:
            # 32 GB VRAM GPU (RTX 5090 Blackwell)
            batch_size = 250_000 if n_proc >= 48 else 200_000
        elif total_vram_gb >= 20.0:
            batch_size = 200_000 if n_proc >= 32 else 150_000
        elif total_vram_gb >= 14.0:
            # 16 GB VRAM GPU (Vast.ai RTX 4080, RTX 4070 Ti Super, V100, RTX A4000)
            batch_size = 120_000
        elif is_high_end_server:
            batch_size = 100_000
        else:
            batch_size = 50_000

    # 2. Determine minibatch_size
    if requested_minibatch_size > 0:
        minibatch_size = min(batch_size, requested_minibatch_size)
    else:
        if total_vram_gb >= 28.0:
            minibatch_size = min(batch_size, 60_000)
        elif total_vram_gb >= 20.0:
            minibatch_size = min(batch_size, 50_000)
        elif total_vram_gb >= 14.0:
            minibatch_size = min(batch_size, 30_000)
        else:
            minibatch_size = min(batch_size, 25_000)

    # 3. Determine ExperienceBuffer size in RAM
    if total_ram_gb <= 36.0:
        exp_buffer_size = batch_size * 2
    else:
        exp_buffer_size = batch_size * 3

    # 4. Determine checkpoint retention
    keep_checkpoints = 1

    if total_vram_gb >= 28.0:
        hw_label = "SERVEUR ULTRA HAUTE PERFORMANCE BLACKWELL (32 Go VRAM RTX 5090)"
    elif total_vram_gb >= 20.0:
        hw_label = "SERVEUR ULTRA HAUTE PERFORMANCE (24 Go+ VRAM RTX 3090/4090)"
    elif total_vram_gb >= 14.0:
        hw_label = "SERVEUR HAUTE PERFORMANCE (16 Go VRAM GPU / 32 Go RAM)"
    elif is_high_end_server:
        hw_label = "SERVEUR MULTI-COEURS RAM ÉLEVÉE"
    else:
        hw_label = "PC / WORKSTATION STANDARD"

    profile_name = (
        f"{hw_label} ({n_proc} workers | {total_ram_gb:.1f} Go RAM | {total_vram_gb:.1f} Go VRAM | "
        f"batch={batch_size:,} | exp_buffer={exp_buffer_size:,} | keep_chk={keep_checkpoints})"
    )

    return ScaledHyperparams(
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        exp_buffer_size=exp_buffer_size,
        keep_checkpoints=keep_checkpoints,
        profile_name=profile_name,
    )


class ArenaDoneMetricsLogger:
    """Lightweight metrics logger that transports per-arena done flags through shared memory."""

    def collect_metrics(self, state):
        return np.asarray(state, dtype=np.float32)

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        pass


class ContinuousSusuLearner(Learner):
    """
    Extends rlgym-ppo Learner with:
    - Verification that PPO policy & value networks are on the dedicated CUDA GPU.
    - Real-time in-game experience & cumulative wall-clock time calculation on every iteration.
    - Automatic Windows Command Prompt title bar updates.
    - Automatic live TorchScript export to src/susu-model.pt on every checkpoint save.
    - Persistent all-time status tracking into training_status.json and BOOK_KEEPING_VARS.json.
    """

    def __init__(
        self,
        *args,
        export_model_path: Optional[Path] = None,
        status_file_path: Optional[Path] = None,
        tick_skip: int = 8,
        initial_wall_time_seconds: float = 0.0,
        target_vram_gb: float = 0.0,
        **kwargs,
    ):
        if kwargs.get("metrics_logger") is None:
            kwargs["metrics_logger"] = ArenaDoneMetricsLogger()
        super().__init__(*args, **kwargs)
        self.export_model_path = export_model_path
        self.status_file_path = status_file_path
        self.tick_skip = tick_skip
        self.cumulative_wall_time_seconds = float(initial_wall_time_seconds)
        self._last_wall_tick = time.perf_counter()

        # Verify PPO Policy and Value Networks are placed on GPU when CUDA is used
        policy_dev = next(self.ppo_learner.policy.parameters()).device
        critic_dev = next(self.ppo_learner.value_net.parameters()).device
        print(f"[PPO GPU Check] Policy Network device : {policy_dev} | Critic Value Net device : {critic_dev}")
        if "cuda" in str(self.device):
            assert policy_dev.type == "cuda" and critic_dev.type == "cuda", (
                f"Les réseaux PPO devraient être sur CUDA mais sont sur {policy_dev} / {critic_dev} !"
            )
            # Move ExperienceBuffer directly onto GPU VRAM so PPO updates & GAE run 100% in VRAM without CPU RAM copies
            self.experience_buffer.device = self.device
            self.experience_buffer.clear()

            # Hold dedicated GPU VRAM workspace tensor permanently on CUDA with strict <= 3.0 GB limit on 4GB GPUs
            self._vram_workspace = None
            if target_vram_gb > 0:
                try:
                    dev_str = str(self.device)
                    dev_idx = int(dev_str.split(":")[1]) if ":" in dev_str else 0
                    dev_idx = min(dev_idx, max(0, torch.cuda.device_count() - 1))
                    props = torch.cuda.get_device_properties(dev_idx)
                    total_gpu_gb = props.total_memory / (1024.0 ** 3)
                    if total_gpu_gb <= 4.5:
                        # 4 GB GPU (RTX 3050 Laptop): strict 3.0 GB ceiling enforced via allocator
                        hard_vram_limit_gb = min(3.0, float(target_vram_gb))
                        torch.cuda.set_per_process_memory_fraction(min(0.75, hard_vram_limit_gb / total_gpu_gb), dev_idx)
                        max_safe_gb = max(0.5, hard_vram_limit_gb - 0.55)  # 550 MB headroom for PPO activations & ExperienceBuffer
                    else:
                        hard_vram_limit_gb = float(target_vram_gb)
                        max_safe_gb = total_gpu_gb * 0.78
                    alloc_gb = min(hard_vram_limit_gb, max_safe_gb)
                    already_gb = torch.cuda.memory_allocated(dev_idx) / (1024.0 ** 3)
                    rem_gb = max(0.0, alloc_gb - already_gb)
                    n_floats = int((rem_gb * (1024 ** 3)) // 4)
                    if n_floats > 0:
                        self._vram_workspace = torch.zeros(n_floats, dtype=torch.float32, device=self.device)
                        actual_gb = torch.cuda.memory_allocated(0) / (1024.0 ** 3)
                        print(f"[GPU VRAM Turbo] {actual_gb:.2f} Go de VRAM verrouillés sur {self.device} (Plafond strict ≤ 3.0 Go garanti)")
                except Exception as e:
                    print(f"[GPU VRAM Turbo] Note allocation VRAM : {e}")

        # Install Turbo GPU & IPC hooks on BatchedAgentManager and DiscreteFF policy
        self._install_turbo_hooks()

    @torch.no_grad()
    def add_new_experience(self, experience):
        """
        GPU-accelerated experience ingestion + GAE without calling torch.cuda.empty_cache()
        (so the 12 GB GPU VRAM workspace & ExperienceBuffer remain locked in VRAM).
        Also pre-dispatches the next wave of actions to workers BEFORE GAE & PPO learning
        so CPU workers continue stepping at 100% in parallel while the GPU trains!
        """
        # Immediately dispatch pending actions to CPU workers so they compute while GPU runs GAE + PPO!
        try:
            self.agent._send_actions()
        except Exception:
            pass

        from rlgym_ppo.util import torch_functions
        states, actions, log_probs, rewards, next_states, dones, truncated = experience
        value_net = self.ppo_learner.value_net

        # Standardize states on GPU if standardize_obs is enabled
        if self.agent.standardize_obs and self.agent.obs_stats is not None:
            obs_mean = torch.as_tensor(self.agent.obs_stats.mean[0], dtype=torch.float32, device=self.device)
            obs_std = torch.as_tensor(self.agent.obs_stats.std[0], dtype=torch.float32, device=self.device) + 1e-8
            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            states_t = torch.clamp((states_t - obs_mean) / obs_std, -5.0, 5.0)
            last_next_t = torch.as_tensor(next_states[-1:], dtype=torch.float32, device=self.device)
            last_next_t = torch.clamp((last_next_t - obs_mean) / obs_std, -5.0, 5.0)
            val_inp = torch.cat([states_t, last_next_t], dim=0)
            states = states_t
        else:
            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            last_next_t = torch.as_tensor(next_states[-1:], dtype=torch.float32, device=self.device)
            val_inp = torch.cat([states_t, last_next_t], dim=0)
            states = states_t

        val_preds = value_net(val_inp).cpu().flatten().tolist()

        ret_std = self.return_stats.std[0] if self.standardize_returns else None
        value_targets, advantages, returns = torch_functions.compute_gae(
            rewards,
            dones,
            truncated,
            val_preds,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
            return_std=ret_std,
        )

        if self.standardize_returns:
            n_to_increment = min(self.max_returns_per_stats_increment, len(returns))
            self.return_stats.increment(returns[:n_to_increment], n_to_increment)

        self.experience_buffer.submit_experience(
            states,
            actions,
            log_probs,
            rewards,
            next_states,
            dones,
            truncated,
            value_targets,
            advantages,
        )

    def _install_turbo_hooks(self):
        import selectors
        from numpy import frombuffer, prod
        from rlgym_ppo.batched_agents import BatchedTrajectory, comm_consts

        class MultiArenaBatchedTrajectory(BatchedTrajectory):
            """
            Supports per-agent `done` vectors (from MultiArenaSimEnv) so that when an individual
            arena inside a multi-arena worker scores a goal or resets, GAE (`compute_gae`) properly
            receives `done = 1.0` for that arena's agents instead of leaking returns across episodes.
            """

            def update(self):
                if (
                    self.state is not None
                    and self.action is not None
                    and self.log_prob is not None
                    and self.reward is not None
                    and self.next_state is not None
                    and self.done is not None
                    and self.truncated is not None
                ):
                    if type(self.reward) not in (list, tuple, np.ndarray):
                        self.reward = [self.reward]
                    self.complete_timesteps.append(
                        (
                            self.state,
                            self.action,
                            self.log_prob,
                            self.reward,
                            self.next_state,
                            self.done,
                            self.truncated,
                        )
                    )
                    done = self.done
                    self.state = None
                    self.action = None
                    self.log_prob = None
                    self.reward = None
                    self.next_state = None
                    self.done = None
                    if isinstance(done, (list, tuple, np.ndarray)):
                        return False
                    if done:
                        return True
                return False

            def get_all(self):
                if len(self.complete_timesteps) == 0:
                    return []
                trajectories = []
                n_trajectories = len(self.complete_timesteps[0][3])
                zeroes = np.zeros_like(self.complete_timesteps[0][0][0])
                for i in range(n_trajectories):
                    states, rewards, actions, log_probs, next_states, dones, truncateds = [], [], [], [], [], [], []
                    for timestep in self.complete_timesteps:
                        state, action, log_prob, reward, next_trajectory_state, done, truncated = timestep
                        next_state = zeroes if i >= len(next_trajectory_state) else next_trajectory_state[i]
                        if isinstance(done, (list, tuple, np.ndarray)) and len(done) > i:
                            agent_done = 1.0 if float(done[i]) > 0.5 else 0.0
                        else:
                            agent_done = float(done)
                        states.append(state[i])
                        actions.append(action[i])
                        log_probs.append(log_prob[i])
                        next_states.append(next_state)
                        dones.append(agent_done)
                        rewards.append(reward[i])
                        truncateds.append(truncated)
                    trajectories.append([states, actions, log_probs, rewards, next_states, dones, truncateds])
                self.complete_timesteps = []
                return trajectories

        policy = self.ppo_learner.policy
        mgr = self.agent
        mgr.steps_per_obs_stats_increment = 100
        mgr.trajectory_map = [MultiArenaBatchedTrajectory() for _ in range(len(mgr.trajectory_map))]
        dev = self.device

        @torch.no_grad()
        def fast_get_action(obs, deterministic=False):
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            if mgr.standardize_obs and mgr.obs_stats is not None:
                obs_mean = torch.as_tensor(mgr.obs_stats.mean[0], dtype=torch.float32, device=dev)
                obs_std = torch.as_tensor(mgr.obs_stats.std[0], dtype=torch.float32, device=dev) + 1e-8
                obs = torch.clamp((obs - obs_mean) / obs_std, -5.0, 5.0)
            probs = policy.model(obs).view(-1, policy.n_actions)
            probs = torch.clamp(probs, min=1e-11, max=1.0)
            if deterministic:
                return probs.cpu().numpy().argmax(), 0
            action = torch.multinomial(probs, 1, True)
            log_prob = torch.log(probs).gather(-1, action)
            combined = torch.cat([action.float(), log_prob], dim=-1).cpu()
            return combined[:, 0], combined[:, 1]

        policy.get_action = fast_get_action

        def fast_collect_response(proc_id, parent_end, shm_view, collected_metrics):
            available_data = parent_end.recv(8192)
            header = frombuffer(available_data, dtype=np.float32, count=comm_consts.HEADER_LEN)
            if header[0] != comm_consts.ENV_STEP_DATA_HEADER[0]:
                return 0

            prev_n_agents = int(shm_view[0])
            done = shm_view[1]
            truncated = shm_view[2]
            n_elements_in_state_shape = int(shm_view[3])
            metrics_len = int(shm_view[4])

            if metrics_len != 0:
                metrics_shape = [int(d) for d in shm_view[5 : 5 + metrics_len]]
                n_metrics = prod(metrics_shape)
            else:
                metrics_shape = (0,)
                n_metrics = 0

            state_shape_start = 5 + metrics_len
            if n_elements_in_state_shape == 1:
                state_shape = [1, int(shm_view[state_shape_start])]
            else:
                state_shape = [
                    int(arg)
                    for arg in shm_view[state_shape_start : state_shape_start + n_elements_in_state_shape]
                ]

            rew_start = state_shape_start + n_elements_in_state_shape
            rew_end = rew_start + prev_n_agents

            shm_shapes = mgr.shm_shapes[proc_id]
            if shm_shapes is None or shm_shapes != (metrics_shape, state_shape, prev_n_agents):
                mgr.shm_shapes[proc_id] = (metrics_shape, state_shape, prev_n_agents)
                rews = np.reshape(shm_view[rew_start:rew_end], (rew_end - rew_start,))
                metrics = np.reshape(shm_view[rew_end : rew_end + n_metrics], metrics_shape)
                obs = np.reshape(
                    shm_view[rew_end + n_metrics : rew_end + n_metrics + prod(state_shape)],
                    state_shape,
                )
                mgr.shm_cache[proc_id] = (rews, metrics, obs)

            rews, metrics, next_observation = mgr.shm_cache[proc_id]
            next_observation = next_observation.copy()

            if mgr.standardize_obs:
                if mgr.steps_since_obs_stats_update > mgr.steps_per_obs_stats_increment:
                    mgr.obs_stats.increment(next_observation, state_shape[0])
                    mgr.steps_since_obs_stats_update = 0
                else:
                    mgr.steps_since_obs_stats_update += 1

            has_per_agent_dones = (n_metrics == prev_n_agents)
            if prev_n_agents > 1:
                n_collected = prev_n_agents
                ep_r = mgr.ep_rews[proc_id]
                for i in range(prev_n_agents):
                    if i >= len(ep_r):
                        ep_r.append(float(rews[i]))
                    else:
                        ep_r[i] += float(rews[i])
                    if has_per_agent_dones and metrics[i] > 0.5:
                        if mgr.average_reward is None:
                            mgr.average_reward = ep_r[i]
                        else:
                            mgr.average_reward = mgr.average_reward * 0.9 + ep_r[i] * 0.1
                        ep_r[i] = 0.0
            else:
                n_collected = 1
                mgr.ep_rews[proc_id][0] += float(rews[0])

            if done or truncated:
                if mgr.average_reward is None:
                    mgr.average_reward = mgr.ep_rews[proc_id][0]
                else:
                    for ep_rew in mgr.ep_rews[proc_id]:
                        mgr.average_reward = mgr.average_reward * 0.9 + ep_rew * 0.1
                mgr.ep_rews[proc_id] = [0]

            if proc_id not in mgr.current_pids:
                mgr.current_pids.append(proc_id)

            mgr.next_obs[proc_id] = next_observation
            traj = mgr.trajectory_map[proc_id]
            traj.reward = rews.tolist()
            traj.next_state = next_observation
            traj.done = metrics.copy() if has_per_agent_dones else done
            traj.truncated = truncated

            if state_shape[0] != prev_n_agents:
                mgr.completed_trajectories.append(traj)
                mgr.trajectory_map[proc_id] = MultiArenaBatchedTrajectory()

            return n_collected

        def fast_collect_responses(n_obs_per_inference):
            n_collected = 0
            mgr.current_pids = []
            collected_metrics = []
            target = mgr.min_inference_size

            while True:
                timeout = 0.0 if n_collected >= target else None
                ready = mgr.selector.select(timeout=timeout)
                if not ready:
                    break
                for key, event in ready:
                    if not (event & selectors.EVENT_READ):
                        continue
                    parent_end, fd, events, proc_id = key
                    process, parent_end, child_endpoint, shm_view = mgr.processes[proc_id]
                    n_collected += fast_collect_response(
                        proc_id, parent_end, shm_view, collected_metrics
                    )
            return collected_metrics, n_collected

        def fast_sync_trajectories():
            for proc_id in mgr.current_pids:
                trajectory = mgr.trajectory_map[proc_id]
                if trajectory.update():
                    mgr.completed_trajectories.append(trajectory)
                    mgr.trajectory_map[proc_id] = MultiArenaBatchedTrajectory()

        def fast_collect_timesteps(n):
            t1 = time.perf_counter()
            states, actions, log_probs, rewards, next_states, dones, truncated = [], [], [], [], [], [], []
            n_collected = 0
            collected_metrics = []

            while n_collected < n:
                mgr._send_actions()
                col_m, n_col_pass = mgr._collect_responses(mgr.min_inference_size)
                n_collected += n_col_pass
                for proc_id in mgr.current_pids:
                    if mgr.next_obs[proc_id] is not None:
                        mgr.current_obs[proc_id] = mgr.next_obs[proc_id]
                        mgr.next_obs[proc_id] = None
                mgr._sync_trajectories()

            for proc_id, trajectory in enumerate(mgr.trajectory_map):
                mgr.completed_trajectories.append(trajectory)
                mgr.trajectory_map[proc_id] = MultiArenaBatchedTrajectory()

            # Pre-dispatch next actions to all ready workers BEFORE trajectory flattening, GAE, and PPO learning!
            mgr._send_actions()

            for trajectory in mgr.completed_trajectories:
                trajectories = trajectory.get_all()
                if not trajectories:
                    continue
                for traj in trajectories:
                    t_states, t_actions, t_log_probs, t_rewards, t_next_states, t_dones, t_truncated = traj
                    t_truncated[-1] = 1 if t_dones[-1] == 0 else 0
                    states.extend(t_states)
                    actions.extend(t_actions)
                    log_probs.extend(t_log_probs)
                    rewards.extend(t_rewards)
                    next_states.extend(t_next_states)
                    dones.extend(t_dones)
                    truncated.extend(t_truncated)

            mgr.cumulative_timesteps += n_collected
            t2 = time.perf_counter()
            mgr.completed_trajectories = []

            return (
                (
                    np.asarray(states, dtype=np.float32),
                    np.asarray(actions, dtype=np.float32),
                    np.asarray(log_probs, dtype=np.float32),
                    np.asarray(rewards, dtype=np.float32),
                    np.asarray(next_states, dtype=np.float32),
                    np.asarray(dones, dtype=np.float32),
                    np.asarray(truncated, dtype=np.float32),
                ),
                collected_metrics,
                n_collected,
                t2 - t1,
            )

        mgr._collect_responses = fast_collect_responses
        mgr._sync_trajectories = fast_sync_trajectories
        mgr.collect_timesteps = fast_collect_timesteps

    def update_wall_clock(self) -> float:
        now = time.perf_counter()
        delta = max(0.0, now - self._last_wall_tick)
        self.cumulative_wall_time_seconds += delta
        self._last_wall_tick = now
        return self.cumulative_wall_time_seconds

    def _learn(self):
        try:
            kb = KBHit() if sys.stdin is not None and sys.stdin.isatty() else None
        except Exception:
            kb = None
        self._last_wall_tick = time.perf_counter()
        print("Commandes clavier : (Ctrl+C) ou (q) sauvegarder, afficher le temps total et quitter | (p) pause | (c) sauvegarder\n")

        while self.agent.cumulative_timesteps < self.timestep_limit:
            epoch_start = time.perf_counter()
            report = {}

            experience, collected_metrics, steps_collected, collection_time = self.agent.collect_timesteps(
                self.ts_per_epoch
            )

            if self.metrics_logger is not None:
                self.metrics_logger.report_metrics(collected_metrics, self.wandb_run, self.agent.cumulative_timesteps)

            self.add_new_experience(experience)

            ppo_report = self.ppo_learner.learn(self.experience_buffer)
            epoch_stop = time.perf_counter()
            epoch_time = max(1e-6, epoch_stop - epoch_start)

            # Update cumulative real wall-clock training time
            self.update_wall_clock()

            report.update(ppo_report)
            if self.epoch < 1:
                report["Value Function Loss"] = np.nan

            report["Cumulative Timesteps"] = self.agent.cumulative_timesteps
            report["Total Iteration Time"] = epoch_time
            report["Timesteps Collected"] = steps_collected
            report["Timestep Collection Time"] = collection_time
            report["Timestep Consumption Time"] = epoch_time - collection_time
            report["Collected Steps per Second"] = steps_collected / max(1e-6, collection_time)
            report["Overall Steps per Second"] = steps_collected / epoch_time
            report["PPO Device"] = str(self.device)

            self.ts_since_last_save += steps_collected
            if self.agent.average_reward is not None:
                report["Policy Reward"] = self.agent.average_reward
            else:
                report["Policy Reward"] = np.nan

            # In-game experience & wall-clock duration computation
            in_game_str, exp_dict = format_in_game_time(self.agent.cumulative_timesteps, tick_skip=self.tick_skip)
            wall_str, _ = format_duration_seconds(self.cumulative_wall_time_seconds)
            report["TEMPS TOTAL SIMULE"] = in_game_str
            report["TEMPS REEL CUMULE"] = wall_str
            report["Matchs 5min Joues"] = f"{exp_dict['total_matches_5min']:.1f} matchs"
            report["Annees Jouees"] = f"{exp_dict['years']:.4f} ans"
            report["Jours Joues"] = f"{exp_dict['total_days']:.2f} jours"

            # Update Windows terminal window title
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetConsoleTitleW(
                        f"susu_bot RocketSim C++ & CUDA ({self.device}) - Simule: {in_game_str} | SPS: {steps_collected / max(1e-6, collection_time):.0f} | Steps: {self.agent.cumulative_timesteps:,}"
                    )
                except Exception:
                    pass

            # Log to wandb and print to the console
            reporting.report_metrics(
                loggable_metrics=report,
                debug_metrics=None,
                wandb_run=self.wandb_run,
            )

            now_str = datetime.now().strftime("%H:%M:%S")
            sps = steps_collected / max(1e-6, collection_time)
            print(
                f"[{now_str}] 🚀 RocketSim C++ ({sps:,.0f} steps/s) + GPU ({self.device}) | "
                f"🕒 Total Simulé : {in_game_str} ({exp_dict['total_matches_5min']:.1f} matchs) | Réel : {wall_str}"
            )

            report.clear()
            ppo_report.clear()

            if kb is not None and kb.kbhit():
                c = kb.getch()
                if c == "\x03":
                    raise KeyboardInterrupt
                if c == "p":
                    print("En pause. Appuyez sur n'importe quelle touche pour reprendre...")
                    while True:
                        if kb is not None and kb.kbhit():
                            break
                    self._last_wall_tick = time.perf_counter()
                if c in ("c", "q"):
                    self.save(self.agent.cumulative_timesteps)
                if c == "q":
                    raise KeyboardInterrupt
                if c in ("c", "p"):
                    print("Reprise de l'entrainement...\n")

            if self.ts_since_last_save >= self.save_every_ts:
                self.save(self.agent.cumulative_timesteps)
                self.ts_since_last_save = 0

            self.epoch += 1

    def save(self, cumulative_timesteps):
        self.update_wall_clock()
        # Defensively remove non-digit files in checkpoints directory to avoid crash in rlgym-ppo's [int(arg) for arg in os.listdir]
        try:
            if os.path.isdir(self.checkpoints_save_folder):
                for item in os.listdir(self.checkpoints_save_folder):
                    if not item.isdigit():
                        item_path = os.path.join(self.checkpoints_save_folder, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
        except Exception:
            pass
        super().save(cumulative_timesteps)

        folder_path = Path(self.checkpoints_save_folder) / str(cumulative_timesteps)
        policy_path = folder_path / "PPO_POLICY.pt"
        bkv_path = folder_path / "BOOK_KEEPING_VARS.json"

        if bkv_path.exists():
            try:
                with open(bkv_path, "r", encoding="utf-8") as f:
                    bkv = json.load(f)
                bkv["cumulative_wall_time_seconds"] = round(self.cumulative_wall_time_seconds, 2)
                with open(bkv_path, "w", encoding="utf-8") as f:
                    json.dump(bkv, f, indent=4)
            except Exception:
                pass

        # 1. Live TorchScript model export (always traced on CPU for RLBot compatibility)
        if self.export_model_path is not None and policy_path.exists():
            try:
                export_checkpoint(policy_path, self.export_model_path)
            except Exception as e:
                print(f"[ContinuousSusuLearner] Erreur lors de l'auto-export TorchScript: {e}")

        # 2. Cumulative in-game & wall-clock experience calculation
        in_game_str, exp_dict = format_in_game_time(cumulative_timesteps, tick_skip=self.tick_skip)
        wall_str, wall_dict = format_duration_seconds(self.cumulative_wall_time_seconds)

        # 3. Status persistence to training_status.json (100% dynamic relative paths)
        if self.status_file_path is not None:
            try:
                avg_rew = float(self.agent.average_reward) if self.agent.average_reward is not None else 0.0
                status_data = {
                    "cumulative_timesteps": int(cumulative_timesteps),
                    "in_game_experience": in_game_str,
                    "in_game_years": round(exp_dict["years"], 4),
                    "in_game_days": round(exp_dict["total_days"], 2),
                    "in_game_hours": round(exp_dict["total_hours"], 2),
                    "in_game_seconds": round(exp_dict["total_seconds"], 1),
                    "total_matches_5min": round(exp_dict["total_matches_5min"], 1),
                    "cumulative_wall_time_seconds": round(self.cumulative_wall_time_seconds, 2),
                    "cumulative_wall_time_formatted": wall_str,
                    "cumulative_wall_time_hours": round(wall_dict["total_hours"], 2),
                    "policy_average_reward": round(avg_rew, 4),
                    "ppo_device": str(self.device),
                    "epoch": int(self.epoch),
                    "latest_checkpoint": _to_rel_path(folder_path),
                    "exported_model": _to_rel_path(self.export_model_path) if self.export_model_path else None,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(self.status_file_path, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, indent=4)
            except Exception as e:
                print(f"[ContinuousSusuLearner] Erreur sauvegarde status JSON: {e}")

        # 4. Display live experience banner
        print("\n" + "=" * 80)
        print("🎮 [susu_bot / Susu-v0 - ROCKETSIM C++ & rlgym-sim + GPU CUDA]")
        print(f"🕒 TEMPS TOTAL DE JEU SIMULÉ (Cumulé) : {in_game_str}")
        print(f"📅 Années / Jours de match cumulés    : {exp_dict['years']:.4f} ans ({exp_dict['total_days']:.2f} jours | {exp_dict['total_matches_5min']:.1f} matchs de 5 min)")
        print(f"⏱️ TEMPS RÉEL D'ENTRAÎNEMENT CUMULÉ   : {wall_str} ({self.cumulative_wall_time_seconds:.1f}s au total)")
        print(f"📊 Timesteps totaux cumulés           : {cumulative_timesteps:,} steps")
        if self.agent.average_reward is not None:
            print(f"⭐ Récompense moyenne                 : {self.agent.average_reward:.3f}")
        if self.export_model_path is not None:
            print(f"💾 Modèle RLBot mis à jour            : {self.export_model_path}")
        print("=" * 80 + "\n")


def print_all_time_summary_banner(
    total_timesteps: int,
    session_timesteps: int,
    total_wall_seconds: float,
    session_wall_seconds: float,
    out_model: Path,
    status_file: Path,
    tick_skip: int = 8,
):
    total_sim_str, total_sim = format_in_game_time(total_timesteps, tick_skip=tick_skip)
    sess_sim_str, sess_sim = format_in_game_time(session_timesteps, tick_skip=tick_skip)
    total_wall_str, total_wall = format_duration_seconds(total_wall_seconds)
    sess_wall_str, _ = format_duration_seconds(session_wall_seconds)

    print("\n" + "#" * 84)
    print("🛑 ARRÊT DE L'ENTRAÎNEMENT (Ctrl+C) — BILAN CUMULÉ DE TOUTE L'HISTOIRE DU BOT")
    print("#" * 84)
    print("🏆 1. TEMPS TOTAL D'ENTRAÎNEMENT QUE LE BOT A ACCUMULÉ (TOUTES SÉANCES CONFONDUES) :")
    print(f"   • Temps total de match simulé (RLGym) : {total_sim_str}")
    print(f"   • Équivalent en années / jours / h    : {total_sim['years']:.4f} ans | {total_sim['total_days']:.2f} jours | {total_sim['total_hours']:.2f} heures")
    print(f"   • Total de matchs de 5 min disputés   : {total_sim['total_matches_5min']:,.1f} matchs complets de 5 minutes")
    print(f"   • Temps réel d'entraînement cumulé    : {total_wall_str} ({total_wall['total_hours']:.2f} heures réelles)")
    print(f"   • Total des pas d'apprentissage       : {total_timesteps:,} timesteps")
    print("-" * 84)
    print("📈 2. PROGRESSION AJOUTÉE DURANT CETTE SÉANCE UNIQUEMENT :")
    print(f"   • Durée réelle de cette séance        : {sess_wall_str} ({session_wall_seconds:.1f}s)")
    print(f"   • Temps de jeu simulé gagné           : +{sess_sim_str} (+{sess_sim['total_matches_5min']:.1f} matchs de 5 min)")
    print(f"   • Pas effectués durant cette séance   : +{session_timesteps:,} timesteps")
    print("-" * 84)
    print(f"💾 Modèle RLBot exporté et prêt : {out_model}")
    print(f"📄 Statut cumulé sauvegardé     : {status_file}")
    print("#" * 84 + "\n")


def interactive_training_menu() -> str:
    print("=" * 80)
    print("🏋️ SUSU_BOT — ENTRAÎNEMENT PAR RENFORCEMENT (RocketSim C++ + RLGym-PPO)")
    print("=" * 80)
    print("Choisissez le programme d'entraînement spécialisé :")
    print("  [1] Aérienne      (Air dribbles, Wall pops, Flip resets, Keep-up, Boost illimité)")
    print("  [2] Flick         (Contrôle au toit, Musty flick, Enroulés, Post-flick catch)")
    print("  [3] Tir Précis    (Frappes puissantes, Sniping lucarnes, Balles rebondissantes)")
    print("  [4] Déplacement   (Supersonique, Atterrissages 4 roues, Fast rotations)")
    print("  [5] 1v1 Normal    (Match Soccar complet, Kick-offs, Duel compétitif)")
    print("  [0] Quitter")
    print("=" * 80)
    try:
        choice = input("Votre choix (1-5, ou Entrée pour 1v1 Normal) : ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)

    mapping = {
        "1": "aerial",
        "2": "flick",
        "3": "shooting",
        "4": "movement",
        "5": "mix",
        "": "mix",
        "0": "quit",
        "q": "quit",
    }
    res = mapping.get(choice, "mix")
    if res == "quit":
        print("Entraînement annulé.")
        sys.exit(0)
    return res


def parse_args():
    project_root = Path(__file__).resolve().parent
    env_ckpt_dir = os.environ.get("SUSU_BOT_CKPT_DIR", "").strip()
    default_checkpoints = Path(env_ckpt_dir) if env_ckpt_dir else (project_root / "training" / "checkpoints")
    default_model = project_root / "src" / "susu-model.pt"
    default_status = project_root / "training" / "training_status.json"
    max_cpu_cores = os.cpu_count() or 12

    parser = argparse.ArgumentParser(
        description="Train susu_bot (Susu-v0) with C++ RocketSim + rlgym-sim + PyTorch CUDA GPU"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        choices=["aerial", "flick", "movement", "shooting", "mix", "1v1", ""],
        help="Alias for --specialization ('aerial', 'flick', 'movement', 'shooting', '1v1', 'mix')",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=0,
        help="Number of parallel RocketSim C++ environments (0 = auto-detect with RAM-safe ceiling: 12-14 workers max to prevent swap)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "auto",
        help="PyTorch device for PPO updates ('cuda:0', 'gpu', 'auto', or 'cpu')",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="Total training timesteps to add in this session (0 = continuous training without end)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=True,
        help="Train endlessly until Ctrl+C",
    )
    parser.add_argument(
        "--match_duration",
        type=float,
        default=300.0,
        help="Simulated match timeout in seconds (default: 300.0 = 5 minutes)",
    )
    parser.add_argument(
        "--no_touch_timeout",
        type=float,
        default=30.0,
        help="No-touch timeout in seconds to reset stagnant states quickly",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="rocketsim",
        choices=["rocketsim", "rlgym_sim", "sim", "rlgym", "live"],
        help="Training engine: 'rocketsim' (C++ RocketSim + rlgym-sim, default)",
    )
    parser.add_argument(
        "--curriculum_stage",
        type=int,
        default=-1,
        help="Curriculum stage for rlgym fallback",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(default_checkpoints),
        help="Directory to save PPO checkpoints",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_model),
        help="Destination path for exported TorchScript model",
    )
    parser.add_argument(
        "--status_file",
        type=str,
        default=str(default_status),
        help="JSON file path for live cumulative training status",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100_000,
        help="Save checkpoint and auto-export model every N timesteps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="PPO batch size / ts per iteration (0 = auto-scale: 50k on PC, 120k on 16GB GPU, 150k-200k on RTX 3090/4090)",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=0,
        help="PPO minibatch size on GPU (0 = auto-scale: 25k on 4-12GB GPU, 50k on 24GB RTX 3090/4090)",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=3,
        help="Number of PPO epochs per iteration on GPU",
    )
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=1,
        help="Maximum number of rotating checkpoints to keep on disk (default: 3 to protect 30GB disk)",
    )
    parser.add_argument(
        "--instance_launch_delay",
        type=float,
        default=0.1,
        help="Delay in seconds between spawning worker processes to eliminate IPC pipe/socket contention (default: 0.1s)",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="Entropy bonus",
    )
    parser.add_argument(
        "--arenas_per_worker",
        type=int,
        default=0,
        help="Number of simultaneous 1v1 C++ RocketSim arenas inside each CPU worker process (0 = auto: 6 on servers, 4 on PC)",
    )
    parser.add_argument(
        "--target_vram_gb",
        type=float,
        default=float(os.environ.get("SUSU_BOT_VRAM_GB", "0.0")),
        help="Target GPU VRAM workspace cache in GB (0 = auto: 2.45 GB on 4GB PC, 12 GB on 16GB server)",
    )
    parser.add_argument(
        "--mechanic",
        type=str,
        default=os.environ.get("SUSU_BOT_MECHANIC", "mix"),
        help="Target mechanic scenario ('mix', 'kickoff', 'dribble_flick', 'musty_ground', 'musty_air', 'air_keepup', 'air_dribble', 'air_reset', 'air_takeoff', 'normal')",
    )
    parser.add_argument(
        "--force_role",
        type=str,
        default=os.environ.get("SUSU_BOT_FORCE_ROLE", ""),
        help="Force specialist role ('kickoff_master', 'flick_dribble', 'musty', 'air_dribble', 'air_reset_keepup', 'all_rounder')",
    )
    parser.add_argument(
        "--specialization",
        type=str,
        default=os.environ.get("SUSU_BOT_SPECIALIZATION", "mix"),
        choices=["aerial", "flick", "movement", "shooting", "mix"],
        help="Focus specialization: 'aerial' (infinite boost, air dribble, keep-up), 'flick' (roof dribble, pop, catch), 'movement' (recovery, speed, rotations), 'shooting' (bouncing strikes, power shots), 'mix' (all)",
    )
    parser.add_argument(
        "--infinite_boost",
        action="store_true",
        default=bool(os.environ.get("SUSU_BOT_INFINITE_BOOST", "0").strip().lower() in ("1", "true", "yes")),
        help="Enable infinite 100%% boost (always active for 'aerial' specialization)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--watchdog",
        action="store_true",
        default=True,
        help="Auto-recover and resume on any transient error",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode:
        args.specialization = "mix" if args.mode == "1v1" else args.mode
    if len(sys.argv) == 1:
        args.specialization = interactive_training_menu()
    if args.mechanic:
        os.environ["SUSU_BOT_MECHANIC"] = str(args.mechanic).strip().lower()
    if args.force_role:
        os.environ["SUSU_BOT_FORCE_ROLE"] = str(args.force_role).strip().lower()
    if args.specialization:
        os.environ["SUSU_BOT_SPECIALIZATION"] = str(args.specialization).strip().lower()
    if args.specialization == "aerial" or args.infinite_boost:
        args.infinite_boost = True
        os.environ["SUSU_BOT_INFINITE_BOOST"] = "1"

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_model = Path(args.model_path).resolve()
    status_file = Path(args.status_file).resolve()
    status_file.parent.mkdir(parents=True, exist_ok=True)

    total_ram_gb = get_effective_ram_gb()
    detected_cores = os.cpu_count() or 12
    n_proc = calculate_safe_worker_count(
        detected_cores=detected_cores,
        total_ram_gb=total_ram_gb,
        requested_n_proc=args.n_proc,
    )
    if args.arenas_per_worker > 0:
        arenas_per_worker = args.arenas_per_worker
    else:
        if total_ram_gb >= 120.0 and detected_cores >= 64:
            arenas_per_worker = 6  # Maximisation totale de la RAM sur serveurs 258 Go
        elif total_ram_gb >= 50.0 and detected_cores >= 32:
            arenas_per_worker = 4
        elif detected_cores >= 20:
            arenas_per_worker = 4
        else:
            arenas_per_worker = 2
    total_agents = n_proc * 2 * arenas_per_worker
    min_inference = max(1, int(round(total_agents * 0.45)))
    shm_buffer_size = max(16384, 4096 * arenas_per_worker)

    # Verify CPU cores, RAM, C++ RocketSim, rlgym-sim, and PyTorch CUDA GPU (RTX 3050 / 3090 / 4090)
    verified_device, total_ram_gb, total_vram_gb = verify_hardware_and_cuda_gpu(
        n_proc=n_proc, requested_device=args.device, total_ram_gb=total_ram_gb
    )

    # Auto-scale PPO batch_size and minibatch_size for Vast.ai (30 cores, 32GB RAM, 16GB VRAM) and high-end servers
    scaled_params = auto_scale_server_hyperparams(
        n_proc=n_proc,
        requested_batch_size=args.batch_size,
        requested_minibatch_size=args.minibatch_size,
        total_ram_gb=total_ram_gb,
        total_vram_gb=total_vram_gb,
    )
    batch_size = scaled_params.batch_size
    minibatch_size = scaled_params.minibatch_size
    exp_buffer_size = scaled_params.exp_buffer_size
    keep_checkpoints = args.keep_checkpoints if args.keep_checkpoints != 1 else scaled_params.keep_checkpoints
    hw_profile = scaled_params.profile_name

    # Auto-scale target VRAM budget (max 3.0 GB total on 4GB GPUs)
    target_vram_gb = args.target_vram_gb
    if target_vram_gb <= 0:
        if total_vram_gb <= 4.5:
            target_vram_gb = 2.45  # Enforces strictly <= 3.0 GB with activations
        elif total_vram_gb >= 14.0:
            target_vram_gb = 12.0
        else:
            target_vram_gb = min(3.0, total_vram_gb * 0.65)
    elif total_vram_gb <= 4.5 and target_vram_gb > 3.0:
        target_vram_gb = 2.45

    # Initial export of latest checkpoint if present
    initial_checkpoint = find_latest_checkpoint(save_dir)
    initial_loaded_steps = 0
    if initial_checkpoint is not None:
        upgrade_checkpoint_obs_dim(initial_checkpoint.parent, target_in_dim=142)
        try:
            initial_loaded_steps = int(initial_checkpoint.parent.name)
        except ValueError:
            initial_loaded_steps = 0
        export_checkpoint(initial_checkpoint, out_model)

    initial_wall_time = load_persisted_wall_time(status_file, save_dir, loaded_steps=initial_loaded_steps)
    session_start_wall = time.perf_counter()

    if args.timesteps > 0:
        target_timesteps = initial_loaded_steps + args.timesteps if args.timesteps < initial_loaded_steps else args.timesteps
        run_endlessly = False
    else:
        target_timesteps = 100_000_000_000_000
        run_endlessly = True

    init_sim_str, init_sim = format_in_game_time(initial_loaded_steps)
    init_wall_str, _ = format_duration_seconds(initial_wall_time)

    print("⚙️ Configuration de l'entraînement RLGym C++ (RocketSim + rlgym-sim + CUDA) :")
    print(f"  - Profil Matériel       : {hw_profile}")
    print(f"  - Simulation Physique   : C++ RocketSim ({n_proc} workers × {arenas_per_worker} arènes = {n_proc * arenas_per_worker} matchs / {total_agents} voitures en parallèle)")
    print(f"  - Spécialisation Bot    : {args.specialization.upper()} {'(BOOST ILLIMITÉ ACTIF)' if args.infinite_boost else '(Boost standard)'}")
    print("  - Action Parser         : LookupAction discret (90 actions Susu)")
    print("  - Observation Builder   : SusuSimObsBuilder (142 floats, 20 horizons de 0.05s à 2.50s)")
    print("  - Système de Récompenses: Buts + Spécialistes (Bouncing contest, Wall aerials, Flick+Catch, Musty, Flip Reset, Anti-Slow Kickoff)")
    print(f"  - Accélération PPO GPU  : {verified_device} ({torch.cuda.get_device_name(0) if 'cuda' in verified_device else 'CPU'}) | Cache VRAM: {args.target_vram_gb:.1f} Go")
    print(f"  - Historique déjà acquis: {init_sim_str} ({init_sim['total_matches_5min']:.1f} matchs de 5 min | Réel: {init_wall_str})")
    print(f"  - Batch / Minibatch     : {batch_size:,} / {minibatch_size:,} steps ({args.ppo_epochs} epochs PPO)")
    print(f"  - ExperienceBuffer      : {exp_buffer_size:,} timesteps (100% résident en VRAM GPU {verified_device})")
    print(f"  - Modèle RLBot exporté  : {_to_rel_path(out_model)} ({out_model})")
    print("=" * 84)

    learner: Optional[ContinuousSusuLearner] = None
    current_wall_time = initial_wall_time
    final_steps = initial_loaded_steps

    while True:
        try:
            latest_checkpoint = find_latest_checkpoint(save_dir)
            if latest_checkpoint is not None:
                upgrade_checkpoint_obs_dim(latest_checkpoint.parent, target_in_dim=142)
            checkpoint_folder = str(latest_checkpoint.parent) if latest_checkpoint else None

            if checkpoint_folder:
                try:
                    loaded_steps = int(latest_checkpoint.parent.name)
                    final_steps = loaded_steps
                    in_game_str, exp_dict = format_in_game_time(loaded_steps)
                    print(f"\n[susu_bot] Reprise depuis le checkpoint : {_to_rel_path(latest_checkpoint.parent)}")
                    print(
                        f"[susu_bot] Expérience totale déjà accumulée : {in_game_str} "
                        f"({exp_dict['years']:.4f} ans / {exp_dict['total_days']:.2f} jours / {exp_dict['total_matches_5min']:.1f} matchs de 5 min)"
                    )
                except ValueError:
                    print(f"\n[susu_bot] Reprise depuis le checkpoint : {_to_rel_path(latest_checkpoint.parent)}")
            else:
                print("\n[susu_bot] Démarrage d'une nouvelle session d'entraînement Self-Play...")

            learner = ContinuousSusuLearner(
                make_env_create_fn(
                    initial_stage=args.curriculum_stage,
                    engine_type=args.engine,
                    timeout_seconds=args.match_duration,
                    no_touch_timeout=args.no_touch_timeout,
                    arenas_per_worker=arenas_per_worker,
                    specialization=args.specialization,
                    infinite_boost=args.infinite_boost,
                ),
                n_proc=n_proc,
                min_inference_size=min_inference,
                metrics_logger=None,
                ppo_batch_size=batch_size,
                ts_per_iteration=batch_size,
                exp_buffer_size=exp_buffer_size,
                ppo_minibatch_size=minibatch_size,
                ppo_ent_coef=args.ent_coef,
                ppo_epochs=args.ppo_epochs,
                standardize_returns=True,
                standardize_obs=True,
                save_every_ts=args.save_every,
                timestep_limit=target_timesteps,
                checkpoints_save_folder=str(save_dir),
                add_unix_timestamp=False,
                checkpoint_load_folder=checkpoint_folder,
                log_to_wandb=args.wandb,
                n_checkpoints_to_keep=keep_checkpoints,
                shm_buffer_size=shm_buffer_size,
                device=verified_device,
                export_model_path=out_model,
                status_file_path=status_file,
                initial_wall_time_seconds=current_wall_time,
                target_vram_gb=target_vram_gb,
                instance_launch_delay=0.01,
            )

            learner.learn()
            current_wall_time = learner.update_wall_clock()
            final_steps = int(learner.agent.cumulative_timesteps)

            if not run_endlessly:
                learner.save(final_steps)
                break

        except KeyboardInterrupt:
            print("\n[susu_bot] Signal d'arrêt reçu (Ctrl+C). Fermeture des environnements RocketSim et sauvegarde finale...")
            if learner is not None:
                try:
                    current_wall_time = learner.update_wall_clock()
                    final_steps = int(learner.agent.cumulative_timesteps)
                    learner.save(final_steps)
                except Exception as save_err:
                    print(f"[susu_bot] Note lors de la sauvegarde finale: {save_err}")
            break
        except Exception as e:
            if learner is not None:
                try:
                    current_wall_time = learner.update_wall_clock()
                    final_steps = int(learner.agent.cumulative_timesteps)
                    learner.cleanup()
                except Exception:
                    pass
            if not args.watchdog:
                raise
            print(f"\n[susu_bot Watchdog] Alerte : exception rencontrée : {e}")
            print("[susu_bot Watchdog] Redémarrage et reprise automatique depuis le dernier checkpoint dans 5 secondes...")
            time.sleep(5)

    session_wall_elapsed = max(0.0, time.perf_counter() - session_start_wall)
    total_wall_seconds = max(current_wall_time, initial_wall_time + session_wall_elapsed)
    session_steps = max(0, final_steps - initial_loaded_steps)

    latest = find_latest_checkpoint(save_dir)
    if latest is not None:
        try:
            export_checkpoint(latest, out_model)
        except Exception:
            pass

    try:
        in_game_str, exp_dict = format_in_game_time(final_steps)
        wall_str, wall_dict = format_duration_seconds(total_wall_seconds)
        avg_rew = 0.0
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    prev_st = json.load(f)
                avg_rew = float(prev_st.get("policy_average_reward", 0.0))
            except Exception:
                pass
        if learner is not None and getattr(learner.agent, "average_reward", None) is not None:
            avg_rew = float(learner.agent.average_reward)

        status_data = {
            "cumulative_timesteps": int(final_steps),
            "in_game_experience": in_game_str,
            "in_game_years": round(exp_dict["years"], 4),
            "in_game_days": round(exp_dict["total_days"], 2),
            "in_game_hours": round(exp_dict["total_hours"], 2),
            "in_game_seconds": round(exp_dict["total_seconds"], 1),
            "total_matches_5min": round(exp_dict["total_matches_5min"], 1),
            "cumulative_wall_time_seconds": round(total_wall_seconds, 2),
            "cumulative_wall_time_formatted": wall_str,
            "cumulative_wall_time_hours": round(wall_dict["total_hours"], 2),
            "last_session_timesteps": int(session_steps),
            "last_session_wall_time_seconds": round(session_wall_elapsed, 2),
            "policy_average_reward": round(avg_rew, 4),
            "ppo_device": verified_device,
            "epoch": int(learner.epoch) if learner is not None else 0,
            "latest_checkpoint": _to_rel_path(latest.parent) if latest is not None else None,
            "exported_model": _to_rel_path(out_model),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=4)
    except Exception as e:
        print(f"[susu_bot] Note sauvegarde training_status.json: {e}")

    print_all_time_summary_banner(
        total_timesteps=final_steps,
        session_timesteps=session_steps,
        total_wall_seconds=total_wall_seconds,
        session_wall_seconds=session_wall_elapsed,
        out_model=out_model,
        status_file=status_file,
    )


if __name__ == "__main__":
    main()

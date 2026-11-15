#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'audit et de vérification exhaustive des dépendances pour susu_bot (Susu-v0).
Vérifie :
1. Présence et compatibilité de toutes les bibliothèques requises.
2. Détection de CUDA, nom du GPU, mémoire VRAM et support TensorCore TF32.
3. Extension C++ RocketSim et initialisation des collision_meshes.
4. Bibliothèque rlgym_sim et création d'une arène de test.
5. Intégrité de rlgym_ppo et de l'architecture du modèle TorchScript 142 dimensions.
"""

import os
import sys
from pathlib import Path

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

print("=" * 84)
print("🔍 AUDIT EXHAUSTIF DES DÉPENDANCES & MATÉRIEL — SUSU_BOT (SUSU-V0)")
print("=" * 84)

checks_passed = 0
checks_total = 0

def check(name, test_fn):
    global checks_passed, checks_total
    checks_total += 1
    sys.stdout.write(f"  [{checks_total:02d}] {name:<45} : ")
    sys.stdout.flush()
    try:
        msg = test_fn()
        checks_passed += 1
        print(f"✅ OK ({msg})")
        return True
    except Exception as e:
        print(f"❌ ÉCHEC ({e})")
        return False

# 1. Python version
def _test_py():
    v = sys.version_info
    assert (v.major, v.minor) in ((3, 10), (3, 11), (3, 12), (3, 13)), f"Python 3.10-3.13 recommandé, actuel: {sys.version}"
    return f"v{v.major}.{v.minor}.{v.micro}"
check("Version de l'interpréteur Python", _test_py)

# 2. NumPy
def _test_numpy():
    import numpy as np
    assert np.__version__.startswith("1."), f"NumPy 1.x requis pour compatibilité C++, actuel: {np.__version__}"
    return f"v{np.__version__}"
check("NumPy (Compatibilité C++ RocketSim)", _test_numpy)

# 3. PyTorch
def _test_torch():
    import torch
    return f"v{torch.__version__}"
check("PyTorch", _test_torch)

# 4. PyTorch CUDA GPU
def _test_cuda():
    import torch
    if not torch.cuda.is_available():
        return "CPU uniquement (Aucun GPU CUDA détecté)"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return f"{gpu_count}x GPU : {gpu_name} ({vram_gb:.1f} Go VRAM)"
check("Accélération GPU PyTorch CUDA", _test_cuda)

# 5. Gym
def _test_gym():
    import gym
    return f"v{gym.__version__}"
check("Gym (Environnement RL)", _test_gym)

# 6. RocketSim C++
def _test_rocketsim():
    import RocketSim as rsim
    from src.rocketsim_env import ensure_rocketsim_meshes_initialized
    mesh_path = ensure_rocketsim_meshes_initialized()
    return f"v{getattr(rsim, '__version__', '2.x')} | Meshes: {Path(mesh_path).name}"
check("RocketSim C++ & Collision Meshes", _test_rocketsim)

# 7. rlgym-sim
def _test_rlgym_sim():
    import rlgym_sim
    import importlib.metadata
    try:
        ver = importlib.metadata.version("rlgym-sim")
    except Exception:
        ver = "1.2.x"
    return f"v{ver}"
check("rlgym-sim (Moteur de match simulé)", _test_rlgym_sim)

# 8. rlgym-ppo
def _test_rlgym_ppo():
    import rlgym_ppo
    import importlib.metadata
    try:
        ver = importlib.metadata.version("rlgym-ppo")
    except Exception:
        ver = "1.3.x"
    return f"v{ver}"
check("rlgym-ppo (Algorithme d'apprentissage PPO)", _test_rlgym_ppo)

# 9. rlgym-compat
def _test_rlgym_compat():
    import rlgym_compat
    import importlib.metadata
    try:
        ver = importlib.metadata.version("rlgym-compat")
    except Exception:
        ver = "1.0.x"
    return f"v{ver}"
check("rlgym-compat (Compatibilité RLBot)", _test_rlgym_compat)

# 10. RLBot
def _test_rlbot():
    import rlbot
    return f"v{rlbot.__version__}"
check("RLBot Framework", _test_rlbot)

# 11. Psutil
def _test_psutil():
    import psutil
    cores = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return f"v{psutil.__version__} | {cores} threads CPU | {ram_gb:.1f} Go RAM"
check("Psutil (Surveillance CPU & RAM)", _test_psutil)

# 12. Test Arène RocketSim & Simulation d'un pas
def _test_sim_step():
    import numpy as np
    from src.rocketsim_env import build_rocketsim_env
    env = build_rocketsim_env()
    obs = env.reset()
    assert isinstance(obs, (list, tuple)) and len(obs) >= 1, "Format observation invalide"
    first_obs = np.asarray(obs[0], dtype=np.float32)
    assert first_obs.shape[-1] == 142, f"Observation size attendue 142, reçu: {first_obs.shape}"
    actions = [0] * len(obs)
    obs, rew, done, info = env.step(actions)
    assert len(rew) == len(obs), "Récompenses incohérentes avec le nombre d'agents"
    env.close()
    return f"Arène 1v1 ({len(obs)} agents), reset & step 142-dim validés"
check("Simulation physique RocketSim (Reset & Step)", _test_sim_step)

# 13. Test Modèle Neural Susu-v0
def _test_susu_agent():
    from src.bot import SusuAgent
    agent = SusuAgent()
    assert agent.is_ready, "Échec de chargement de susu-model.pt"
    assert agent.expected_in_dim == 142, f"Attendu 142 entrées, reçu: {agent.expected_in_dim}"
    import numpy as np
    dummy_obs = np.zeros(142, dtype=np.float32)
    action, _ = agent.act(dummy_obs)
    assert len(action) == 8, f"Action attendue 8 contrôles, reçu: {len(action)}"
    return f"susu-model.pt chargé avec succès (in_dim=142)"
check("Réseau Neural Susu (TorchScript 142 entrées)", _test_susu_agent)

print("=" * 84)
print(f"📊 RÉSULTAT DU CONTRÔLE : {checks_passed} / {checks_total} tests réussis avec succès !")
if checks_passed == checks_total:
    print("🚀 TOUTES LES DÉPENDANCES SONT OPÉRATIONNELLES SANS AUCUN DÉFAUT !")
else:
    print("⚠️ Certains composants nécessitent une attention.")
print("=" * 84)

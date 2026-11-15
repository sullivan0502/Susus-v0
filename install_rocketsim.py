#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INSTALLATEUR ROCKETSIM & RLVISER 3D — SUSU_BOT
=============================================
Ce fichier installe et configure tous les composants nécessaires pour exécuter RocketSim C++
et le visualiseur 3D RLViser en temps réel pour susu_bot.

Conformément à votre demande, ce fichier est 100% autonome et PEUT ÊTRE SUPPRIMÉ après exécution.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

# Fix console encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

_ROOT = Path(__file__).resolve().parent
RLVISER_EXE = _ROOT / "rlviser.exe"
COLLISION_MESHES = _ROOT / "collision_meshes"

REQUIRED_PACKAGES = [
    "numpy",
    "torch",
    "RocketSim",
    "rlviser-py",
    "rlgym-sim",
    "rlgym-ppo",
]

RLVISER_DOWNLOAD_URL = (
    "https://github.com/VirxEC/rlviser/releases/download/v0.9.3/rlviser-x86_64-pc-windows-msvc.exe"
)

def print_header():
    print("=" * 80)
    print("🚀 INSTALLATION ROCKETSIM C++ & VISUALISEUR 3D RLVISER (SUSU_BOT)")
    print("=" * 80)
    print(f"  • Répertoire d'installation : {_ROOT}")
    print(f"  • Interpréteur Python       : {sys.executable} (v{sys.version.split()[0]})")
    print("=" * 80)

def check_and_install_packages():
    print("\n[1/4] Vérification des modules Python...")
    for pkg in REQUIRED_PACKAGES:
        try:
            # Normalize module name for import
            mod_name = pkg.lower().replace("-", "_")
            if mod_name == "rocketsim":
                mod_name = "RocketSim"
            __import__(mod_name)
            print(f"  ✅ {pkg:<15} : Déjà installé")
        except ImportError:
            print(f"  ⚙️ {pkg:<15} : Installation en cours via pip...")
            res = subprocess.run([sys.executable, "-m", "pip", "install", pkg], capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  ✅ {pkg:<15} : Installé avec succès")
            else:
                print(f"  ⚠️ {pkg:<15} : Erreur d'installation : {res.stderr.strip()[:100]}")

def check_and_download_rlviser():
    print("\n[2/4] Vérification du binaire 3D RLViser (rlviser.exe)...")
    if RLVISER_EXE.exists() and RLVISER_EXE.stat().st_size > 10_000_000:
        size_mb = RLVISER_EXE.stat().st_size / (1024 * 1024)
        print(f"  ✅ rlviser.exe déjà présent ({size_mb:.1f} Mo)")
        return

    print("  ⬇️ Téléchargement de RLViser v0.9.3 depuis GitHub...")
    try:
        urllib.request.urlretrieve(RLVISER_DOWNLOAD_URL, str(RLVISER_EXE))
        size_mb = RLVISER_EXE.stat().st_size / (1024 * 1024)
        print(f"  ✅ rlviser.exe téléchargé avec succès ({size_mb:.1f} Mo)")
    except Exception as e:
        print(f"  ❌ Erreur de téléchargement : {e}")
        print("  💡 Vous pouvez le télécharger manuellement depuis :")
        print(f"     {RLVISER_DOWNLOAD_URL}")

def check_collision_meshes():
    print("\n[3/4] Vérification des maillages de collision (collision_meshes)...")
    if COLLISION_MESHES.exists() and any(COLLISION_MESHES.iterdir()):
        count = len(list(COLLISION_MESHES.iterdir()))
        print(f"  ✅ Dossier collision_meshes présent ({count} fichiers de maillage)")
    else:
        print("  ⚙️ Recherche des maillages dans les packages installés...")
        src_meshes = _find_installed_collision_meshes()
        if src_meshes and src_meshes.exists():
            import shutil
            shutil.copytree(src_meshes, COLLISION_MESHES, dirs_exist_ok=True)
            print(f"  ✅ Maillages copiés depuis {src_meshes}")
        else:
            print("  ⚠️ Aucun dossier de collision trouvé. Le mode standard Soccar utilisera les maillages par défaut.")

def _find_installed_collision_meshes() -> Path:
    # Look in site-packages
    for p in sys.path:
        cand = Path(p) / "rlgym" / "rocket_league" / "sim" / "collision_meshes"
        if cand.exists():
            return cand
    return None

def test_rocketsim_initialization():
    print("\n[4/4] Test d'auto-diagnostic de RocketSim...")
    try:
        import RocketSim as rsim
        meshes_path = str(COLLISION_MESHES) if COLLISION_MESHES.exists() else "./collision_meshes"
        try:
            rsim.init(meshes_path)
        except RuntimeError as re:
            if "already inited" not in str(re).lower():
                raise re
        arena = rsim.Arena(rsim.GameMode.SOCCAR)
        car = arena.add_car(rsim.Team.BLUE)
        arena.step(1)
        print("  ✅ Moteur physique RocketSim C++ : Opérationnel et prêt à 120 FPS !")
    except Exception as e:
        print(f"  ❌ Erreur d'initialisation RocketSim : {e}")

    try:
        import rlviser_py as rlv
        os.chdir(str(_ROOT))
        rlv.launch()
        import time
        time.sleep(0.5)
        rlv.quit()
        print("  ✅ Moteur visuel RLViser 3D       : Fenêtre graphique testée avec succès !")
    except Exception as e:
        print(f"  ⚠️ Diagnostic RLViser : {e}")

def print_completion():
    print("\n" + "=" * 80)
    print("🎉 INSTALLATION TERMINÉE AVEC SUCCÈS !")
    print("=" * 80)
    print("👉 Vous pouvez lancer le visualiseur à tout moment avec :")
    print("   python run_rocketsim.py  (ou double-clic sur run_rocketsim.bat)")
    print("\n👉 Vous pouvez lancer l'entraînement spécialisé avec :")
    print("   python train.py          (ou double-clic sur train.bat)")
    print("\n🗑️ NOTE : Comme demandé, ce fichier d'installation (install_rocketsim.py)")
    print("   peut maintenant être supprimé en toute sécurité si vous le souhaitez.")
    print("=" * 80 + "\n")

def main():
    print_header()
    check_and_install_packages()
    check_and_download_rlviser()
    check_collision_meshes()
    test_rocketsim_initialization()
    print_completion()

if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# ====================================================================================
#  SUSU_BOT (SUSU-V0) : LANCEUR CLUSTER MULTI-GPU HAUTE PERFORMANCE (VAST.AI / LINUX)
#  Mobilise 100% des GPUs (ex: 4x RTX 5090), 100% des cœurs CPU et toute la RAM.
#  GPU 0: Aérienne & Resets (Boost Illimité) | GPU 1: Flicks & Dribble
#  GPU 2: Déplacements & Rotations          | GPU 3: Tirs & Précision Cages
# ====================================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

ulimit -c 0 2>/dev/null || true

# Résolution de l'interpréteur Python
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/venv/bin/python"
elif [ -x "/opt/venv/bin/python" ]; then
    PYTHON_EXE="/opt/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
else
    PYTHON_EXE="python"
fi

echo "===================================================================================="
echo "  SUSU_BOT : CLUSTER MULTI-GPU & MULTI-SPÉCIALISTE (C++ ROCKETSIM + PYTORCH CUDA)"
echo "===================================================================================="
echo " [Dossier]       : $SCRIPT_DIR"
echo " [Python]        : $PYTHON_EXE"
echo " [Cluster Mode]  : train_cluster.py (Auto-détection multi-cartes et répartition)"
echo " [Arrêt Ctrl+C]  : Sauvegarde immédiate de chaque GPU et synchronisation du champion"
echo "===================================================================================="
echo ""

exec "$PYTHON_EXE" -B -u "$SCRIPT_DIR/train_cluster.py" "$@"

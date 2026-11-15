#!/usr/bin/env bash
# ====================================================================================
#  SUSU_BOT (SUSU-V0) : SCRIPT D'ENTRAINEMENT SERVEUR LINUX / VAST.AI
#  Chemins 100% dynamiques - Plafond RAM sécurisé - IPC OMP_NUM_THREADS=1
# ====================================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Si le mode cluster est demandé, déléguer directement vers launch_cluster.sh
for arg in "$@"; do
    if [ "$arg" = "--cluster" ]; then
        exec "$SCRIPT_DIR/launch_cluster.sh" "$@"
    fi
done

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

ulimit -c 0 2>/dev/null || true

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
echo "  SUSU_BOT (SUSU-V0) : ENTRAINEMENT SERVEUR C++ ROCKETSIM + rlgym-sim + CUDA GPU"
echo "===================================================================================="
echo " [Dossier]       : $SCRIPT_DIR (100% dynamique)"
echo " [Python]        : $PYTHON_EXE"
echo " [Auto-Scale]    : Plafond RAM sécurisé (<80% RAM, 0 Go swap)"
echo " [Multi-GPU]     : Pour lancer sur tous les GPUs en parallèle -> ./launch_cluster.sh"
echo " [Arret Ctrl+C]  : Appuyez sur Ctrl+C (ou 'q') pour sauvegarder et afficher le bilan"
echo "===================================================================================="
echo ""

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Demarrage de la simulation C++ RocketSim + GPU CUDA..."
    "$PYTHON_EXE" -B -u "$SCRIPT_DIR/train.py" --continuous --engine rocketsim --device cuda:0 --match_duration 300 --save_every 100000 "$@"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
        echo ""
        echo "[susu_bot] Entrainement arrete proprement. Le temps total cumule est affiche ci-dessus."
        break
    fi

    echo ""
    echo "===================================================================================="
    echo " [susu_bot Watchdog] Erreur inattendue detectee (Code: $EXIT_CODE)."
    echo " Redemarrage automatique de la boucle d'entrainement dans 5 secondes..."
    echo "===================================================================================="
    sleep 5
done

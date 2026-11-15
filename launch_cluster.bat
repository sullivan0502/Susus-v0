@echo off
setlocal
cd /d "%~dp0"
title SUSU_BOT Multi-GPU Cluster Orchestrator (RocketSim C++ + PyTorch CUDA)

set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set TORCH_NUM_THREADS=1

rem Resolution dynamique de l'interpreteur Python avec CUDA
set "PYTHON_EXE="
if exist "%LOCALAPPDATA%\RLBotGUIX\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\RLBotGUIX\Python311\python.exe"
) else if exist "%~dp0..\..\Python311\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\Python311\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo ====================================================================================
echo   SUSU_BOT : ORCHESTRATEUR CLUSTER MULTI-GPU & MULTI-SPECIALISTE (C++ ROCKETSIM)
echo ====================================================================================
echo  [Chemins]        : 100%% Dynamiques (%~dp0)
echo  [Python]         : %PYTHON_EXE%
echo  [Cluster Mode]   : train_cluster.py (Partitionnement CPU/RAM et Dashboard Live)
echo  [Arret Ctrl+C]   : Sauvegarde immediate de tous les GPUs et modele champion exporte
echo ====================================================================================
echo.

"%PYTHON_EXE%" -B -u train_cluster.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ====================================================================================
    echo  [SUSU_BOT Cluster] Fin d'execution ou arret signale [Code: %ERRORLEVEL%]
    echo ====================================================================================
)

pause

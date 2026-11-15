@echo off
chcp 65001 >nul
title Installation RocketSim & RLViser - susu_bot

set "PYTHON_EXE=C:\Users\sulli\AppData\Local\RLBotGUIX\Python311\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ======================================================================
echo 🚀 Lancement de l'installateur RocketSim ^& RLViser pour susu_bot...
echo ======================================================================

"%PYTHON_EXE%" "%~dp0install_rocketsim.py"

echo.
pause

@echo off
chcp 65001 >nul
title Entrainement RocketSim & RLGym-PPO - susu_bot

set "PYTHON_EXE=C:\Users\sulli\AppData\Local\RLBotGUIX\Python311\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%~dp0train.py" %*

echo.
pause

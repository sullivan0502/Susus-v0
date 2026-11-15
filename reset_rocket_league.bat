@echo off
title Reinitialisation Rocket League et RLBot
echo ==============================================================================
echo   NETTOYAGE DES PROCESSUS ZOMBIES ROCKET LEAGUE ET RLBOT
echo ==============================================================================
echo.
echo Fermeture forcee de RocketLeague.exe...
taskkill /F /IM RocketLeague.exe 2>nul
echo Fermeture forcee de RLBot.exe...
taskkill /F /IM RLBot.exe 2>nul
echo.
echo Nettoyage termine avec succes ! Vous pouvez maintenant relancer votre match dans RLBotGUIX.
pause

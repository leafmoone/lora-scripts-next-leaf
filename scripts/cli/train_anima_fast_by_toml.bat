@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
title Anima Fast train (TOML)

set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"

set "CONFIG_FILE=%~1"
if not defined CONFIG_FILE set "CONFIG_FILE=docs\examples\anima-lora-benchmark-fast.toml"
if not exist "%CONFIG_FILE%" (
    echo [Error] Config not found: %CONFIG_FILE%
    echo Usage: %~nx0 [path\to\config.toml]
    pause
    exit /b 1
)

set "FAST_PY=extensions\anima_lora\.venv\Scripts\python.exe"
set "TRAIN_PY=extensions\anima_lora\source\train.py"
if not exist "%FAST_PY%" (
    echo [Error] Fast venv missing: %FAST_PY%
    echo Run scripts\cli\install_anima_fast.bat first.
    pause
    exit /b 1
)
if not exist "%TRAIN_PY%" (
    echo [Error] train.py missing: %TRAIN_PY%
    pause
    exit /b 1
)

if not defined HF_HOME set "HF_HOME=%PROJECT_ROOT%\huggingface"
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"
set "PYTHONUTF8=1"

echo ========================================
echo   Anima Fast CLI Train
echo   Config: %CONFIG_FILE%
echo ========================================
echo.

"%FAST_PY%" "%TRAIN_PY%" --config_file "%CONFIG_FILE%"
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo [Error] Training failed / exit %RC%
    pause
    exit /b %RC%
)

echo.
echo Done.
pause
exit /b 0

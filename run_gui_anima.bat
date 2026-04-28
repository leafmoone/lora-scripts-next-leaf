@echo off
setlocal EnableExtensions

chcp 65001 >nul
cd /d "%~dp0"
title Anima LoRA GUI - SD-Trainer

set "GUI_PORT=30021"
set "TENSORBOARD_PORT=6012"
set "HF_HOME=huggingface"
set "PYTHONUTF8=1"
set "PYTHON_EXE=.venv-anima-test\Scripts\python.exe"

echo ============================================================
echo  Anima LoRA GUI launcher
echo ============================================================
echo.
echo Working directory: %CD%
echo GUI port:          %GUI_PORT%
echo TensorBoard port:  %TENSORBOARD_PORT%
echo.

if not exist "%PYTHON_EXE%" (
    echo [WARN] Cannot find %PYTHON_EXE%
    echo [WARN] Falling back to python from PATH.
    set "PYTHON_EXE=python"
)

set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /R /C:":%GUI_PORT% .*LISTENING"') do (
    set "PORT_PID=%%P"
)

if defined PORT_PID (
    echo [WARN] Port %GUI_PORT% is already in use by PID %PORT_PID%.
    choice /C YN /M "Stop that process and reuse port %GUI_PORT%?"
    if errorlevel 2 (
        set "GUI_PORT=30022"
        echo [INFO] Will use GUI port %GUI_PORT% instead.
    ) else (
        taskkill /PID %PORT_PID% /F
        timeout /t 2 /nobreak >nul
    )
)

echo.
echo Required Anima model paths:
echo   DiT:   .\sd-models\anima\split_files\diffusion_models\anima-preview3-base.safetensors
echo   VAE:   .\sd-models\anima\split_files\vae\qwen_image_vae.safetensors
echo   Qwen3: .\sd-models\anima\split_files\text_encoders\qwen_3_06b_base.safetensors
echo.
echo Starting GUI. Training logs will appear in this window.
echo Open: http://127.0.0.1:%GUI_PORT%
echo.

"%PYTHON_EXE%" "gui.py" --skip-prepare-environment --disable-tageditor --port %GUI_PORT% --tensorboard-port %TENSORBOARD_PORT%

echo.
echo GUI process exited.
pause

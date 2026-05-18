@echo off
chcp 65001 >nul 2>&1
title Download Anima Model
cd /d "%~dp0"

echo ============================================================
echo   Anima Model Downloader / Anima 模型下载器
echo ============================================================
echo.
echo   Files / 将下载以下文件:
echo     1. anima-base-v1.0.safetensors   (DiT)
echo     2. qwen_3_06b_base.safetensors   (Text Encoder / 文本编码器)
echo     3. qwen_image_vae.safetensors    (VAE)
echo.
echo   Save to / 保存到: SD-Trainer\sd-models\anima\
echo.
echo   Source / 来源: ModelScope (circlestone-labs/Anima)
echo ============================================================
echo.

set "MODEL_DIR=%~dp0SD-Trainer\sd-models\anima"
set "BASE_URL=https://www.modelscope.cn/models/circlestone-labs/Anima/resolve/master/split_files"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

:: ---- File 1: DiT ----
if exist "%MODEL_DIR%\anima-base-v1.0.safetensors" (
    echo [Skip] anima-base-v1.0.safetensors already exists / 已存在
) else (
    echo [1/3] Downloading anima-base-v1.0.safetensors ...
    echo       DiT model / DiT 主模型
    curl -L -# -o "%MODEL_DIR%\anima-base-v1.0.safetensors" "%BASE_URL%/diffusion_models/anima-base-v1.0.safetensors"
    if errorlevel 1 (
        echo       [Failed] Download failed / 下载失败
        del "%MODEL_DIR%\anima-base-v1.0.safetensors" 2>nul
    ) else (
        echo       [OK] Done
    )
)
echo.

:: ---- File 2: Text Encoder ----
if exist "%MODEL_DIR%\qwen_3_06b_base.safetensors" (
    echo [Skip] qwen_3_06b_base.safetensors already exists / 已存在
) else (
    echo [2/3] Downloading qwen_3_06b_base.safetensors ...
    echo       Qwen3 Text Encoder / Qwen3 文本编码器
    curl -L -# -o "%MODEL_DIR%\qwen_3_06b_base.safetensors" "%BASE_URL%/text_encoders/qwen_3_06b_base.safetensors"
    if errorlevel 1 (
        echo       [Failed] Download failed / 下载失败
        del "%MODEL_DIR%\qwen_3_06b_base.safetensors" 2>nul
    ) else (
        echo       [OK] Done
    )
)
echo.

:: ---- File 3: VAE ----
if exist "%MODEL_DIR%\qwen_image_vae.safetensors" (
    echo [Skip] qwen_image_vae.safetensors already exists / 已存在
) else (
    echo [3/3] Downloading qwen_image_vae.safetensors ...
    echo       Qwen Image VAE
    curl -L -# -o "%MODEL_DIR%\qwen_image_vae.safetensors" "%BASE_URL%/vae/qwen_image_vae.safetensors"
    if errorlevel 1 (
        echo       [Failed] Download failed / 下载失败
        del "%MODEL_DIR%\qwen_image_vae.safetensors" 2>nul
    ) else (
        echo       [OK] Done
    )
)
echo.

:: ---- Summary ----
echo ============================================================
echo   Download complete / 下载完成
echo.
echo   Model path for WebUI / WebUI 中填写的模型路径:
echo     DiT:           ./sd-models/anima/anima-base-v1.0.safetensors
echo     Text Encoder:  ./sd-models/anima/qwen_3_06b_base.safetensors
echo     VAE:           ./sd-models/anima/qwen_image_vae.safetensors
echo ============================================================
pause

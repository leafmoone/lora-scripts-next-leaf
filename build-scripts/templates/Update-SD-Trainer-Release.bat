@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
title Update SD-Trainer from Release
set "PORTABLE_ROOT=%~dp0"
set "PROJECT_DIR=%PORTABLE_ROOT%SD-Trainer"

echo ========================================
echo   SD-Trainer Release Update
echo   从 GitHub Release 更新整合包
echo ========================================
echo.
echo This downloads the latest SD-Trainer-v*.7z from Releases,
echo merges SD-Trainer code, and keeps your models/output/logs.
echo 将下载最新 Release 整合包，合并代码并保留你的数据目录。
echo.
echo Recommended when:
echo   - git update fails or Git is not installed
echo   - your 7z has no .git folder
echo   - you prefer a full release sync
echo.
echo 适用于：无 Git、git 更新失败、或希望与 Release 完全对齐。
echo.

if not exist "%PROJECT_DIR%\gui.py" (
    echo [Error] SD-Trainer not found / 未找到 SD-Trainer
    pause
    exit /b 1
)

echo Please close SD-Trainer WebUI before updating.
echo 请先关闭 WebUI。
echo.

set "PS_SCRIPT=%PORTABLE_ROOT%SD-Trainer\scripts\portable\update_from_release.ps1"
if not exist "%PS_SCRIPT%" (
    echo [Error] Missing %PS_SCRIPT%
    echo Your package is too old for in-place release update.
    echo 当前包过旧，请手动从 Releases 下载新版 7z。
    pause
    exit /b 1
)

where powershell >nul 2>&1
if errorlevel 1 (
    echo [Error] PowerShell not found / 未找到 PowerShell
    pause
    exit /b 1
)

where curl >nul 2>&1
if errorlevel 1 (
    echo [Error] curl not found / 未找到 curl（Win10 1803+ 通常自带）
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -PortableRoot "%PORTABLE_ROOT%"
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo [Error] Release update failed / Release 更新失败
    echo You can still use Update-SD-Trainer.bat ^(git^) or download 7z manually:
    echo https://github.com/wochenlong/lora-scripts-next/releases
    pause
    exit /b %RC%
)

echo.
pause
exit /b 0

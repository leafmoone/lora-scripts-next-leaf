@echo off
chcp 65001 >nul 2>&1
title Update SD-Trainer
cd /d "%~dp0SD-Trainer"

echo ========================================
echo   SD-Trainer Update / 更新项目代码
echo ========================================
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo [Error] Git not found / 未找到 Git
    echo Please install Git: https://git-scm.com/
    pause
    exit /b 1
)

echo Pulling latest code / 拉取最新代码...
echo.
git pull
echo.

echo Updating submodules / 更新子模块...
git submodule update --init --recursive
echo.

if exist "scripts\portable\sync_portable_root_launchers.bat" (
    echo Refreshing portable root launchers / 刷新整合包根目录启动脚本...
    call "scripts\portable\sync_portable_root_launchers.bat" --nopause
) else (
    echo [Note] No scripts\portable\sync_portable_root_launchers.bat — if GUI fails after update, re-copy run_gui.bat from release or re-download 7z.
)

echo.
echo ========================================
echo   Done / 更新完成
echo ========================================
pause

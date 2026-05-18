@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: 检查是否已安装（venv 存在或 python 目录存在）
if exist "venv\Scripts\activate.bat" goto :launch
if exist "python\python.exe" goto :launch

:: 未安装，先执行安装
echo [首次使用] 正在安装依赖，请耐心等待...
powershell -ExecutionPolicy Bypass -File "%~dp0install-cn.ps1"
if errorlevel 1 (
    echo 安装失败，请检查网络连接后重试。
    pause
    exit /b 1
)

:launch
powershell -ExecutionPolicy Bypass -File "%~dp0run_gui.ps1"
pause

@echo off
chcp 65001 >nul 2>&1
:: Refresh portable ROOT launchers from SD-Trainer (run after git pull in SD-Trainer)
set "PORTABLE_ROOT=%~dp0..\..\..\"
set "SRC_GUI=%PORTABLE_ROOT%SD-Trainer\run_gui.bat"
set "SRC_PORTABLE=%PORTABLE_ROOT%SD-Trainer\scripts\portable\run_gui_portable_shim.bat"

if not exist "%SRC_GUI%" set "SRC_GUI=%PORTABLE_ROOT%run_gui.bat"
if not exist "%SRC_GUI%" (
    echo [ERROR] Missing run_gui.bat under SD-Trainer or portable root
    if /I not "%~1"=="--nopause" pause
    exit /b 1
)

copy /Y "%SRC_GUI%" "%PORTABLE_ROOT%run_gui.bat" >nul
echo  Updated: %PORTABLE_ROOT%run_gui.bat

if exist "%SRC_PORTABLE%" (
    copy /Y "%SRC_PORTABLE%" "%PORTABLE_ROOT%run_gui_portable.bat" >nul
    echo  Updated: %PORTABLE_ROOT%run_gui_portable.bat
)

set "SRC_UPDATE=%PORTABLE_ROOT%SD-Trainer\build-scripts\templates\Update-SD-Trainer.bat"
if not exist "%SRC_UPDATE%" set "SRC_UPDATE=%PORTABLE_ROOT%Update-SD-Trainer.bat"
if exist "%SRC_UPDATE%" (
    copy /Y "%SRC_UPDATE%" "%PORTABLE_ROOT%Update-SD-Trainer.bat" >nul
    echo  Updated: %PORTABLE_ROOT%Update-SD-Trainer.bat
)

set "SRC_UPDATE_REL=%PORTABLE_ROOT%SD-Trainer\build-scripts\templates\Update-SD-Trainer-Release.bat"
if not exist "%SRC_UPDATE_REL%" set "SRC_UPDATE_REL=%PORTABLE_ROOT%Update-SD-Trainer-Release.bat"
if exist "%SRC_UPDATE_REL%" (
    copy /Y "%SRC_UPDATE_REL%" "%PORTABLE_ROOT%Update-SD-Trainer-Release.bat" >nul
    echo  Updated: %PORTABLE_ROOT%Update-SD-Trainer-Release.bat
)

echo.
echo  Done. Root launchers now match SD-Trainer\ (for old 7z without scripts\portable\).
if /I not "%~1"=="--nopause" pause

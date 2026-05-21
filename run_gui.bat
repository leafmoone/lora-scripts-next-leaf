@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: Stable user entrypoint. Dispatches to the correct launcher for the current layout.
:: - Source checkout: run_gui_source.bat
:: - Portable package: run_gui_portable.bat

if exist "python_embeded\python.exe" if exist "SD-Trainer\gui.py" (
    if exist "run_gui_portable.bat" (
        call "%~dp0run_gui_portable.bat" %*
        exit /b %errorlevel%
    )
)

if exist "run_gui_source.bat" (
    call "%~dp0run_gui_source.bat" %*
    exit /b %errorlevel%
)

echo [ERROR] No launcher found. Please make sure the package is fully extracted.
pause
exit /b 1

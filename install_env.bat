@echo off
setlocal
cd /d "%~dp0"

set VENV_DIR=python-embedded
set REQ_FILE=requirements.txt

echo ============================================
echo  Tuner - Python Environment Setup
echo ============================================
echo.

:: Check if venv already exists
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Existing environment found at %VENV_DIR%\
    choice /C YN /M "Recreate from scratch"
    if errorlevel 2 goto :install_packages
    echo Removing old environment...
    rmdir /s /q "%VENV_DIR%"
)

:: Find Python 3.10
echo Looking for Python 3.10...

:: Try py launcher first (most reliable on Windows)
where py >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('py -3.10 -c "import sys; print(sys.executable)" 2^>nul') do set PYTHON=%%i
    if defined PYTHON (
        echo Found: %PYTHON%
        goto :create_venv
    )
)

:: Try conda
where conda >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('conda run -n py3.10-audio python -c "import sys; print(sys.executable)" 2^>nul') do set PYTHON=%%i
    if defined PYTHON (
        echo Found via conda: %PYTHON%
        goto :create_venv
    )
)

:: Try python on PATH
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('python -c "import sys; v=sys.version_info; print(sys.executable) if v.major==3 and v.minor>=10 else exit(1)" 2^>nul') do set PYTHON=%%i
    if defined PYTHON (
        echo Found: %PYTHON%
        goto :create_venv
    )
)

echo.
echo ERROR: Python 3.10+ not found.
echo Install Python 3.10 from https://www.python.org/downloads/
echo or activate a conda environment with: conda activate py3.10-audio
goto :error

:create_venv
echo.
echo Creating virtual environment in %VENV_DIR%\...
"%PYTHON%" -m venv "%VENV_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    goto :error
)

echo Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo WARNING: pip upgrade failed, continuing with existing version.
)

:install_packages
echo.
echo Installing packages from %REQ_FILE%...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ_FILE%"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Package installation failed.
    goto :error
)

echo.
echo ============================================
echo  Environment ready!
echo ============================================
echo.
echo  Run the tuner:    run.bat
echo  Debug mode:       run_debug.bat
echo  Build standalone: build.bat
echo.
pause
exit /b 0

:error
echo.
pause
exit /b 1

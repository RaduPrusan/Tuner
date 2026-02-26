@echo off
cd /d "%~dp0"
"%~dp0python-embedded\Scripts\python.exe" -m nuitka ^
    --standalone ^
    --plugin-disable=upx ^
    --windows-console-mode=attach ^
    --windows-icon-from-ico=icons\diapason.ico ^
    --include-data-files=style.css=style.css ^
    --include-data-files=icons\diapason.ico=icons\diapason.ico ^
    --enable-plugin=pyside6 ^
    --lto=yes ^
    --include-package=numpy ^
    --nofollow-import-to=PySide6.QtQml ^
    --nofollow-import-to=PySide6.QtQuick ^
    --nofollow-import-to=PySide6.QtMultimedia ^
    --nofollow-import-to=PySide6.QtWebEngineWidgets ^
    --nofollow-import-to=numba ^
    --nofollow-import-to=cython ^
    --output-filename=Tuner ^
    --product-name=Tuner ^
    --product-version=2.0.0.0 ^
    tuner-2.py

:: Copy libffi DLL required by conda's _ctypes.pyd (not included by Nuitka)
:: The DLL lives in the base conda env, not the venv itself
for /f "tokens=*" %%i in ('"python-embedded\Scripts\python.exe" -c "import sys,os; p=os.path.join(sys.base_prefix,'Library','bin','ffi-8.dll'); print(p) if os.path.exists(p) else None"') do (
    if exist "%%i" (
        copy /Y "%%i" "tuner-2.dist\ffi-8.dll"
        echo Copied ffi-8.dll to dist folder
    )
)
pause

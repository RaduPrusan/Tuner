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
pause

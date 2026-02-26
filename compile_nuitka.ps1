#conda activate py3.10-audio

#pyinstaller --onefile --windowed --icon=tuner.ico --name=Tuner --add-data "tuner.ico;." tuner_chatgpt_o3_013.py

# --mode=standalone, --mode=onefile, --mode=app
#python -m nuitka --mode=app --product-name=Tuner --output-filename=Tuner --product-version=1.0.0.0 --windows-console-mode=disable --windows-icon-from-ico=icons\diapason.ico --enable-plugin=pyside6 --lto=yes --follow-imports --include-package=numpy --include-package=pyaudio tuner_chatgpt_o3_014.py
#python -m nuitka --mode=app --product-name=Tuner --output-filename=Tuner --product-version=1.0.0.0 --windows-console-mode=disable --windows-icon-from-ico=icons\diapason.ico --enable-plugin=pyside6 --lto=yes --nofollow-import-to=numba --nofollow-import-to=cython tuner_chatgpt_o3_014.py


#python -m nuitka --standalone --plugin-disable=upx --windows-console-mode=disable --windows-icon-from-ico=icons\diapason.ico --enable-plugin=pyside6 --lto=yes --include-package=numpy --include-package=pyaudio --nofollow-import-to=PySide6.QtQml --nofollow-import-to=PySide6.QtQuick --nofollow-import-to=PySide6.QtMultimedia --nofollow-import-to=numba --nofollow-import-to=cython --nofollow-import-to=PySide6.QtWebEngineWidgets --output-filename=Tuner --product-name=Tuner --product-version=1.0.8.0 tuner_036_v1.0.8.py


python -m nuitka --standalone --plugin-disable=upx --windows-console-mode=attach --windows-icon-from-ico=icons\diapason.ico --include-data-files=style.css=style.css --include-data-files=icons\diapason.ico=icons\diapason.ico --enable-plugin=pyside6 --lto=yes --include-package=numpy --include-package=pyaudio --nofollow-import-to=PySide6.QtQml --nofollow-import-to=PySide6.QtQuick --nofollow-import-to=PySide6.QtMultimedia --nofollow-import-to=numba --nofollow-import-to=cython --nofollow-import-to=PySide6.QtWebEngineWidgets --output-filename=Tuner --product-name=Tuner --product-version=2.0.0.0 tuner-2.py


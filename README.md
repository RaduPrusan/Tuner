# Tuner - Real-Time Vocal Pitch Monitor

A desktop application that captures microphone audio in real time, detects pitch using multiple algorithms, and displays results on a scrolling piano roll with musical scale analysis.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green)
![Windows](https://img.shields.io/badge/platform-Windows-lightgrey)

## Features

- **Real-time pitch detection** with 7 algorithms: YIN, pYIN, MPM, SWIPE, Cepstrum, HPS, and Autocorrelation
- **Scrolling piano roll** rendered with hardware-accelerated OpenGL (V-Sync, 4x antialiasing)
- **Musical scale highlighting** — 14 scale types across all 12 keys, with correctly spelled note names following Western music theory conventions
- **Dual notation** — Solfege (Do Re Mi) and letter names (C D E), with smart sharp/flat selection based on key context
- **Simplified musical staff** displaying the detected note in standard notation
- **ASIO & WASAPI support** for low-latency audio input
- **Configurable** — all settings persisted to `config.json`, dark theme with pink accent

### Scale Types

Major, Natural Minor, Harmonic Minor, Melodic Minor, all 7 modes (Ionian through Locrian), Major Pentatonic, Minor Pentatonic, and Chromatic.

Grid lines are color-coded: root notes highlighted, diatonic notes labeled, non-diatonic notes dimmed.

## Screenshot

*Piano roll with D Harmonic Minor scale highlighting, pitch line in green, solfege + letter notation.*

## Requirements

- Windows 10/11
- Python 3.10+
- Audio input device (microphone)

## Quick Start

### Option A: Automated Setup

1. **Install the environment** — double-click `install_env.bat`
   - Finds Python 3.10+ automatically (tries `py` launcher, conda, PATH)
   - Creates a local virtual environment in `python-embedded/`
   - Installs all dependencies from `requirements.txt`

2. **Run the tuner** — double-click `run.bat`

### Option B: Manual Setup

```bash
python -m venv python-embedded
python-embedded\Scripts\python.exe -m pip install -r requirements.txt
python-embedded\Scripts\pythonw.exe tuner-2.py
```

### Option C: Using conda

```bash
conda create -n tuner python=3.10
conda activate tuner
pip install -r requirements.txt
python tuner-2.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `install_env.bat` | Create virtual environment and install dependencies |
| `run.bat` | Launch the tuner (no console window) |
| `run_debug.bat` | Launch with console output for debugging |
| `build.bat` | Build standalone Windows executable with Nuitka |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical operations for audio processing |
| sounddevice | 0.5.1 | Audio I/O (ASIO/WASAPI) |
| PySide6 | 6.8.2 | Qt GUI framework + OpenGL widget |
| Nuitka | 2.6.4 | Standalone executable compilation (build only) |

## Architecture

Single-file application (`tuner-2.py`, ~3600 lines) following an MVC pattern with multi-threaded audio:

```
MainWindow (Controller)
├── AudioStreamWorker (QThread) ── audio capture + pitch detection loop
│   └── AudioStreamCallbackHandler ── device I/O, ring buffer, ASIO/WASAPI
├── PianoRollWidget (QOpenGLWidget) ── hardware-accelerated scrolling display
└── PitchInfoPanel (QWidget) ── note display, sliders, musical staff
```

**Signal flow:** Audio callback fills ring buffer → worker thread reads frames, applies gain, runs pitch detection → emits `pitchDetected(time, freq, rms)` signal → main thread routes to piano roll and info panel.

**Thread safety:** Ring buffer access synchronized with `QMutex`. No direct cross-thread UI access — all communication via Qt signals/slots.

### Pitch Detection

Seven algorithms with a common interface (audio frame + sample rate → frequency in Hz):

| Algorithm | Strengths |
|-----------|-----------|
| YIN | Fast, good for monophonic voice |
| pYIN | Multi-threshold probabilistic, more robust |
| MPM | McLeod Pitch Method, good harmonic handling |
| SWIPE | Spectral, handles noise well |
| Cepstrum | Frequency-domain, good for harmonic-rich signals |
| HPS | Harmonic Product Spectrum, strong fundamental detection |
| Autocorrelation | Classic time-domain method |

### Scale Spelling System

Scales are generated from lookup tables (not chromatic index arithmetic) to produce correctly spelled note names:

- **Major/Minor scales**: Hardcoded for all 12 keys following circle-of-fifths conventions
- **Modes**: Derived by rotating the parent major scale
- **Harmonic/Melodic Minor**: Derived by raising degrees from the natural minor
- **Pentatonics**: Subsets of their parent scales

Resolution maps (`ROOT_TO_MAJOR_KEY`, `ROOT_TO_MINOR_KEY`) ensure enharmonic spellings avoid double accidentals.

## Configuration

All settings are stored in `config.json` and auto-saved on exit. Key sections:

- **Audio**: sample rate (48kHz), chunk size, rebuffer size, sensitivity thresholds
- **Display**: MIDI range, colors (hex), font sizes, line widths, dash patterns
- **Scale**: root note, scale type, scale intervals, enharmonic preferences
- **Window**: position, size, debug mode toggle

## Building a Standalone Executable

```bash
build.bat
```

Uses Nuitka with `--standalone`, PySide6 plugin, and LTO. The output is a self-contained folder with `Tuner.exe`. Excludes unused Qt modules (QtQml, QtQuick, QtMultimedia, QtWebEngine) for smaller size.

## License

All rights reserved.

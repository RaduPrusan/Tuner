"""
Real-Time Vocal Pitch Monitor

This application monitors vocal pitch in real time using the microphone as the audio input.
It employs one of two pitch detection algorithms (YIN or Autocorrelation) to determine the
fundamental frequency of incoming audio frames. Audio processing is done in a separate QThread
(AudioStreamWorker) to ensure low-latency processing and responsiveness. The GUI features a
scrolling piano roll display, a pitch information panel (including a simplified musical notation view),
and a set of controls to adjust input source, noise rejection, volume, scroll speed, boost mode,
extra sensitivity, pitch detection method, and scale highlighting.

After the Notation option, two new dropdowns let you select a root note and a scale/mode.
When a scale is selected (anything other than "None"), the horizontal grid lines are colored as follows:
  - The grid line corresponding to the root note is drawn in COLOR_ROOT (yellow, solid).
  - Diatonic (scale) notes are drawn in COLOR_DIA (white, dashed with pattern DASH_PATTERN) and labeled.
  - Non-diatonic notes are drawn in COLOR_NONDIA (dark gray, dashed with pattern DASH_PATTERN).
If "None" is selected as the scale, the grid labels use the same template: the note name (e.g. Do [C])
followed by the octave rendered in a smaller subscript-style font. In non-highlighted mode, the same
line styles are applied, with octave boundaries using COLOR_ROOT and other pitches using COLOR_DIA.

All note displays use the format: Do [C]<sub>2</sub> (with the octave rendered in a smaller font),
using the font sizes defined in the defaults.

Tradeoffs:
  - Lowering the YIN threshold (Extra Sensitivity mode) increases the chance of detecting weak signals,
    but may also increase false detections.
  - Scale highlighting adds visual complexity but is useful for musical analysis.

"""

# --------------------------- Imports ---------------------------
import os
# Set environment variable before importing sounddevice.
os.environ["SD_ENABLE_ASIO"] = "1"
os.environ["SD_ASIO_NONEXCLUSIVE"] = "1"

from datetime import datetime

import sys            # Provides access to some variables and functions used or maintained by the interpreter.
import math           # Provides access to mathematical functions.
import numpy as np    # NumPy is used for efficient numerical operations on large arrays.
import json           # Used for saving and loading configuration.
import sounddevice as sd   # Use sounddevice for audio I/O.
import time

# PySide6 imports:
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMutex, QWaitCondition
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics, QIcon, QTextCursor, QPainterPath, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QFrame,
                               QSlider, QLabel, QHBoxLayout, QVBoxLayout, QComboBox,
                               QCheckBox, QMessageBox, QTextEdit, QToolTip)
from PySide6.QtOpenGLWidgets import QOpenGLWidget


# --------------------------- Music Theory Constants ---------------------------

# Chromatic note name arrays (used for non-scale-context pitch display)
SHARP_LETTER_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_LETTER_NAMES  = ['C', 'D♭', 'D', 'E♭', 'E', 'F', 'G♭', 'G', 'A♭', 'A', 'B♭', 'B']
SHARP_SOLFEGE_NAMES = ['Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']
FLAT_SOLFEGE_NAMES = ['Do', 'Re♭', 'Re', 'Mi♭', 'Mi', 'Fa', 'Sol♭', 'Sol', 'La♭', 'La', 'Si♭', 'Si']

# Correctly spelled major scales for all 12 keys (circle of fifths)
MAJOR_SCALES = {
    "C":  ["C", "D", "E", "F", "G", "A", "B"],
    "G":  ["G", "A", "B", "C", "D", "E", "F#"],
    "D":  ["D", "E", "F#", "G", "A", "B", "C#"],
    "A":  ["A", "B", "C#", "D", "E", "F#", "G#"],
    "E":  ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "B":  ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "F#": ["F#", "G#", "A#", "B", "C#", "D#", "E#"],
    "F":  ["F", "G", "A", "B♭", "C", "D", "E"],
    "B♭": ["B♭", "C", "D", "E♭", "F", "G", "A"],
    "E♭": ["E♭", "F", "G", "A♭", "B♭", "C", "D"],
    "A♭": ["A♭", "B♭", "C", "D♭", "E♭", "F", "G"],
    "D♭": ["D♭", "E♭", "F", "G♭", "A♭", "B♭", "C"],
    "G♭": ["G♭", "A♭", "B♭", "C♭", "D♭", "E♭", "F"],
}

# Correctly spelled natural minor scales for all 12 keys
NATURAL_MINOR_SCALES = {
    "A":  ["A", "B", "C", "D", "E", "F", "G"],
    "E":  ["E", "F#", "G", "A", "B", "C", "D"],
    "B":  ["B", "C#", "D", "E", "F#", "G", "A"],
    "F#": ["F#", "G#", "A", "B", "C#", "D", "E"],
    "C#": ["C#", "D#", "E", "F#", "G#", "A", "B"],
    "G#": ["G#", "A#", "B", "C#", "D#", "E", "F#"],
    "D":  ["D", "E", "F", "G", "A", "B♭", "C"],
    "G":  ["G", "A", "B♭", "C", "D", "E♭", "F"],
    "C":  ["C", "D", "E♭", "F", "G", "A♭", "B♭"],
    "F":  ["F", "G", "A♭", "B♭", "C", "D♭", "E♭"],
    "B♭": ["B♭", "C", "D♭", "E♭", "F", "G♭", "A♭"],
    "E♭": ["E♭", "F", "G♭", "A♭", "B♭", "C♭", "D♭"],
    # A♭ natural minor (enharmonic alt for G# when harmonic/melodic would need ##)
    "A♭": ["A♭", "B♭", "C♭", "D♭", "E♭", "F♭", "G♭"],
}

# Map composite root notes to preferred enharmonic for major keys
# Avoids double accidentals by choosing the simpler spelling
ROOT_TO_MAJOR_KEY = {
    "C": "C", "G": "G", "D": "D", "A": "A", "E": "E", "B": "B",
    "F#": "F#", "G♭": "G♭", "F": "F",
    "B♭": "B♭", "E♭": "E♭", "A♭": "A♭", "D♭": "D♭",
    # Enharmonic redirects (prefer fewer accidentals)
    "C#": "D♭", "D#": "E♭", "G#": "A♭", "A#": "B♭",
}

# Map composite root notes to preferred enharmonic for minor keys
ROOT_TO_MINOR_KEY = {
    "A": "A", "E": "E", "B": "B", "F#": "F#", "C#": "C#", "G#": "G#",
    "D": "D", "G": "G", "C": "C", "F": "F",
    "B♭": "B♭", "E♭": "E♭",
    # Enharmonic redirects (avoid double accidentals in harmonic/melodic)
    "D#": "E♭", "A#": "B♭",
    "G♭": "F#", "D♭": "C#", "A♭": "G#",
}

# Mode offsets: semitones from mode root down to parent major root
MODE_OFFSETS = {
    "Ionian": 0,
    "Dorian": -2,
    "Phrygian": -4,
    "Lydian": -5,
    "Mixolydian": -7,
    "Aeolian": -9,
    "Locrian": -11,
}

# Sharp major keys (for determining sharp vs flat context)
SHARP_MAJOR_KEYS = {"C", "G", "D", "A", "E", "B", "F#"}


# --------------------------- Defaults Class ---------------------------
class Defaults:
    """
    This class encapsulates all default constants and configuration values.
    Methods:
        InitDefaults() - Loads or resets values to defaults.
        LoadConfigFromFile() - Load configuration from a file.
        SaveConfigToFile() - Save current configuration to a file.
        updateWindowGeometry(window) - Update window geometry in the defaults based on the given window.
    """
    def InitDefaults(self):
        # Enable debug logging (if True, a debug log text box will be shown in the GUI)
        self.ENABLE_DEBUG = True

        # Window and Device Defaults
        self.DEFAULT_WINDOW_WIDTH = 840          # Initial width of the main window.
        self.DEFAULT_WINDOW_HEIGHT = 1400        # Initial height of the main window.
        self.DEFAULT_WINDOW_POS_X = 350          # Initial X position of the window.
        self.DEFAULT_WINDOW_POS_Y = 350          # Initial Y position of the window.
        self.WINDOW_OFFSET_Y = 0                 # Offset between the window frame and the client area.
        self.DEFAULT_INPUT_SOURCE_INDEX = None   # The default input source index; determined dynamically.

        # Control Defaults
        self.DEFAULT_SENSITIVITY = 5             # Default sensitivity percentage (for noise rejection).
        self.DEFAULT_VOLUME_DB = 10              # Default volume preamplification in decibels.
        self.DEFAULT_SCROLL_SPEED = 140          # Default scroll speed (pixels per second) for the piano roll display.
        self.DEFAULT_ENHARMONIC_NOTATION = "Sharps"  # Default notation style for enharmonic notes.
        self.DEFAULT_PITCH_METHOD = "YIN"        # Default pitch detection algorithm ("YIN" or "Autocorrelation").
        self.current_pitch_method = self.DEFAULT_PITCH_METHOD  # Persisted pitch method selection.
        self.DEFAULT_ROOT_NOTE = "C"             # Default root note for scale highlighting.
        self.DEFAULT_SCALE = "-"              # Default scale highlighting mode ("None" means no highlighting).

        # Font Sizes used for labels in different areas of the interface.
        self.GRID_FONT_FAMILY = ""        # Font family is now defined in the CSS file, empty string here for defaults.
        self.FONT_SIZE_NOTE_LABEL = 38           # Font size for note display in the Pitch Information Panel.
        self.FONT_SIZE_FREQ_LABEL = 20           # Font size for frequency display.
        self.FONT_SIZE_CENTS_LABEL = 20          # Font size for cents (pitch deviation) display.
        self.GRID_FONT_MAIN_SIZE = 13            # Font size for the main part of grid labels.
        self.GRID_FONT_SUB_SIZE = 10              # Font size for the subscript (octave) part of grid labels.
        self.GRID_FONT_LABEL_SIZE = 13           # Font size for pitch label in the piano roll.

        # Pitch label display options
        self.PITCH_LABEL_SOLFEGE = True          # If True, display solfege names.
        self.PITCH_LABEL_LETTERS = False         # If True, display letter names.
        self.PITCH_LABEL_OCTAVE = True           # If True, display the octave number.

        # Layout Defaults (Spacing and Paddings)
        self.DEFAULT_PADDING = 10                # Padding around the piano roll area.
        self.DEFAULT_VERT_PADDING = 30                # Vertical padding for the piano roll area
        self.DEFAULT_RIGHT_MARGIN = 70          # Extra margin on the right side for labels.

        # Staff display (for the Pitch Info Panel) defaults:
        self.DEFAULT_STAFF_TOP_OFFSET = 450      # Y-coordinate for the top of the upper (G-clef) staff.
        self.DEFAULT_STAFF_HEIGHT = 40           # Height allocated for each staff region.
        self.STAFF_MAIN_COLOR = QColor(180, 180, 180)  # Color used for drawing staff lines.
        self.STAFF_CLEF_COLOR = QColor(250, 250, 250)  # Color used for drawing clef symbols.
        self.STAFF_NOTE_COLOR = QColor(137, 198, 42)     # Color used to draw note heads on the staff.

        # MIDI Range for Display (limiting the vertical span of the piano roll).
        self.MIDI_START = 24 + 6                 # Minimum MIDI note (after an offset).
        self.MIDI_END = 84 - 6                   # Maximum MIDI note (after an offset).

        # Scale Highlighting Colors and Dash Pattern for grid lines:
        #self.COLOR_ROOT = QColor(205, 190, 80)
        self.COLOR_ROOT = QColor(235, 235, 235)
        self.COLOR_DIA = QColor(160, 160, 160)
        self.COLOR_NONDIA = QColor(70, 70, 70)
        self.COLOR_PAUSED = QColor(54, 150, 180)
        self.COLOR_TIME_LINES = QColor(60, 60, 60)
        self.PITCH_LINE_ALPHA_ADJUSTMENT = False
        self.PITCH_LINE_DRAW_COLOR = QColor(127, 220, 32)
        self.PITCH_LABEL_ENABLED = False
        self.PITCH_LABEL_COLOR  = QColor(127, 220, 32)
        self.VUMETER_FILL_COLOR = QColor(127, 220, 32)
        self.DRAW_DETECTED_LINE = True
        self.COLOR_DETECTED = QColor(127, 220, 32)
        self.DASH_PATTERN = [2, 3]
        self.DASH_PATTERN_VERTICAL = [2, 3]
        self.PITCH_LINE_WIDTH = 3
        self.GRID_LINE_WIDTH = 1
        self.GRID_LINE_WIDTH_ROOT = 2

        # Data retention: maximum age (in seconds) of data points kept in the canvas.
        self.MAX_DATA_AGE = 120

        # --- New Hardware Acceleration parameters ---
        self.ENABLE_HW_ACCELERATION = True       # Enable hardware accelerated rendering (uses QOpenGLWidget)
        self.ENABLE_VSYNC = True                 # Enable VSync in OpenGL rendering
        self.ANTIALIASING_SAMPLES = 4            # Number of antialiasing samples to use

        # Audio Processing Defaults
        self.SAMPLE_RATE = 48000                 # Audio sample rate in Hertz.
        # To get a similar window length as the 45ms window at 22050Hz,
        # use roughly 2048 samples at 48000 Hz.
        self.CHUNK_SIZE = 2048                   # Number of audio samples per processing block.
        self.REBUFFER_SIZE = 4096                # Maximum number of samples to keep in the ring buffer.
        self.MIN_SCROLL_SPEED = 10               # Minimum allowed scroll speed (pixels per second).
        self.MAX_SCROLL_SPEED = 200              # Maximum allowed scroll speed (pixels per second).
        self.MIN_THRESHOLD = 0.001               # Minimum noise threshold (0% sensitivity).
        self.MAX_THRESHOLD = 0.04                # Maximum noise threshold (100% sensitivity).

        self.SILENT_DURATION = 0.5               # Duration (in seconds) of silence before triggering auto-pause and a break.

        self.ENABLE_FORCED_PIANOROLL_UPDATE = False
        self.DEFAULT_PIANO_ROLL_UPDATE_FPS = 60  # Number of updates per second for the Piano Roll display.
        self.MAX_AUDIO_WORKER_LOOP_HZ = 60       # Maximum frequency (in Hz) for AudioStreamWorker loop iterations.

        self._FMAX = 1000
        self._FMIN = 30

        self.TUNING_FREQUENCY = 440.0            # Reference tuning frequency for A4 (in Hertz).

        # YIN algorithm tuning parameters
        self.YIN_DEFAULT_THRESHOLD = 0.1
        self.YIN_SENSITIVE_THRESHOLD = 0.07

        # Probabilistic YIN (pYIN) algorithm tuning parameters
        self.PYIN_THRESHOLD_RANGE = (0.1, 0.35)  # Defines the range of threshold values to explore for pYIN.
        self.PYIN_THRESHOLD_STEPS = 3            # Number of discrete threshold values to evaluate.
        self.PYIN_MIN_PEAK_VAL = 0.95            # Increased from 0.90 to 0.95 to require a stronger peak.
        self.PYIN_STABILITY_WINDOW = 5           # Number of consecutive frames required for pitch stability.
        self.PYIN_CONFIDENCE_THRESHOLD = 0.85    # Minimum confidence level required for a candidate to be accepted.
        self.PYIN_MIN_RMS = 0.002                # Minimum RMS threshold for noise gating.

        # Smoothing parameters for pitch detection (used in AudioStreamWorker)
        self.SMOOTHING_ENABLED = True            # Enable or disable pitch smoothing
        self.SMOOTHING_THRESHOLD_RATIO = 0.8     # New pitch must be at least 80% of previous pitch to be accepted
        self.SMOOTHING_COUNT_THRESHOLD = 2       # Number of consecutive outlier frames before accepting new pitch

        # Note/Scale Data
        self.global_prefer_sharps = True         # Global flag indicating whether sharps are preferred in the notation.
        self.current_root_note = self.DEFAULT_ROOT_NOTE  # Global variable for the currently selected root note.
        self.current_scale = self.DEFAULT_SCALE  # Global variable for the currently selected scale/mode.
        self.NOTE_TO_INDEX = {
            "C": 0,
            "C#/D♭": 1,
            "D": 2,
            "D#/E♭": 3,
            "E": 4,
            "F": 5,
            "F#/G♭": 6,
            "G": 7,
            "G#/A♭": 8,
            "A": 9,
            "A#/B♭": 10,
            "B": 11
        }
        self.SCALE_INTERVALS = {
            "Chromatic": list(range(12)),
            "Major": [0, 2, 4, 5, 7, 9, 11],
            "Natural Minor": [0, 2, 3, 5, 7, 8, 10],
            "Harmonic Minor": [0, 2, 3, 5, 7, 8, 11],
            "Melodic Minor": [0, 2, 3, 5, 7, 9, 11],
            "Ionian": [0, 2, 4, 5, 7, 9, 11],
            "Dorian": [0, 2, 3, 5, 7, 9, 10],
            "Phrygian": [0, 1, 3, 5, 7, 8, 10],
            "Lydian": [0, 2, 4, 6, 7, 9, 11],
            "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "Aeolian": [0, 2, 3, 5, 7, 8, 10],
            "Locrian": [0, 1, 3, 5, 6, 8, 10],
            "Major Pentatonic": [0, 2, 4, 7, 9],
            "Minor Pentatonic": [0, 3, 5, 7, 10]
        }
        self.SCALE_OPTIONS = [
            "-", "Chromatic", "Major", "Natural Minor", "Harmonic Minor", "Melodic Minor",
            "Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian",
            "Major Pentatonic", "Minor Pentatonic"
        ]

        # Maintain a set of attribute names that are QColors.
        self._color_attrs = {
            "STAFF_MAIN_COLOR", "STAFF_CLEF_COLOR", "STAFF_NOTE_COLOR",
            "COLOR_ROOT", "COLOR_DIA", "COLOR_NONDIA", "COLOR_PAUSED",
            "PITCH_LINE_DRAW_COLOR", "PITCH_LABEL_COLOR", "VUMETER_FILL_COLOR",
            "COLOR_DETECTED", "COLOR_TIME_LINES"
        }

        # Version tracking for configuration
        self.CONFIG_VERSION = 2  # Increment when defaults change significantly

    def generate_scale(self, root_note=None, scale_type=None, with_solfege=False):
        """
        Generate a correctly spelled scale using lookup tables.

        Args:
            root_note (str, optional): The root note. Defaults to current_root_note.
            scale_type (str, optional): The scale type. Defaults to current_scale.
            with_solfege (bool, optional): Return solfege names instead. Defaults to False.

        Returns:
            list: Correctly spelled note names for the scale.
        """
        if root_note is None:
            root_note = self.current_root_note
        if scale_type is None:
            scale_type = self.current_scale

        if scale_type == "-":
            return []

        # Parse composite root: "C#/D♭" -> use first part's canonical form
        # For non-composite roots, preserve the original form (e.g., "G♭" stays "G♭")
        if "/" in root_note:
            parts = root_note.split("/")
            raw_root = canonical_note(parts[0])
        else:
            raw_root = root_note

        scale_notes = self._build_scale(raw_root, scale_type)

        if with_solfege:
            return [note_to_solfege(n) for n in scale_notes]
        return scale_notes

    def _build_scale(self, raw_root, scale_type):
        """
        Internal: build the correctly spelled scale for the given root and type.
        raw_root may be in sharp or flat form (e.g., "F#" or "G♭").
        """
        # Canonical form for index-based operations; original form for resolution maps
        canon_root = canonical_note(raw_root)

        # --- Chromatic: use global preference ---
        if scale_type == "Chromatic":
            if defaults.global_prefer_sharps:
                return list(SHARP_LETTER_NAMES)
            else:
                return list(FLAT_LETTER_NAMES)

        # --- Major scale: direct lookup ---
        if scale_type == "Major":
            resolved = ROOT_TO_MAJOR_KEY.get(raw_root, ROOT_TO_MAJOR_KEY.get(canon_root, raw_root))
            return list(MAJOR_SCALES.get(resolved, MAJOR_SCALES["C"]))

        # --- Natural Minor: direct lookup ---
        if scale_type == "Natural Minor":
            resolved = ROOT_TO_MINOR_KEY.get(raw_root, ROOT_TO_MINOR_KEY.get(canon_root, raw_root))
            return list(NATURAL_MINOR_SCALES.get(resolved, NATURAL_MINOR_SCALES["A"]))

        # --- Harmonic Minor: natural minor with raised 7th ---
        if scale_type == "Harmonic Minor":
            resolved = ROOT_TO_MINOR_KEY.get(raw_root, ROOT_TO_MINOR_KEY.get(canon_root, raw_root))
            minor = list(NATURAL_MINOR_SCALES.get(resolved, NATURAL_MINOR_SCALES["A"]))
            try:
                minor[6] = _raise_note(minor[6])
            except ValueError:
                # Double sharp needed — switch to enharmonic key directly
                # (bypass ROOT_TO_MINOR_KEY to avoid looping back to the sharp key)
                alt_root = self._enharmonic_minor_alt(canon_root)
                minor = list(NATURAL_MINOR_SCALES.get(alt_root, NATURAL_MINOR_SCALES["A"]))
                minor[6] = _raise_note(minor[6])
            return minor

        # --- Melodic Minor: natural minor with raised 6th and 7th ---
        if scale_type == "Melodic Minor":
            resolved = ROOT_TO_MINOR_KEY.get(raw_root, ROOT_TO_MINOR_KEY.get(canon_root, raw_root))
            minor = list(NATURAL_MINOR_SCALES.get(resolved, NATURAL_MINOR_SCALES["A"]))
            try:
                minor[5] = _raise_note(minor[5])
                minor[6] = _raise_note(minor[6])
            except ValueError:
                alt_root = self._enharmonic_minor_alt(canon_root)
                minor = list(NATURAL_MINOR_SCALES.get(alt_root, NATURAL_MINOR_SCALES["A"]))
                minor[5] = _raise_note(minor[5])
                minor[6] = _raise_note(minor[6])
            return minor

        # --- Modes: find parent major, rotate ---
        if scale_type in MODE_OFFSETS:
            try:
                root_index = SHARP_LETTER_NAMES.index(canon_root)
            except ValueError:
                root_index = 0
            parent_index = (root_index + MODE_OFFSETS[scale_type]) % 12
            parent_root = SHARP_LETTER_NAMES[parent_index]
            parent_resolved = ROOT_TO_MAJOR_KEY.get(parent_root, parent_root)
            parent_scale = list(MAJOR_SCALES.get(parent_resolved, MAJOR_SCALES["C"]))

            # Find the mode root in the parent scale
            for i, note in enumerate(parent_scale):
                if canonical_note(note) == canon_root:
                    return parent_scale[i:] + parent_scale[:i]

            # Fallback: return parent scale unrotated
            return parent_scale

        # --- Major Pentatonic: degrees 1,2,3,5,6 of major ---
        if scale_type == "Major Pentatonic":
            major = self._build_scale(raw_root, "Major")
            return [major[i] for i in (0, 1, 2, 4, 5)]

        # --- Minor Pentatonic: degrees 1,3,4,5,7 of natural minor ---
        if scale_type == "Minor Pentatonic":
            minor = self._build_scale(raw_root, "Natural Minor")
            return [minor[i] for i in (0, 2, 3, 4, 6)]

        # Fallback: major scale
        return list(MAJOR_SCALES.get("C"))

    def _enharmonic_minor_alt(self, canon_root):
        """Find the enharmonic alternative root for a minor key to avoid double sharps."""
        try:
            idx = SHARP_LETTER_NAMES.index(canon_root)
        except ValueError:
            return canon_root
        # Get the flat name for the same pitch
        return FLAT_LETTER_NAMES[idx] if FLAT_LETTER_NAMES[idx] != SHARP_LETTER_NAMES[idx] else canon_root

    def LoadConfigFromFile(self, filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")):
        """
        Loads configuration values from a JSON file and updates defaults.
        All keys are expected to be in lowercase.

        Args:
            filename (str): The path to the configuration file.
        """
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    loaded_config = json.load(f)
                # Loop over all public attributes (skip ones starting with "_")
                for attr in self.__dict__:
                    if attr.startswith("_"):
                        continue
                    config_key = attr.lower()
                    if config_key in loaded_config:
                        value = loaded_config[config_key]
                        if attr in self._color_attrs and isinstance(value, str):
                            setattr(self, attr, QColor(value))
                        else:
                            setattr(self, attr, value)
                print(f"Config loaded from file {filename}")
            else:
                print(f"Config file {filename} not found. Using default values.")
        except Exception as e:
            print(f"Error loading config from {filename}: {e}")

    def SaveConfigToFile(self, filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")):
        """
        Saves the current configuration values to a JSON file.
        All keys are saved in lowercase.

        Args:
            filename (str): The path to the configuration file.
        """
        try:
            config_to_save = {}
            # Only save public attributes (skip any that start with "_")
            for attr, v in self.__dict__.items():
                if attr.startswith("_") or callable(v):
                    continue
                if attr in self._color_attrs and isinstance(v, QColor):
                    config_to_save[attr.lower()] = v.name()
                else:
                    config_to_save[attr.lower()] = v
            with open(filename, "w") as f:
                json.dump(config_to_save, f, indent=4, default=str)
        except Exception as e:
            print(f"Error saving config to {filename}: {e}")

    def updateWindowGeometry(self, window):
        """
        Updates the window geometry defaults based on the given window.

        Args:
            window (QMainWindow): The main application window.
        """
        try:

            # Get the full frame geometry (includes window decorations)
            frame = window.frameGeometry()
            # And the client geometry (content area)
            client = window.geometry()
            offset_y = frame.top() - client.top()

            self.DEFAULT_WINDOW_POS_X = client.x()
            self.DEFAULT_WINDOW_POS_Y = client.y()
            self.DEFAULT_WINDOW_WIDTH = client.width()
            self.DEFAULT_WINDOW_HEIGHT = client.height()

            self.WINDOW_OFFSET_Y = offset_y

        except Exception as e:
            print(f"Error updating window geometry: {e}")

    def load_config(self, config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Version check and upgrade
            if config.get("CONFIG_VERSION", 1) < self.CONFIG_VERSION:
                # Force update of critical parameters
                config["CHUNK_SIZE"] = self.CHUNK_SIZE
                config["REBUFFER_SIZE"] = self.REBUFFER_SIZE
                config["CONFIG_VERSION"] = self.CONFIG_VERSION

            # Validate numeric parameters
            if config.get("CHUNK_SIZE", 0) < 128:
                config["CHUNK_SIZE"] = self.CHUNK_SIZE
            if config.get("REBUFFER_SIZE", 0) < config["CHUNK_SIZE"]:
                config["REBUFFER_SIZE"] = self.REBUFFER_SIZE

            return config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return {}


# Create a global instance of Defaults and initialize it.
defaults = Defaults()
defaults.InitDefaults()


# --------------------------- Helper Function for Drawing Adjusted Text ---------------------------
def drawAdjustedText(painter, x, y, text, flat_char="♭", adjustment=-2):
    """
    Draws a text string with adjusted spacing for each occurrence of flat_char.
    The adjustment value is added to the x position when drawing the flat character.

    Parameters:
        painter (QPainter): The painter to draw on.
        x, y (int): Starting coordinates.
        text (str): The full text string.
        flat_char (str): The character to adjust (default is "♭").
        adjustment (int): The x offset adjustment to apply for the flat character.
    """
    font = painter.font()
    metrics = QFontMetrics(font)
    current_x = x
    parts = text.split(flat_char)
    for i, part in enumerate(parts):
        painter.drawText(current_x, y, part)
        current_x += metrics.horizontalAdvance(part)
        if i < len(parts) - 1:
            painter.drawText(current_x + adjustment, y, flat_char)
            current_x += metrics.horizontalAdvance(flat_char) - 5
    return current_x - x



# New function to format notes with HTML for better flat symbol handling
def format_note_with_html(text, flat_char="♭"):
    """
    Formats a text string by wrapping each occurrence of flat_char in a span with a specified class.
    This allows targeting the flat symbol with CSS for better spacing control.
    
    Parameters:
        text (str): The text string to format.
        flat_char (str): The character to wrap (default is "♭").
        flat_class (str): The CSS class to apply to the span.
        
    Returns:
        str: HTML formatted string with spans around flat characters.
    """
    if flat_char not in text:
        return text
        
    parts = text.split(flat_char)
    result = parts[0]
    
    for i in range(1, len(parts)):
        result += f"<span style='letter-spacing: -20px;'> </span><span style='letter-spacing: -10px;'>{flat_char}</span>" + parts[i]
        #result += f"{flat_char}" + parts[i]

    return result





# --------------------------- Helper Function for Loading Stylesheets ---------------------------
def load_stylesheet(file_path="style.css"):
    """
    Loads a stylesheet from a file.

    Args:
        file_path (str): Path to the stylesheet file.

    Returns:
        str: Contents of the stylesheet, or an empty string if an error occurs.
    """
    complete_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), file_path)
    if not os.path.exists(complete_path):
        print(f"Stylesheet file '{file_path}' not found.")
        return ""
    try:
        with open(complete_path, 'r') as f:
            stylesheet = f.read()
        return stylesheet
    except Exception as e:
        print("Error loading stylesheet:", e)
        return ""


# --------------------------- Helper Functions for Scale Spelling ---------------------------
def _raise_note(note):
    """
    Raise a note by one chromatic step while preserving its letter name.
    Used for deriving harmonic/melodic minor from natural minor.

    bb -> b, b -> natural, natural -> #
    Raises ValueError if result would be ## (double sharp).
    """
    letter = note[0]
    accidental = note[1:] if len(note) > 1 else ''

    if accidental == '♭♭':       # double flat -> flat
        return letter + '♭'
    elif accidental == '♭':           # flat -> natural
        return letter
    elif accidental == '':                 # natural -> sharp
        return letter + '#'
    elif accidental == '#':                # sharp -> double sharp (avoid)
        raise ValueError(f"Raising {note} would produce double sharp {letter}##")
    else:
        raise ValueError(f"Unknown accidental in note: {note}")


def use_sharps_for_key():
    """
    Determines whether to use sharp or flat notation based on the current scale and root note.
    Resolves the effective key and checks if it's a sharp-side or flat-side key.

    Returns:
        bool: True if sharps should be used; False if flats should be used.
    """
    root_note = defaults.current_root_note
    scale_type = defaults.current_scale

    if scale_type == "-" or scale_type == "Chromatic":
        return defaults.global_prefer_sharps

    # Parse composite root: "C#/D♭" -> try first part
    if "/" in root_note:
        parts = root_note.split("/")
        raw_root = canonical_note(parts[0])
    else:
        raw_root = canonical_note(root_note)

    # Determine which key we actually resolve to
    if scale_type in MODE_OFFSETS:
        # Find parent major key
        try:
            root_index = SHARP_LETTER_NAMES.index(raw_root)
        except ValueError:
            root_index = 0
        parent_index = (root_index + MODE_OFFSETS[scale_type]) % 12
        parent_root = SHARP_LETTER_NAMES[parent_index]
        # Check if parent maps to a flat key
        resolved = ROOT_TO_MAJOR_KEY.get(parent_root, parent_root)
        return resolved in SHARP_MAJOR_KEYS

    elif scale_type == "Major" or scale_type == "Major Pentatonic":
        resolved = ROOT_TO_MAJOR_KEY.get(raw_root, raw_root)
        return resolved in SHARP_MAJOR_KEYS

    elif scale_type in ("Natural Minor", "Harmonic Minor", "Melodic Minor", "Minor Pentatonic"):
        resolved = ROOT_TO_MINOR_KEY.get(raw_root, raw_root)
        # Minor key is sharp if its relative major is sharp
        # Relative major is 3 semitones up
        try:
            minor_index = SHARP_LETTER_NAMES.index(canonical_note(resolved))
        except ValueError:
            minor_index = 0
        rel_major_index = (minor_index + 3) % 12
        rel_major_root = SHARP_LETTER_NAMES[rel_major_index]
        rel_major_resolved = ROOT_TO_MAJOR_KEY.get(rel_major_root, rel_major_root)
        return rel_major_resolved in SHARP_MAJOR_KEYS

    return defaults.global_prefer_sharps


# --------------------------- Enharmonic Mapping Functions ---------------------------
def canonical_note(note):
    """
    Convert note representations to their canonical sharp representations.
    Handles standard accidentals, double sharps, and double flats.
    
    Args:
        note (str): The note string which might be in flat notation.
        
    Returns:
        str: The canonical (sharp) representation of the note.
    """
    # Map all enharmonic equivalents to a canonical form
    enharmonic_to_canonical = {
        # Standard flats to sharps
        'D♭': 'C#',
        'E♭': 'D#',
        'G♭': 'F#',
        'A♭': 'G#',
        'B♭': 'A#',
        # Natural notes with flats/sharps
        'F♭': 'E',
        'E#': 'F',
        'C♭': 'B',
        'B#': 'C',
        # Double sharps and flats
        'C##': 'D',
        'D##': 'E',
        'E##': 'F#',
        'F##': 'G',
        'G##': 'A',
        'A##': 'B',
        'B##': 'C#',
        'C♭♭': 'A#',
        'D♭♭': 'C',
        'E♭♭': 'D',
        'F♭♭': 'D#',
        'G♭♭': 'F',
        'A♭♭': 'G',
        'B♭♭': 'A'
    }
    # Remove any octave numbers before lookup
    base_note = note.rstrip('0123456789')
    octave = ''.join(c for c in note if c.isdigit())
    
    # Get the canonical form of the base note
    canonical = enharmonic_to_canonical.get(base_note, base_note)
    
    # Reattach octave number if present
    if octave:
        return canonical + octave
    return canonical


# --------------------------- Utility Functions ---------------------------
def note_to_solfege(note_name):
    """
    Convert a note name to its solfege equivalent, handling enharmonics, double sharps, and double flats.
    
    Args:
        note_name (str): The note name (e.g., 'C#', 'B#', 'G##')
        
    Returns:
        str: The corresponding solfege name (e.g., 'Do#', 'Si#', 'Sol##')
    """
    # Base solfege syllables
    base_solfege = {
        'C': 'Do',
        'D': 'Re',
        'E': 'Mi',
        'F': 'Fa',
        'G': 'Sol',
        'A': 'La',
        'B': 'Si'
    }
    
    # Handle notes with or without octave numbers
    base_note = note_name.rstrip('0123456789')
    octave = ''.join(c for c in note_name if c.isdigit())
    
    # Separate the letter from accidentals
    letter = base_note[0]
    accidentals = base_note[1:] if len(base_note) > 1 else ''
    
    # Convert the letter to solfege
    solfege = base_solfege.get(letter, letter)  # Fallback to the letter if not found
    
    # Append the accidentals to the solfege name
    solfege_with_accidentals = solfege + accidentals
    
    # If there was an octave, add it back
    if octave:
        solfege_with_accidentals += octave
        
    return solfege_with_accidentals

def frequency_to_note(frequency, prefer_sharps=True):
    if frequency <= 0:
        return "", "", -1, 0
    midi = 69 + 12 * math.log2(frequency / defaults.TUNING_FREQUENCY)
    midi_int = int(round(midi))
    cents = int(round((midi - midi_int) * 100))

    # Compute the exact (equal-tempered) frequency corresponding to the computed MIDI note.
    et_frequency = defaults.TUNING_FREQUENCY * (2 ** ((midi_int - 69) / 12))
    if midi_int == 69:
        et_frequency = defaults.TUNING_FREQUENCY

    one_cent = et_frequency * (2 ** (1/1200) - 1)
    if abs(frequency - et_frequency) < one_cent / 2:
        frequency = et_frequency
        cents = 0

    note_index = midi_int % 12
    octave_val = (midi_int // 12) - 1

    # Scale-aware note naming: if a scale is active, use scale spelling
    letter = None
    if defaults.current_scale not in ("-", "Chromatic"):
        scale_notes = defaults.generate_scale()
        canonical_pitch = SHARP_LETTER_NAMES[note_index]
        for scale_note in scale_notes:
            if canonical_note(scale_note) == canonical_pitch:
                letter = scale_note
                break

    # Fallback to chromatic array
    if letter is None:
        if prefer_sharps:
            letter = SHARP_LETTER_NAMES[note_index]
        else:
            letter = FLAT_LETTER_NAMES[note_index]

    solfege = note_to_solfege(letter)

    note_text = ""
    if defaults.PITCH_LABEL_SOLFEGE:
        note_text += f"{solfege}"
    if defaults.PITCH_LABEL_LETTERS:
        note_text += f" {letter}"

    note_no_octave = note_text + f" ({frequency:.1f} Hz)"

    if defaults.PITCH_LABEL_OCTAVE:
        full_note = note_text + f"<sub>{octave_val}</sub>"
    else:
        full_note = note_text

    return full_note, note_no_octave, midi_int, cents


def yin_pitch(frame, sample_rate, fmin=50, fmax=1000, threshold=0.1):
    """
    Implements the YIN pitch detection algorithm on an audio frame.

    Args:
        frame (np.array): The audio frame as a NumPy array.
        sample_rate (int): The audio sample rate.
        fmin (float): Minimum frequency to consider.
        fmax (float): Maximum frequency to consider.
        threshold (float): Threshold for cumulative mean normalized difference.

    Returns:
        float: Detected pitch (in Hz) or 0.0 if no pitch is detected.
    """
    frame = frame.astype(np.float32)
    N = len(frame)
    max_tau = int(sample_rate / fmin)
    if max_tau >= N:
        max_tau = N - 1
    # Create an array of potential lags (tau) values.
    tau_range = np.arange(1, max_tau)
    # Calculate the squared difference (difference function) for each lag.
    d = np.array([np.sum((frame[:-tau] - frame[tau:])**2) / (N - tau) for tau in tau_range])
    # Compute CMND vectorized:
    cmnd = np.empty_like(d)
    cmnd[0] = 1.0
    if len(d) > 1:
        cumulative = np.cumsum(d[1:])
        indices = np.arange(1, len(d))
        cmnd[1:] = np.where(cumulative == 0, 1.0, d[1:] * indices / cumulative)

    tau_est = None
    candidate = np.nonzero(cmnd < threshold)[0]
    if candidate.size > 0:
        tau = candidate[0]
        # Continue forward if the next CMND value is lower
        while (tau + 1) < len(cmnd) and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
        tau_est = tau + 1
    if tau_est is None:
        return 0.0
    # Parabolic interpolation for a more precise estimation.
    if tau_est > 1 and tau_est < len(cmnd) - 1:
        s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        denom = 2 * (2 * s1 - s2 - s0)
        if denom != 0:
            tau_est = tau_est + (s2 - s0) / denom
    pitch = sample_rate / tau_est
    # Verify that the detected pitch is within the expected bounds.
    if pitch < fmin or pitch > fmax:
        return 0.0
    return pitch


# Updated Probabilistic YIN (pYIN) pitch detection function
def pyin_pitch(audio_frame, sample_rate, fmin=defaults._FMIN, fmax=defaults._FMAX, debug_callback=None):
    """
    Enhanced Probabilistic YIN (pYIN) pitch detection algorithm with debug output.
    """

    frame = audio_frame.astype(np.float32)
    frame = frame - np.mean(frame)  # Remove DC offset

    # Apply Hanning window
    window = np.hanning(len(frame))
    frame = frame * window

    N = len(frame)
    rms = np.sqrt(np.mean(frame ** 2))

    # --- New: Noise gate based on RMS ---
    debug_msg = f"Signal RMS: {rms:.3f}\n"
    if rms < defaults.PYIN_MIN_RMS:
        debug_msg += f"Low RMS ({rms:.3f} < {defaults.PYIN_MIN_RMS}), likely noise so ignoring this frame.\n"
        if debug_callback:
            debug_callback(debug_msg)
        return 0.0
    # --- End new noise gating ---

    debug_msg += f"Frame length: {N}\n"

    # Calculate tau range
    tau_min = max(1, int(sample_rate / fmax))
    tau_max = min(int(sample_rate / (fmin * 0.9)), N // 2)  # Allow slightly lower frequencies
    debug_msg += f"Tau range: {tau_min} to {tau_max}\n"

    # Step 1: Compute the normalized squared differences
    diff = np.zeros(tau_max + 1)
    for tau in range(tau_min, tau_max + 1):
        diff[tau] = np.sum((frame[:-tau] - frame[tau:]) ** 2) / (N - tau)

    # Step 2: Cumulative mean normalized difference function
    cmnd = np.zeros_like(diff)
    cmnd[0] = 1.0

    # Compute normalization factor for each tau
    running_sum = 0.0
    for tau in range(1, len(diff)):
        running_sum += diff[tau]
        cmnd[tau] = tau * diff[tau] / (running_sum + 1e-12) / (N - tau)

    debug_msg += f"CMND range: {np.min(cmnd):.3f} to {np.max(cmnd):.3f}\n"

    # Step 3: Find candidates using absolute threshold method
    candidates = []
    thresholds = np.linspace(defaults.PYIN_THRESHOLD_RANGE[0],
                              defaults.PYIN_THRESHOLD_RANGE[1],
                              defaults.PYIN_THRESHOLD_STEPS)

    for threshold in thresholds:
        for tau in range(tau_min + defaults.PYIN_STABILITY_WINDOW, tau_max - defaults.PYIN_STABILITY_WINDOW):
            window_vals = cmnd[tau - defaults.PYIN_STABILITY_WINDOW:tau + defaults.PYIN_STABILITY_WINDOW + 1]

            # Check if this point is a local minimum
            if (cmnd[tau] < threshold and
                cmnd[tau] == min(window_vals) and
                (1 - cmnd[tau]) > defaults.PYIN_MIN_PEAK_VAL):

                # Parabolic interpolation
                alpha = cmnd[tau - 1]
                beta = cmnd[tau]
                gamma = cmnd[tau + 1]

                if beta < alpha and beta < gamma:
                    # Refined period estimation using parabolic interpolation
                    peak_pos = tau + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)

                    # Calculate confidence metrics
                    peak_height = 1 - beta
                    stability = 1 - np.std(window_vals)

                    pitch = sample_rate / peak_pos
                    confidence = peak_height * stability * (1 - threshold / defaults.PYIN_THRESHOLD_RANGE[1])

                    # Add additional confidence boost for strong peaks
                    if peak_height > 0.8:
                        confidence *= 1.2

                    if fmin <= pitch <= fmax and confidence > defaults.PYIN_CONFIDENCE_THRESHOLD:
                        candidates.append((pitch, confidence))
                        debug_msg += (f"Candidate: {pitch:.1f}Hz (pos:{peak_pos:.1f}, "
                                    f"conf:{confidence:.3f}, th:{threshold:.3f})\n")

    debug_msg += f"Total candidates: {len(candidates)}\n"

    if not candidates:
        debug_msg += "No valid candidates found\n"
        if debug_callback:
            debug_callback(debug_msg)
        return 0.0

    # Sort by confidence and apply final selection
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_pitch = candidates[0][0]
    best_confidence = candidates[0][1]

    # If we have multiple candidates, check for octave errors
    if len(candidates) > 1:
        for pitch, conf in candidates[1:]:
            ratio = max(pitch, best_pitch) / min(pitch, best_pitch)
            # If we have a candidate near double frequency with good confidence
            if 1.9 < ratio < 2.1 and conf > best_confidence * 0.8:
                # Prefer the higher octave if confidence is similar
                if pitch > best_pitch:
                    best_pitch = pitch
                    best_confidence = conf

    debug_msg += f"Selected pitch: {best_pitch:.1f}Hz (conf:{best_confidence:.3f})\n"

    #if debug_callback:
    #    debug_callback(debug_msg)

    return best_pitch


def autocorrelation_pitch(frame, sample_rate, fmin=50, fmax=1000):
    """
    Detects pitch using the autocorrelation method.

    Args:
        frame (np.array): The audio frame as a NumPy array.
        sample_rate (int): The audio sample rate.
        fmin (float): Minimum frequency to consider.
        fmax (float): Maximum frequency to consider.

    Returns:
        float: Detected pitch (in Hz) or 0.0 if no pitch is detected.
    """
    # Remove DC offset by subtracting the mean.
    frame = frame - np.mean(frame)
    # Compute the full autocorrelation of the frame.
    corr = np.correlate(frame, frame, mode="full")
    # Only consider the second half (non-negative lags).
    corr = corr[len(corr)//2:]
    # Define lag range limits based on fmax and fmin.
    min_lag = int(sample_rate / fmax)
    max_lag = int(sample_rate / fmin)
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    # Extract the segment of the autocorrelation corresponding to the desired lags.
    segment = corr[min_lag:max_lag]
    if len(segment) == 0:
        return 0.0
    # Find the lag with the maximum correlation value.
    peak_index = np.argmax(segment)
    lag = peak_index + min_lag
    # Calculate pitch from the lag.
    pitch = sample_rate / lag if lag != 0 else 0.0
    return pitch


# --- New pitch detection algorithms added below ---

def mpm_pitch(frame, sample_rate, fmin=50, fmax=1000, threshold=0.9):
    """
    McLeod Pitch Method (MPM) with cumulative sum precomputation to optimize the per-tau denominator.
    """
    frame = frame.astype(np.float32)
    frame = frame - np.mean(frame)
    N = len(frame)
    max_tau = int(sample_rate / fmin)
    min_tau = int(sample_rate / fmax)
    nsdf = np.zeros(max_tau + 1, dtype=np.float32)

    # Precompute squared values and its cumulative sum for fast energy calculation.
    squared = frame ** 2
    cumsum = np.cumsum(squared)
    total_energy = cumsum[-1]

    for tau in range(min_tau, max_tau + 1):
        numerator = 2 * np.dot(frame[:N - tau], frame[tau:])
        energy1 = cumsum[N - tau - 1] if (N - tau - 1) >= 0 else 0.0
        energy2 = total_energy - (cumsum[tau - 1] if tau - 1 >= 0 else 0.0)
        denominator = energy1 + energy2
        nsdf[tau] = numerator / denominator if denominator > 0 else 0.0

    candidates = np.where(nsdf > threshold)[0]
    if candidates.size == 0:
        return 0.0

    tau_est = candidates[np.argmax(nsdf[candidates])]
    if tau_est > 0 and tau_est < max_tau:
        s0 = nsdf[tau_est - 1]
        s1 = nsdf[tau_est]
        s2 = nsdf[tau_est + 1]
        denom = 2 * (s0 - 2 * s1 + s2)
        if denom != 0:
            tau_est = tau_est - (s2 - s0) / denom
    pitch = sample_rate / tau_est if tau_est != 0 else 0.0
    if pitch < fmin or pitch > fmax:
        return 0.0
    return pitch


def swipe_pitch(frame, sample_rate, fmin=50, fmax=1000):
    """
    Implements a simplified version of the SWIPE algorithm.
    Args:
        frame (np.array): Audio frame.
        sample_rate (int): Sampling rate.
        fmin (float): Minimum pitch considered.
        fmax (float): Maximum pitch considered.
    Returns:
        float: Detected pitch in Hz.
    """
    frame = frame.astype(np.float32)
    N = len(frame)
    window = np.hanning(N)
    xw = frame * window
    nfft = 2 ** int(np.ceil(np.log2(N)))
    X = np.abs(np.fft.rfft(xw, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    candidate_frequencies = np.linspace(fmin, fmax, num=100)
    best_score = -np.inf
    best_pitch = 0.0
    for f in candidate_frequencies:
        score = 0.0
        h = 1
        # Consider up to 10 harmonics (or until the harmonic exceeds Nyquist):
        while h * f < sample_rate / 2 and h < 10:
            target_freq = h * f
            # Optimization: use searchsorted instead of argmin for fast lookup:
            idx = np.searchsorted(freqs, target_freq)
            if idx > 0 and (idx == len(freqs) or abs(freqs[idx] - target_freq) > abs(freqs[idx - 1] - target_freq)):
                idx = idx - 1
            weight = 1.0 / h
            score += weight * X[idx]
            h += 1
        if score > best_score:
            best_score = score
            best_pitch = f
    return best_pitch


def cepstrum_pitch(frame, sample_rate, fmin=50, fmax=1000):
    """
    Implements pitch detection using Cepstrum Analysis.
    Args:
        frame (np.array): Audio frame.
        sample_rate (int): Sampling rate.
        fmin (float): Minimum pitch considered.
        fmax (float): Maximum pitch considered.
    Returns:
        float: Detected pitch in Hz, or 0.0 if not found.
    """
    frame = frame.astype(np.float32)
    N = len(frame)
    window = np.hanning(N)
    xw = frame * window
    nfft = 2 ** int(np.ceil(np.log2(N)))
    spec = np.fft.rfft(xw, n=nfft)
    log_spec = np.log(np.abs(spec) + 1e-10)
    cepstrum = np.fft.irfft(log_spec)
    qmin = int(sample_rate / fmax)
    qmax = int(sample_rate / fmin)
    if qmax >= len(cepstrum):
        qmax = len(cepstrum) - 1
    cep_region = cepstrum[qmin:qmax]
    if len(cep_region) == 0:
        return 0.0
    peak_index = np.argmax(cep_region) + qmin
    pitch = sample_rate / peak_index if peak_index != 0 else 0.0
    if pitch < fmin or pitch > fmax:
        return 0.0
    return pitch


def hps_pitch(frame, sample_rate, fmin=50, fmax=1000, harmonics=4):
    """
    Harmonic Product Spectrum (HPS) pitch detection.
    Uses np.searchsorted to quickly index into the spectrum and multiplies decimated versions.
    """
    frame = frame.astype(np.float32)
    frame = frame - np.mean(frame)
    N = len(frame)
    window = np.hanning(N)
    xw = frame * window
    nfft = 2 ** int(np.ceil(np.log2(N)))
    spectrum = np.abs(np.fft.rfft(xw, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)

    # Find the maximum index corresponding to fmax
    max_index = np.searchsorted(freqs, fmax)
    if max_index == 0 or max_index > len(spectrum):
        max_index = len(spectrum)

    hps_spec = spectrum[:max_index].copy()
    for h in range(2, harmonics + 1):
        decimated = spectrum[::h][:max_index]
        hps_spec[:len(decimated)] *= decimated

    min_index = np.searchsorted(freqs, fmin)
    index = np.argmax(hps_spec[min_index:]) + min_index
    pitch = freqs[index]
    if pitch < fmin or pitch > fmax:
        return 0.0
    return pitch
















# --------------------------- Audio Stream Callback Handler ---------------------------
class AudioStreamCallbackHandler:
    """
    This class handles connecting to the audio stream using a callback model,
    fills a ring buffer with incoming audio samples, and converts all data to float32.
    It reliably lists input devices and chooses the appropriate type.
    """
    def _try_open_stream(self, sample_rate, dtype):
        try:
            device_info = sd.query_devices(self.device_index)
            hostapis = sd.query_hostapis()
            hostapi = hostapis[device_info['hostapi']]

            # Special handling for ASIO devices
            if "ASIO" in hostapi['name'].upper():
                # ASIO requires using the device's native sample rate
                if 'default_samplerate' not in device_info:
                    raise ValueError("ASIO device does not report default sample rate")

                native_rate = float(device_info['default_samplerate'])
                self.debug_callback(f"ASIO device detected - Native sample rate: {native_rate} Hz")

                # Instead of requiring sample_rate == native_rate, override it if necessary.
                if sample_rate != native_rate:
                    self.debug_callback(f"Overriding sample rate from {sample_rate} Hz to native ASIO sample rate {native_rate} Hz")
                    sample_rate = native_rate

                # Attempt to open the stream with explicit low latency
                try:
                    stream = sd.InputStream(
                        device=self.device_index,
                        samplerate=sample_rate,
                        blocksize=None,  # Let ASIO use native buffer size
                        dtype=dtype,
                        channels=1,
                        callback=self._callback,
                        latency='low'  # Use low latency setting for ASIO
                    )
                    stream.start()
                    return stream, sample_rate, dtype
                except Exception as e:
                    self.debug_callback(f"ASIO stream creation with latency 'low' failed: {e}. Trying without explicit latency.")
                    # Fallback: try opening without specifying latency
                    stream = sd.InputStream(
                        device=self.device_index,
                        samplerate=sample_rate,
                        blocksize=None,
                        dtype=dtype,
                        channels=1,
                        callback=self._callback
                    )
                    stream.start()
                    return stream, sample_rate, dtype

            else:
                # Non-ASIO device handling
                suggested_latency = device_info.get('default_low_input_latency', 0.1)
                if "WASAPI" in hostapi['name'].upper():
                    suggested_latency = min(suggested_latency, 0.02)

                stream = sd.InputStream(
                    device=self.device_index,
                    samplerate=sample_rate,
                    blocksize=self.block_size,
                    dtype=dtype,
                    channels=1,
                    callback=self._callback,
                    latency=suggested_latency
                )

                try:
                    stream.start()
                    return stream, sample_rate, dtype
                except Exception as e:
                    if stream:
                        stream.close()
                    raise

        except Exception as e:
            self.debug_callback(f"Stream creation failed: {e}")
            raise

    def __init__(self, device_index, sample_rate, block_size, debug_callback):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.debug_callback = debug_callback
        self.ring_buffer = np.empty((0,), dtype=np.float32)
        self.mutex = QMutex()
        self.fallback_conversion = False
        self.stream = None

        # Attempt to open stream with float32 first.
        self.dtype = "float32"
        try:
            self.stream, self.sample_rate, self.dtype = self._try_open_stream(self.sample_rate, self.dtype)
            # Don't emit basic message here since we'll emit detailed info below
        except Exception as e:
            self.debug_callback(f"Failed to open stream with float32 at sample rate {self.sample_rate}: {e}")
            # Try fallback candidate sample rates with float32.
            candidate_rates = []
            try:
                device_info = sd.query_devices(self.device_index)
                default_sr = device_info.get("default_samplerate", None)
                if default_sr is not None:
                    candidate_rates.append(float(default_sr))
            except Exception as de:
                self.debug_callback(f"Could not retrieve default samplerate: {de}")
            candidate_rates.extend([96000, 48000, 44100, 22050, 16000, 11025, 8000])
            candidate_rates = sorted(set(candidate_rates), reverse=True)
            opened = False
            for sr in candidate_rates:
                if sr == self.sample_rate:
                    continue
                self.debug_callback(f"Trying sample rate {sr} with float32 for device {self.device_index}...")
                try:
                    self.stream, self.sample_rate, self.dtype = self._try_open_stream(sr, "float32")
                    # Don't emit basic message here since we'll emit detailed info below
                    opened = True
                    break
                except Exception as e2:
                    self.debug_callback(f"Fallback test failed for sample rate {sr} with float32: {e2}")
                    time.sleep(0.1)  # wait 100 ms before the next fallback attempt
            if not opened:
                # Fallback: try int16.
                self.dtype = "int16"
                self.fallback_conversion = True
                try:
                    self.stream, self.sample_rate, self.dtype = self._try_open_stream(self.sample_rate, self.dtype)
                    self.debug_callback(f"Stream opened using int16 for device {self.device_index} at sample rate {self.sample_rate}; conversion enabled")
                except Exception as e:
                    self.debug_callback(f"Failed to open stream with int16 at sample rate {self.sample_rate}: {e}")
                    # Try fallback sample rates with int16.
                    candidate_rates = []
                    try:
                        device_info = sd.query_devices(self.device_index)
                        default_sr = device_info.get("default_samplerate", None)
                        if default_sr is not None:
                            candidate_rates.append(float(default_sr))
                    except Exception as de:
                        self.debug_callback(f"Could not retrieve default samplerate: {de}")
                    candidate_rates.extend([96000, 48000, 44100, 22050, 16000, 11025, 8000])
                    candidate_rates = sorted(set(candidate_rates), reverse=True)
                    opened_int16 = False
                    for sr in candidate_rates:
                        if sr == self.sample_rate:
                            continue
                        self.debug_callback(f"Trying sample rate {sr} with int16 for device {self.device_index}...")
                        try:
                            self.stream, self.sample_rate, self.dtype = self._try_open_stream(sr, "int16")
                            self.debug_callback(f"Stream opened using int16 for device {self.device_index} with fallback sample rate {sr}; conversion enabled")
                            opened_int16 = True
                            break
                        except Exception as e2:
                            self.debug_callback(f"Fallback test failed for sample rate {sr} with int16: {e2}")
                            time.sleep(0.1)  # wait 100 ms before the next fallback attempt
                    if not opened_int16:
                        raise Exception(f"Could not open stream with any supported format for device {self.device_index}")

        if defaults.ENABLE_DEBUG:
            try:
                device_info = sd.query_devices(self.device_index)
                hostapis = sd.query_hostapis()
                hostapi_name = hostapis[device_info['hostapi']]['name'] if device_info.get('hostapi') is not None else "Unknown"
                detailed_msg = (
                    f"Detailed Stream Info:\n"
                    f"  Device: {device_info['name']} (index {self.device_index})\n"
                    f"  Sample Rate: {self.sample_rate}\n"
                    f"  Block Size: {self.block_size}\n"
                    f"  Data Type: {self.dtype}\n"
                    f"  Protocol: {hostapi_name}\n"
                    f"  REBUFFER_SIZE: {defaults.REBUFFER_SIZE}"
                )
                self.debug_callback(detailed_msg)
            except Exception as e:
                self.debug_callback(f"Could not retrieve device info: {e}")

    def _callback(self, indata, frames, time, status):
        if status:
            # Only log overflow if debug is enabled to reduce spam
            if defaults.ENABLE_DEBUG and "overflow" in str(status):
                self.debug_callback(f"Stream callback status: {status}")

        # Copy the incoming data.
        if self.fallback_conversion:
            # Convert int16 data to float32 (normalized)
            new_samples = indata.copy().astype(np.float32) / 32768.0
        else:
            new_samples = indata.copy()
        new_samples = np.squeeze(new_samples)

        self.mutex.lock()
        try:
            # If buffer is getting too large, trim it
            if len(self.ring_buffer) + len(new_samples) > defaults.REBUFFER_SIZE:
                # Keep only the most recent samples that will fit
                available_space = defaults.REBUFFER_SIZE - len(new_samples)
                if available_space > 0:
                    self.ring_buffer = self.ring_buffer[-available_space:]
                else:
                    self.ring_buffer = np.empty((0,), dtype=np.float32)

            self.ring_buffer = np.concatenate((self.ring_buffer, new_samples))

            # Ensure buffer never exceeds maximum size
            if len(self.ring_buffer) > defaults.REBUFFER_SIZE:
                self.ring_buffer = self.ring_buffer[-defaults.REBUFFER_SIZE:]

        finally:
            self.mutex.unlock()

    def get_buffer(self):
        """
        Retrieves and clears the ring buffer data.
        """
        self.mutex.lock()
        try:
            data = self.ring_buffer.copy()
            self.ring_buffer = np.empty((0,), dtype=np.float32)
        finally:
            self.mutex.unlock()
        return data

    def stop(self):
        """Safely stop and clean up the audio stream."""
        self.mutex.lock()
        try:
            if self.stream is not None:
                try:
                    self.stream.stop()
                except Exception as e:
                    self.debug_callback(f"Error stopping stream: {e}")
                try:
                    self.stream.close()
                except Exception as e:
                    self.debug_callback(f"Error closing stream: {e}")
                self.stream = None
        finally:
            self.mutex.unlock()


# --------------------------- Audio Processing Thread ---------------------------
class AudioStreamWorker(QThread):
    """
    QThread subclass that handles real-time audio capture and processing.
    It reads audio frames, applies gain and pitch detection, and emits signals for
    pitch updates, state changes, and VU meter updates.
    """
    pitchDetected = Signal(float, float, float)  # Emits effective time, detected pitch, and post-gain RMS.
    stateChanged = Signal(float, bool)             # Emits effective time and pause state changes.
    vuMeterUpdate = Signal(float)                  # Emits pre-gain RMS level.
    vuMeterPostUpdate = Signal(float)              # Emits post-gain RMS level.
    errorOccurred = Signal(str)                    # Emits error messages.
    debugMessage = Signal(str)
    statsText = Signal(str)
    sampleRateChanged = Signal(float)


    # --- Modified __init__: now it accepts a pre-created callback_handler ---
    def __init__(self, callback_handler, parent=None):
        super(AudioStreamWorker, self).__init__(parent)
        # Use the callback_handler already created on the main thread.
        self.callback_handler = callback_handler
        self.device_index = callback_handler.device_index
        self.sample_rate = callback_handler.sample_rate
        self.chunk_size = callback_handler.block_size
        self.current_stream_sample_rate = callback_handler.sample_rate

        # Use a simple debugMessage emitter if none is provided.
        #self.debugMessage = lambda msg: print(msg)

        # Initialization of other variables remains as before.
        self._last_stats_emit = time.time()
        self.running = True
        self.auto_paused = False     # Auto-pause flag.
        self.user_paused = False     # User-initiated pause flag.
        self.paused = False
        self.silent_duration = 0.0
        self.effective_time = 0.0
        self.sensitivity = defaults.DEFAULT_SENSITIVITY
        self.noise_threshold = defaults.MIN_THRESHOLD + (defaults.MAX_THRESHOLD - defaults.MIN_THRESHOLD) * (self.sensitivity / 100.0)
        self.volume_db = defaults.DEFAULT_VOLUME_DB
        self.gain = 10 ** (self.volume_db / 20.0)
        self.boost_mode = False
        self.extra_sensitivity = False
        self.pitch_method = defaults.DEFAULT_PITCH_METHOD
        self.sample_rate = defaults.SAMPLE_RATE
        self.chunk_size = defaults.CHUNK_SIZE
        self.auto_pause_enabled = True

        self.mutex = QMutex()
        self.pause_condition = QWaitCondition()
        self.just_resumed = False
        self.paused_time = 0.0
        self.pause_start_time = 0.0
        self.last_pitch_event = None
        self.outlier_count = 0
        self.priority_set = False

    # --- New method to replace the callback handler when changing devices ---
    def replace_callback_handler(self, new_handler):
        self.mutex.lock()
        try:
            if self.callback_handler is not None:
                self.callback_handler.stop()
            self.callback_handler = new_handler
            self.device_index = new_handler.device_index
            self.sample_rate = new_handler.sample_rate
            self.chunk_size = new_handler.block_size
            self.current_stream_sample_rate = new_handler.sample_rate
        finally:
            self.mutex.unlock()

    def set_boost_mode(self, mode):
        """
        Enable or disable boost mode which increases the effective gain.
        """
        self.boost_mode = mode

    def set_extra_sensitivity(self, mode):
        """
        Enable or disable extra sensitivity for the pitch detection algorithm.
        """
        self.extra_sensitivity = mode

    # --- Modified set_pitch_method: now it updates the processing chunk size ---
    def set_pitch_method(self, method):
        """
        Set the pitch detection method to be used (e.g., "YIN" or "Autocorrelation").
        For the PYIN method, the block size is doubled compared to the default.
        """
        self.pitch_method = method

    def apply_pitch_smoothing(self, new_pitch):
        """
        Helper to smooth sudden drops across all pitch detection methods.
        If the new pitch is less than SMOOTHING_THRESHOLD_RATIO of the previous pitch, count outlier frames.
        For fewer than SMOOTHING_COUNT_THRESHOLD consecutive outlier frames, return the previous (higher) pitch.
        If the condition persists for that many frames, allow the new pitch and reset the counter.
        """
        if not defaults.SMOOTHING_ENABLED:
            return new_pitch

        if self.last_pitch_event is not None and self.last_pitch_event[1] > 0:
            previous_pitch = self.last_pitch_event[1]
            if new_pitch < previous_pitch * defaults.SMOOTHING_THRESHOLD_RATIO:
                self.outlier_count += 1
                if self.outlier_count < defaults.SMOOTHING_COUNT_THRESHOLD:
                    self.debugMessage.emit(
                        f"Outlier pitch detected: previous {previous_pitch:.2f} Hz, current {new_pitch:.2f} Hz. Smoothing applied (count: {self.outlier_count})."
                    )
                    return previous_pitch
                else:
                    self.debugMessage.emit(
                        f"Outlier condition persisted for {self.outlier_count} frames. Updating pitch to new value: {new_pitch:.2f} Hz."
                    )
                    self.outlier_count = 0
                    return new_pitch
            else:
                self.outlier_count = 0
        return new_pitch

    def run(self):
        """
        Main loop for the thread. Reads audio frames and accumulates them in a rebuffer.
        """
        # Set thread priority now that thread is running
        if not self.priority_set:
            try:
                self.setPriority(QThread.TimeCriticalPriority)
                self.priority_set = True
            except Exception as e:
                self.debugMessage.emit(f"Could not set thread priority: {e}")

        # Add overflow counter
        overflow_count = 0
        last_overflow_time = time.time()

        loop_count = 0
        last_fps_time = time.time()
        current_loop_rate = 0.0
        # Persistent rebuffer array (empty initially, float32 normalized values)
        rebuffer = np.empty((0,), dtype=np.float32)
        total_samples_read = 0  # Total samples read from the stream
        processed_samples = 0   # Total samples processed for pitch detection timing
        last_block_time = 0  # Add tracking of last block time
        drawn_time = 0.0


        while self.running:

            iteration_start = time.time()

            # --- Check for manual pause and wait until resumed ---
            self.mutex.lock()
            if self.user_paused:
                self.debugMessage.emit("Audio processing paused. Waiting...")
                self.pause_condition.wait(self.mutex)
                self.debugMessage.emit("Audio processing resumed.")
            self.mutex.unlock()
            # --- End of new pause check code ---

            # Get new samples from the callback handler's ring buffer.
            if self.callback_handler is None:
                self.msleep(5)
                continue
            new_samples = self.callback_handler.get_buffer()

            # Handle potential overflow
            current_time = time.time()
            if len(new_samples) >= defaults.REBUFFER_SIZE:
                overflow_count += 1
                if current_time - last_overflow_time >= 5.0:
                    self.debugMessage.emit(f"Buffer overflow occurred {overflow_count} times in last 5 seconds")
                    overflow_count = 0
                    last_overflow_time = current_time
                # Trim excess samples to prevent cascading overflow
                new_samples = new_samples[-defaults.REBUFFER_SIZE:]

            if len(new_samples) == 0:
                self.msleep(5)
                continue

            # Append the new samples to the rebuffer.
            rebuffer = np.concatenate((rebuffer, new_samples))
            # Trim the ring buffer to REBUFFER_SIZE if necessary
            if len(rebuffer) > defaults.REBUFFER_SIZE:
                rebuffer = rebuffer[-defaults.REBUFFER_SIZE:]

            # Update total_samples_read and effective time (for scrolling)
            total_samples_read += len(new_samples)
            # Use current_stream_sample_rate for all time computations:
            self.effective_time = total_samples_read / self.current_stream_sample_rate

#            self.statsText.emit(f"""Ch:{self.chunk_size} Bu:{len(new_samples)}<br>
#                                    RB:{len(rebuffer)}/{defaults.REBUFFER_SIZE},<br>
#                                    Eff. Time: {self.effective_time:.01f} s,<br>
#                                    Loop Rate: {current_loop_rate:.1f} Hz""")

            # Process full blocks from the rebuffer
            while len(rebuffer) >= self.chunk_size:

                # Extract one block of CHUNK_SIZE samples.
                current_frame = rebuffer[:self.chunk_size]
                rebuffer = rebuffer[self.chunk_size:]

                # Update processed samples and compute block time.
                processed_samples += self.chunk_size
                block_time = processed_samples / self.current_stream_sample_rate

                # Ensure monotonic time progression
                if block_time <= last_block_time:
                    block_time = last_block_time + (self.chunk_size / self.current_stream_sample_rate)
                last_block_time = block_time

                # Compute pre-gain RMS for VU meter update.
                pre_gain_rms = np.sqrt(np.mean(current_frame ** 2))
                self.vuMeterUpdate.emit(pre_gain_rms)

                # Compute effective gain.
                effective_gain = self.gain
                #effective_gain = self.gain * (4 if self.boost_mode else 1)
                #if self.callback_handler.dtype == 'float32' and not getattr(self, "is_wasapi", False):
                if self.callback_handler.dtype == 'float32':
                    extra_multiplier = 1   # For example, try lowering the multiplier.
                    effective_gain *= extra_multiplier

                # Apply gain and calculate post-gain RMS.
                audio_frame = current_frame * effective_gain
                post_gain_rms = np.sqrt(np.mean(audio_frame ** 2))
                self.vuMeterPostUpdate.emit(post_gain_rms)

                # Adjust noise threshold using the value calculated from sensitivity.
                adjusted_threshold = self.noise_threshold

                # Determine pitch.
                if post_gain_rms < adjusted_threshold:
                    pitch = 0.0
                else:
                    if self.pitch_method == "YIN":
                        th = defaults.YIN_DEFAULT_THRESHOLD
                        if self.extra_sensitivity:
                            th = defaults.YIN_SENSITIVE_THRESHOLD
                        pitch = yin_pitch(audio_frame, self.current_stream_sample_rate, threshold=th)
                    elif self.pitch_method == "Autocorrelation":
                        pitch = autocorrelation_pitch(audio_frame, self.current_stream_sample_rate, fmin=defaults._FMIN, fmax=defaults._FMAX)
                    elif self.pitch_method == "MPM":
                        pitch = mpm_pitch(audio_frame, self.current_stream_sample_rate, fmin=defaults._FMIN, fmax=defaults._FMAX)
                    elif self.pitch_method == "SWIPE":
                        pitch = swipe_pitch(audio_frame, self.current_stream_sample_rate, fmin=defaults._FMIN, fmax=defaults._FMAX)
                    elif self.pitch_method == "Cepstrum":
                        pitch = cepstrum_pitch(audio_frame, self.current_stream_sample_rate, fmin=defaults._FMIN, fmax=defaults._FMAX)
                    elif self.pitch_method == "PYIN":
                        pitch = pyin_pitch(audio_frame, self.current_stream_sample_rate,
                                          fmin=defaults._FMIN, fmax=defaults._FMAX,
                                          debug_callback=self.debugMessage.emit)
                    elif self.pitch_method == "HPS":
                        pitch = hps_pitch(audio_frame, self.current_stream_sample_rate, fmin=defaults._FMIN, fmax=defaults._FMAX)
                    else:
                        pitch = 0.0

                dt_block = self.chunk_size / self.current_stream_sample_rate

                # Update silent duration and manage auto pause/resume using the block time.
                if not self.user_paused:
                    if pitch == 0.0:
                        self.silent_duration += dt_block
                    else:
                        self.silent_duration = 0.0



                    if self.auto_pause_enabled:

                        # initiate auto-pause
                        if self.silent_duration >= defaults.SILENT_DURATION and not self.auto_paused:

                            self.paused = True
                            self.auto_paused = True

                            # Use the timestamp of the last detected pitch, if available, as the auto-pause time.
                            if self.last_pitch_event is not None and self.last_pitch_event[1] > 0:
                                #self.pause_start_time = self.last_pitch_event[0] + self.paused_time
                                self.pause_start_time = self.last_pitch_event[0]
                            else:
                                #self.pause_start_time = self.effective_time - defaults.SILENT_DURATION
                                self.pause_start_time = self.effective_time - self.paused_time - self.silent_duration

                            self.stateChanged.emit(self.pause_start_time, True)

                            # Instead of emitting a break event (-1.0), re-emit the last valid pitch so that
                            # the highlighted horizontal line remains visible on pause.
                            #if self.last_pitch_event is not None and self.last_pitch_event[1] > 0:
                            #    self.pitchDetected.emit(self.pause_start_time - self.paused_time,
                            #                            self.last_pitch_event[1],
                            #                            self.last_pitch_event[2])
                            #else:
                            #    self.pitchDetected.emit(self.pause_start_time - self.paused_time, -1.0, post_gain_rms)


                        # pitch returns
                        if self.auto_paused and pitch != 0.0:

                            self.paused = False
                            self.auto_paused = False
                            #self.paused_time = self.effective_time - self.pause_start_time
                            self.just_resumed = True
                            #drawn_time = self.effective_time - self.paused_time - self.silent_duration
                            self.stateChanged.emit(drawn_time, False)


                        if self.auto_paused:
                            self.paused_time = self.effective_time - self.pause_start_time
                        else:
                            drawn_time = self.effective_time - self.paused_time
                    else:
                        # When auto-pause is disabled, always update drawn_time and ensure not paused
                        self.paused = False
                        self.auto_paused = False
                        drawn_time = self.effective_time - self.paused_time


                # Only send pitch detection if neither user pause nor auto pause is active.
                if not self.paused and not self.auto_paused:
                    resume_time = self.effective_time - self.paused_time + defaults.SILENT_DURATION
                    if self.just_resumed:
                        self.pitchDetected.emit(resume_time, -1.0, 0)
                        self.just_resumed = False
                    else:
                        if pitch > 0:
                            # Apply smoothing for all pitch detection methods
                            pitch = self.apply_pitch_smoothing(pitch)
                            self.last_pitch_event = (resume_time, pitch, post_gain_rms)
                            self.pitchDetected.emit(resume_time, pitch, post_gain_rms)
                        else:
                            self.pitchDetected.emit(resume_time, -1.0, 0)

                #else:
                #    self.msleep(10)



            iteration_duration = time.time() - last_fps_time
            desired_period = 1.0 / defaults.MAX_AUDIO_WORKER_LOOP_HZ
            sleep_time = max(0.001, desired_period - iteration_duration)
            #sleep_time = desired_period - iteration_duration
            #if sleep_time <= 0.001: sleep_time = 0.001
            time.sleep(sleep_time)

            # start new iteration and measure frequency
            loop_count += 1
            if iteration_start - last_fps_time >= 1.0:
                current_loop_rate = loop_count / (iteration_start - last_fps_time)
                loop_count = 0
                last_fps_time = iteration_start


            if defaults.ENABLE_DEBUG:
                current_time = time.time()
                if current_time - self._last_stats_emit >= 0.5:
                    self.statsText.emit(f"""
                        Ch: {self.chunk_size} Bu: {len(new_samples)}<br>
                        RB: {len(rebuffer)}/{defaults.REBUFFER_SIZE}<br>
                        E/D: {self.effective_time:.01f}/{drawn_time:.01f}<br>
                        WorkerRate: {current_loop_rate:.1f}Hz
                        """)
                    self._last_stats_emit = current_time


        if self.callback_handler is not None:
            self.callback_handler.stop()



    def pause(self):
        """
        Pause the processing, marking it as a user-initiated pause.
        """
        self.mutex.lock()
        self.user_paused = True
        self.paused = True
        # If a valid pitch event exists, use its time to freeze the Piano Roll;
        # otherwise use the current effective time.
        if self.last_pitch_event is not None and self.last_pitch_event[1] > 0:
            self.pause_start_time = self.last_pitch_event[0] + self.paused_time
        else:
            self.pause_start_time = self.effective_time
        # Override any auto-pause state when manually pausing.
        self.auto_paused = False
        # --- Flush the ring buffer to ignore accumulated samples during pause ---
        if self.callback_handler is not None:
            self.callback_handler.mutex.lock()
            self.callback_handler.ring_buffer = np.empty((0,), dtype=np.float32)
            self.callback_handler.mutex.unlock()
        # --- End flush code ---
        self.mutex.unlock()
        self.stateChanged.emit(self.pause_start_time - self.paused_time, True)
        # Re-emit the last valid pitch event using the frozen time so that the Piano Roll remains in place.
        if self.last_pitch_event is not None and self.last_pitch_event[1] > 0:
            current_time = self.pause_start_time - self.paused_time
            self.pitchDetected.emit(current_time, self.last_pitch_event[1], self.last_pitch_event[2])

    def resume(self):
        """
        Resume processing after a pause.
        """
        self.mutex.lock()
        self.user_paused = False
        self.paused = False
        # --- Added code to flush the ring buffer to remove audio captured during the pause ---
        if self.callback_handler is not None:
            self.callback_handler.mutex.lock()
            self.callback_handler.ring_buffer = np.empty((0,), dtype=np.float32)
            self.callback_handler.mutex.unlock()
        # --- End added code ---
        self.paused_time += (self.effective_time - self.pause_start_time)  # Add pause duration to total
        self.silent_duration = 0.0
        self.pause_condition.wakeAll()
        self.mutex.unlock()
        self.stateChanged.emit(self.effective_time - self.paused_time, False)

    def set_sensitivity(self, value):
        """
        Update the sensitivity setting and recalculate the noise threshold.

        Args:
            value (int): New sensitivity value (0-100).
        """
        self.sensitivity = value
        # Recalculate noise threshold using defaults.
        self.noise_threshold = defaults.MIN_THRESHOLD + (defaults.MAX_THRESHOLD - defaults.MIN_THRESHOLD) * (value / 100.0)
        self.debugMessage.emit(f"Sensitivity set to: {value}. Updated noise threshold: {self.noise_threshold}")

    def set_volume_db(self, value):
        """
        Update the volume preamplification in decibels and recalculate gain.

        Args:
            value (int): Volume level in decibels.
        """
        self.volume_db = value
        self.gain = 10 ** (value / 20.0)

    def set_input_device(self, device_index):
        """
        Change the audio input device in a thread-safe manner.

        Args:
            device_index (int): The index of the new input device.
        """

        self.mutex.lock()
        try:
            if self.callback_handler is not None:
                self.callback_handler.stop()
                # Increase delay to allow the ASIO device to be released completely
                time.sleep(1.0)  # wait 1 second
            self.device_index = device_index
            # For ASIO devices, use the device's default sample rate instead of defaults.SAMPLE_RATE
            try:
                device_info = sd.query_devices(self.device_index)
                hostapis = sd.query_hostapis()
                hostapi_name = hostapis[device_info['hostapi']]['name']
                if "ASIO" in hostapi_name.upper():
                    self.sample_rate = float(device_info.get("default_samplerate", defaults.SAMPLE_RATE))
                else:
                    self.sample_rate = defaults.SAMPLE_RATE
            except Exception as e:
                self.debugMessage.emit(f"Error retrieving device default sample rate: {e}")
                self.sample_rate = defaults.SAMPLE_RATE
        except Exception as e:
            self.debugMessage.emit(f"Error retrieving device default sample rate: {e}")
            self.sample_rate = defaults.SAMPLE_RATE

        # NEW: When reconnecting use the preferred sample rate from defaults
        self.callback_handler = AudioStreamCallbackHandler(
            self.device_index,
            self.sample_rate,
            self.chunk_size,
            self.debugMessage.emit
        )
        # Update current_stream_sample_rate from the new stream
        self.current_stream_sample_rate = self.callback_handler.sample_rate
        if self.sample_rate != self.callback_handler.sample_rate:
            self.sample_rate = self.callback_handler.sample_rate
            self.sampleRateChanged.emit(self.sample_rate)
        # Print detailed stream info only once, here
        if defaults.ENABLE_DEBUG:
            try:
                device_info = sd.query_devices(self.device_index)
                hostapis = sd.query_hostapis()
                hostapi_name = hostapis[device_info['hostapi']]['name'] if device_info.get('hostapi') is not None else "Unknown"
                msg = (
                    f"Stream Info:\n"
                    f"  Device: {device_info['name']} (index {self.device_index})\n"
                    f"  Sample Rate: {self.sample_rate}\n"
                    f"  Chunk Size: {self.chunk_size}\n"
                    f"  Data Type: {self.callback_handler.dtype}\n"
                    f"  Protocol: {hostapi_name}\n"
                    f"  REBUFFER_SIZE: {defaults.REBUFFER_SIZE}"
                )
                self.debugMessage.emit(msg)
            except Exception as e:
                self.debugMessage.emit(f"Failed to retrieve detailed device info: {e}")
        try:
            self.callback_handler = AudioStreamCallbackHandler(
                self.device_index,
                self.sample_rate,
                self.chunk_size,
                self.debugMessage.emit
            )
            # Update sample rate if it changed during device initialization
            if self.sample_rate != self.callback_handler.sample_rate:
                self.sample_rate = self.callback_handler.sample_rate
                self.sampleRateChanged.emit(self.sample_rate)
            # Print detailed stream info only once, here
            if defaults.ENABLE_DEBUG:
                try:
                    device_info = sd.query_devices(self.device_index)
                    hostapis = sd.query_hostapis()
                    hostapi_name = hostapis[device_info['hostapi']]['name'] if device_info.get('hostapi') is not None else "Unknown"
                    msg = (
                        f"Stream Info:\n"
                        f"  Device: {device_info['name']} (index {self.device_index})\n"
                        f"  Sample Rate: {self.sample_rate}\n"
                        f"  Chunk Size: {self.chunk_size}\n"
                        f"  Data Type: {self.callback_handler.dtype}\n"
                        f"  Protocol: {hostapi_name}\n"
                        f"  REBUFFER_SIZE: {defaults.REBUFFER_SIZE}"
                    )
                    self.debugMessage.emit(msg)
                except Exception as e:
                    self.debugMessage.emit(f"Failed to retrieve detailed device info: {e}")
        except Exception as e:
            error_msg = f"Error setting input device: {e}"
            self.errorOccurred.emit(error_msg)
            self.debugMessage.emit(error_msg)
            self.callback_handler = None
        except Exception as e:
            error_msg = f"Error setting input device: {e}"
            self.errorOccurred.emit(error_msg)
            self.debugMessage.emit(error_msg)
        finally:
            self.mutex.unlock()

    def stop(self):
        self.running = False
        if self.callback_handler:
            self.callback_handler.stop()
        self.mutex.lock()
        self.pause_condition.wakeAll()
        self.mutex.unlock()
        self.wait()


# --------------------------- Piano Roll Widget ---------------------------
class PianoRollWidget(QOpenGLWidget):
    """
    QOpenGLWidget subclass that implements a hardware-accelerated scrolling piano roll display.
    It shows grid lines for each MIDI note and plots the detected pitch over time.
    """
    def __init__(self, parent=None):
        super(PianoRollWidget, self).__init__(parent)
        self.setMouseTracking(True)  # Enable hover events for tooltips
        self.QToolTip = QToolTip  # Use globally imported QToolTip
        self.setMinimumHeight(300)

        # Set OpenGL format for better performance and VSync control
        format = self.format()
        format.setSamples(defaults.ANTIALIASING_SAMPLES)  # Enable antialiasing
        if defaults.ENABLE_VSYNC:
            format.setSwapInterval(1)  # Enable V-Sync if requested
        else:
            format.setSwapInterval(0)  # Disable V-Sync
        self.setFormat(format)

        # Existing initialization code remains the same
        self.padding = defaults.DEFAULT_PADDING
        self.vert_padding = defaults.DEFAULT_VERT_PADDING  # Use vertical padding from defaults
        self.right_margin = defaults.DEFAULT_RIGHT_MARGIN
        self.points = []
        self.markers = []
        self.latest_time = 0.0
        self.scroll_speed = defaults.DEFAULT_SCROLL_SPEED
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        update_interval = int(1000 / defaults.DEFAULT_PIANO_ROLL_UPDATE_FPS)
        self.timer.start(update_interval)
        self.cents = 0
        self.vu_level_pre = 0.0
        self.vu_level_post = 0.0
        self.pr_last_time = time.time()
        self.pr_update_count = 0
        self.pr_fps = 0.0

    def add_pitch_point(self, time_stamp, pitch, label, cents):
        """
        Add a new pitch point to the list, with time-based filtering to reduce jagged lines.
        """
        # Minimum time difference between points (in seconds)
        MIN_TIME_DELTA = 0.01  # 10ms

        if self.points:
            last_point = self.points[-1]
            # Only add point if enough time has passed or it's a different pitch
            if (time_stamp - last_point['time'] >= MIN_TIME_DELTA or
                (pitch != last_point.get('pitch') and pitch is not None)):
                self.points.append({
                    'time': time_stamp,
                    'pitch': pitch,
                    'label': label,
                    'cents': cents
                })
        else:
            # Always add the first point
            self.points.append({
                'time': time_stamp,
                'pitch': pitch,
                'label': label,
                'cents': cents
            })

        self.cents = cents
        if time_stamp > self.latest_time:
            self.latest_time = time_stamp
        if defaults.ENABLE_FORCED_PIANOROLL_UPDATE:
            self.update()

    def add_break_point(self, time_stamp):
        """
        Insert a break in the drawn pitch line (when pitch detection pauses).
        """
        self.points.append({'time': time_stamp, 'pitch': None})
        if time_stamp > self.latest_time:
            self.latest_time = time_stamp
        if defaults.ENABLE_FORCED_PIANOROLL_UPDATE: self.update()

    def add_marker(self, time_stamp, paused):
        """
        Add a marker indicating a pause or resume event.
        """
        if paused:
            self.markers.append({'time': time_stamp, 'paused': paused})

        if time_stamp > self.latest_time:
            self.latest_time = time_stamp

        if defaults.ENABLE_FORCED_PIANOROLL_UPDATE: self.update()

    def set_scroll_speed(self, speed):
        """
        Update the scrolling speed of the piano roll.

        Args:
            speed (int): New scroll speed in pixels per second.
        """
        self.scroll_speed = speed

    def update_vu_meter(self, value):
        """
        Update the pre-gain VU meter value.

        Args:
            value (float): The new pre-gain RMS value.
        """
        self.vu_level_pre = value
        if defaults.ENABLE_FORCED_PIANOROLL_UPDATE: self.update()

    def update_vu_meter_post(self, value):
        """
        Update the post-gain VU meter value.

        Args:
            value (float): The new post-gain RMS value.
        """
        self.vu_level_post = value
        if defaults.ENABLE_FORCED_PIANOROLL_UPDATE: self.update()

    def paintGL(self):
        """
        OpenGL-accelerated painting of the piano roll.
        This replaces the old paintEvent method.
        """
        current_time = time.time()
        self.pr_update_count += 1
        if current_time - self.pr_last_time >= 1.0:
            self.pr_fps = self.pr_update_count / (current_time - self.pr_last_time)
            self.pr_update_count = 0
            self.pr_last_time = current_time

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Get widget dimensions
        rect = self.rect()
        painter.fillRect(rect, QColor(30, 30, 30))
        width = rect.width() - self.right_margin
        height = rect.height()

        # Filter out old points and markers based on MAX_DATA_AGE
        self.points = [pt for pt in self.points if pt['time'] >= self.latest_time - defaults.MAX_DATA_AGE]
        self.markers = [mk for mk in self.markers if mk['time'] >= self.latest_time - defaults.MAX_DATA_AGE]

        use_scale_highlighting = (defaults.current_scale != "-")
        prefer_sharps_local = use_sharps_for_key()

        detected_midi = None
        if self.points and self.points[-1].get('pitch') is not None and self.points[-1]['pitch'] > 0:
            detected_midi = int(round(69 + 12 * math.log2(self.points[-1]['pitch'] / defaults.TUNING_FREQUENCY)))

        min_midi = defaults.MIDI_START
        max_midi = defaults.MIDI_END
        drawn_labels = set()
        
        # Get the current scale with proper enharmonic handling
        current_scale_notes = []
        is_chromatic = False
        if use_scale_highlighting and defaults.current_scale != "-":
            is_chromatic = (defaults.current_scale == "Chromatic")
            current_scale_notes = defaults.generate_scale()
            # Get canonical versions for comparison
            current_scale_canonical = [canonical_note(note) for note in current_scale_notes]
        
        for midi in range(min_midi, max_midi + 1):
            # Use vertical padding for the y-coordinate calculations
            y = self.vert_padding + (max_midi - midi) * (height - 2 * self.vert_padding) / (max_midi - min_midi)
            
            # Determine if this MIDI note is in the current scale
            is_in_scale = False
            is_root_note = False
            if use_scale_highlighting:
                # Get the note name from MIDI number
                note_index = midi % 12
                note_name = SHARP_LETTER_NAMES[note_index] if prefer_sharps_local else FLAT_LETTER_NAMES[note_index]
                
                # Get canonical form for comparison
                canonical_midi_note = canonical_note(note_name)
                
                # Check if this is the root note
                root_canonical = canonical_note(defaults.current_root_note.split("/")[0] if "/" in defaults.current_root_note else defaults.current_root_note)
                is_root_note = (canonical_midi_note == root_canonical)
                
                # Check if this note is in the scale
                is_in_scale = canonical_midi_note in current_scale_canonical
                
                # Style the grid line based on whether it's in the scale
                if is_root_note:
                    pen = QPen(defaults.COLOR_ROOT)
                    pen.setWidth(defaults.GRID_LINE_WIDTH_ROOT)
                    pen.setStyle(Qt.SolidLine)
                elif is_in_scale:
                    pen = QPen(defaults.COLOR_DIA)
                    pen.setWidth(defaults.GRID_LINE_WIDTH)
                    pen.setStyle(Qt.CustomDashLine)
                    pen.setDashPattern(defaults.DASH_PATTERN)
                else:
                    pen = QPen(defaults.COLOR_NONDIA)
                    pen.setWidth(defaults.GRID_LINE_WIDTH)
                    pen.setStyle(Qt.CustomDashLine)
                    pen.setDashPattern(defaults.DASH_PATTERN)
            else:
                # Default styling when no scale is selected
                #if midi % 12 == 0:
                #    pen = QPen(defaults.COLOR_ROOT)
                #    pen.setWidth(defaults.GRID_LINE_WIDTH_ROOT)
                #    pen.setStyle(Qt.SolidLine)
                #else:
                    pen = QPen(defaults.COLOR_NONDIA)
                    pen.setWidth(defaults.GRID_LINE_WIDTH)
                    pen.setStyle(Qt.CustomDashLine)
                    pen.setDashPattern(defaults.DASH_PATTERN)

            painter.setPen(pen)
            painter.drawLine(int(self.padding), int(y), int(width), int(y))

            if (detected_midi is not None and midi == detected_midi) and defaults.DRAW_DETECTED_LINE == True:
                # Calculate alpha based on cents deviation (0-255)
                alpha = max(0, min(255, 255 - (abs(self.cents) * 4)))
                detected_color = QColor(defaults.COLOR_DETECTED)
                detected_color.setAlpha(alpha)
                pen = QPen(detected_color)
                pen.setWidth(1)
                pen.setStyle(Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(int(self.padding), int(y), int(width), int(y))

            # Draw labels for scale notes
            draw_label = (use_scale_highlighting and is_in_scale) or (not use_scale_highlighting)
            if draw_label:
                note_index = midi % 12
                octave_val = (midi // 12) - 1
                
                # If using scale highlighting, try to find the actual scale note (with correct enharmonic)
                if use_scale_highlighting and is_in_scale:
                    # Get the canonical form of this MIDI note
                    midi_note_canonical = canonical_note(SHARP_LETTER_NAMES[note_index])
                    
                    # For chromatic scale, use consistent accidentals based on user preference 
                    if is_chromatic:
                        if prefer_sharps_local:
                            letter = SHARP_LETTER_NAMES[note_index]
                            solfege = SHARP_SOLFEGE_NAMES[note_index]
                        else:
                            letter = FLAT_LETTER_NAMES[note_index]
                            solfege = FLAT_SOLFEGE_NAMES[note_index]
                    else:
                        # For non-chromatic scales, find the matching note in the scale
                        matching_note = None
                        for i, scale_canonical in enumerate(current_scale_canonical):
                            if scale_canonical == midi_note_canonical:
                                matching_note = current_scale_notes[i]
                                break
                        
                        # Use the matching note from the scale if found
                        if matching_note:
                            letter = matching_note
                            solfege = note_to_solfege(matching_note)
                        else:
                            # Fallback to standard mapping
                            if prefer_sharps_local:
                                letter = SHARP_LETTER_NAMES[note_index]
                                solfege = SHARP_SOLFEGE_NAMES[note_index]
                            else:
                                letter = FLAT_LETTER_NAMES[note_index]
                                solfege = FLAT_SOLFEGE_NAMES[note_index]
                else:
                    # Standard mapping when not using scale highlighting
                    if prefer_sharps_local:
                        letter = SHARP_LETTER_NAMES[note_index]
                        solfege = SHARP_SOLFEGE_NAMES[note_index]
                    else:
                        letter = FLAT_LETTER_NAMES[note_index]
                        solfege = FLAT_SOLFEGE_NAMES[note_index]

                canon_letter = canonical_note(letter)
                label_pair = (canon_letter, octave_val)
                if label_pair not in drawn_labels:
                    main_font = QFont()
                    main_font.setPixelSize(defaults.GRID_FONT_MAIN_SIZE)
                    sub_font = QFont()
                    sub_font.setPixelSize(defaults.GRID_FONT_SUB_SIZE)
                    #text_color = defaults.COLOR_ROOT if (is_root_note or midi % 12 == 0) else defaults.COLOR_DIA
                    text_color = defaults.COLOR_ROOT if (is_root_note) else defaults.COLOR_DIA
                    painter.setPen(QPen(text_color))
                    painter.setFont(main_font)
                    main_text = ""
                    if defaults.PITCH_LABEL_SOLFEGE:
                        main_text += f"{solfege}"
                    if defaults.PITCH_LABEL_LETTERS:
                        main_text += f" {letter}"

                    horizontalAdvance = drawAdjustedText(painter, int(width) + 10, int(y) + 4, main_text, flat_char="♭", adjustment=-3)
                    if defaults.PITCH_LABEL_OCTAVE:
                        painter.setFont(sub_font)
                        painter.drawText(int(width) + horizontalAdvance + 10, int(y) + 7, str(octave_val))
                    drawn_labels.add(label_pair)

        # Draw time ticks for each second (using adjusted time) and dump ticks older than MAX_DATA_AGE
        visible_duration = width / self.scroll_speed
        display_time = self.latest_time  # Already adjusted for paused time
        start_time = display_time - visible_duration
        # Ensure we do not draw any time ticks older than MAX_DATA_AGE
        max_age_time = self.latest_time - defaults.MAX_DATA_AGE
        if start_time < max_age_time:
            start_time = max_age_time
        t = math.ceil(start_time)
        while t <= display_time:
            x = width - (display_time - t) * self.scroll_speed
            if t >= 0:
                pen = QPen(defaults.COLOR_TIME_LINES)
                pen.setWidth(1)
                pen.setStyle(Qt.CustomDashLine)
                pen.setDashPattern(defaults.DASH_PATTERN_VERTICAL)
                painter.setPen(pen)

                painter.drawLine(int(x), int(height - self.vert_padding), int(x), int(self.vert_padding))

                painter.setPen(QPen(QColor(220, 220, 220)))
                painter.drawLine(int(x), int(height - self.vert_padding), int(x), int(height - self.vert_padding + 5))
                font = QFont()
                font.setPixelSize(8)
                painter.setFont(font)
                painter.drawText(int(x) - 10, int(height - self.vert_padding + 15), str(t))
            t += 1

        # ------------------- Draw pitch line (optimized) -------------------
        if len(self.points) > 1:
            # Only process points that are visible (i.e. within the drawn time window)
            visible_start_time = display_time - (width / self.scroll_speed)
            visible_points = [pt for pt in self.points if pt['time'] >= visible_start_time]

            # Optional: Downsample if there are many points
            #max_points = 1000
            #if len(visible_points) > max_points:
            #    step = len(visible_points) // max_points
            #    visible_points = visible_points[::step]

            pen = QPen(defaults.PITCH_LINE_DRAW_COLOR)
            pen.setWidth(defaults.PITCH_LINE_WIDTH)  # Make line thicker
            pen.setCapStyle(Qt.RoundCap)             # Round the line ends
            pen.setJoinStyle(Qt.RoundJoin)           # Round the line joins
            painter.setPen(pen)
            path = QPainterPath()
            first_point = True

            for point in visible_points:
                # Check for breaks indicated by pitch <= 0 or a missing pitch
                if point.get('pitch') is None or point.get('pitch') <= 0:
                    # When a break occurs, draw the current segment if any
                    if not first_point:
                        painter.drawPath(path)
                        path = QPainterPath()
                        first_point = True
                    continue

                # Convert pitch to the corresponding MIDI value and compute (x, y)
                midi_val = 69 + 12 * math.log2(point['pitch'] / defaults.TUNING_FREQUENCY)
                midi_val = max(defaults.MIDI_START, min(defaults.MIDI_END, midi_val))
                # Use vertical padding for y calculation
                y = self.vert_padding + (defaults.MIDI_END - midi_val) * (height - 2 * self.vert_padding) / (defaults.MIDI_END - defaults.MIDI_START)
                x = width - (display_time - point['time']) * self.scroll_speed

                # If alpha adjustment is enabled, update pen color before adding the point
                if defaults.PITCH_LINE_ALPHA_ADJUSTMENT:
                    cents = point.get('cents', 0)
                    alpha = max(0, min(255, 255 - (abs(cents) * 5)))
                    line_color = QColor(defaults.PITCH_LINE_DRAW_COLOR)
                    line_color.setAlpha(alpha)
                    pen.setColor(line_color)
                    painter.setPen(pen)

                # Accumulate a continuous path between valid points
                if first_point:
                    path.moveTo(x, y)
                    first_point = False
                else:
                    path.lineTo(x, y)

            # Draw any remaining path segment
            if not first_point and path.elementCount() > 0:
                painter.drawPath(path)
        # ------------------- End pitch line drawing -------------------

        # Draw the last pitch label if available using the latest_time for x coordinate.
        if defaults.PITCH_LABEL_ENABLED:
            if self.points:
                last_point = self.points[-1]
                if last_point.get('pitch') is None:
                    if len(self.points) > 1:
                        last_point = self.points[-2]
                if last_point.get('pitch') is not None:
                    midi_val = 69 + 12 * math.log2(last_point['pitch'] / defaults.TUNING_FREQUENCY)
                    midi_val = max(defaults.MIDI_START, min(defaults.MIDI_END, midi_val))
                    y = self.vert_padding + (defaults.MIDI_END - midi_val) * (height - 2 * self.vert_padding) / (defaults.MIDI_END - defaults.MIDI_START)
                    x = width - (display_time - last_point['time']) * self.scroll_speed
                    painter.setPen(defaults.PITCH_LABEL_COLOR)
                    font = QFont()
                    font.setPixelSize(defaults.GRID_FONT_LABEL_SIZE)
                    painter.setFont(font)
                    _ = drawAdjustedText(painter, int(x) + 5, int(y) - 5, last_point.get('label', ""), flat_char="♭", adjustment=-3)

        # Draw pause/resume markers with filtering to avoid duplicates or very close markers.
        last_marker_x = None
        marker_threshold = 5  # Minimum pixel distance between markers
        for marker in self.markers:
            x = width - (display_time - marker['time']) * self.scroll_speed
            if last_marker_x is not None and abs(x - last_marker_x) < marker_threshold:
                continue
            pen = QPen(defaults.COLOR_PAUSED, 1)
            painter.setPen(pen)
            painter.drawLine(int(x), int(self.vert_padding), int(x), int(height - self.vert_padding))
            last_marker_x = x

        meter_width = 100
        meter_height = 10
        gap = 5
        margin = 20
        meter_post_y = margin + 20
        meter_pre_y = meter_post_y - meter_height - gap

        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawRect(margin, meter_pre_y, meter_width, meter_height)
        fill_fraction_pre = min(self.vu_level_pre / 0.1, 1.0)
        fill_width_pre = int((meter_width-1) * fill_fraction_pre)
        painter.fillRect(margin + 1, meter_pre_y + 1, fill_width_pre, meter_height - 1, defaults.VUMETER_FILL_COLOR)
        font = QFont()
        font.setPixelSize(10)
        painter.setFont(font)
        painter.drawText(margin + meter_width + 5, meter_pre_y + meter_height - 1, "Pre-Gain")

        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawRect(margin, meter_post_y, meter_width, meter_height)
        fill_fraction_post = min(self.vu_level_post / 0.1, 1.0)
        fill_width_post = int((meter_width-1) * fill_fraction_post)
        painter.fillRect(margin + 1, meter_post_y + 1, fill_width_post, meter_height - 1, defaults.VUMETER_FILL_COLOR)
        painter.drawText(margin + meter_width + 5, meter_post_y + meter_height - 1, "Post-Gain")

        painter.end()

    def resizeGL(self, w, h):
        """
        Handle OpenGL viewport resize.
        """
        pass  # The default implementation is sufficient

    def initializeGL(self):
        """
        Initialize OpenGL context.
        """
        pass  # The default implementation is sufficient

    def mouseMoveEvent(self, event):
        """
        Overridden mouse move event that shows a tooltip displaying the note and frequency
        if the mouse is in the static label region (the right margin).
        """
        rect = self.rect()
        # The drawing area for grid lines ends at (rect.width() - self.right_margin)
        region_start = rect.width() - self.right_margin
        if event.position().x() >= region_start:
            height = rect.height()
            y = event.position().y()
            # Clamp y into [self.vert_padding, height - self.vert_padding]
            if y < self.vert_padding:
                y = self.vert_padding
            elif y > height - self.vert_padding:
                y = height - self.vert_padding

            # Calculate MIDI value and frequency...
            midi_value = defaults.MIDI_END - ((y - self.vert_padding) / (height - 2 * self.vert_padding)) * (defaults.MIDI_END - defaults.MIDI_START)
            midi_int = int(round(midi_value))
            midi_int = max(defaults.MIDI_START, min(defaults.MIDI_END, midi_int))

            frequency = defaults.TUNING_FREQUENCY * (2 ** ((midi_int - 69) / 12))
            full_note, _, _, _ = frequency_to_note(frequency, defaults.global_prefer_sharps)
            tool_tip_text = f"{full_note}\n{frequency:.2f} Hz"
            self.QToolTip.showText(event.globalPosition().toPoint(), tool_tip_text, self)
        else:
            self.QToolTip.hideText()
            event.ignore()

    def leaveEvent(self, event):
        """
        Hide the tooltip when the mouse leaves the widget.
        """
        self.QToolTip.hideText()
        event.accept()

    def get_pr_fps(self):
        return self.pr_fps


# --------------------------- Pitch Information Panel (Staff Display) ---------------------------
class PitchInfoPanel(QWidget):
    """
    Widget displaying pitch information along with a simplified musical staff.
    Shows the detected note, its frequency, and cent deviation.
    """
    volumeChanged = Signal(int)
    sensitivityChanged = Signal(int)
    def __init__(self, parent=None):
        super(PitchInfoPanel, self).__init__(parent)
        self.setFixedWidth(140)
        self.full_note = ""
        self.frequency = 0.0
        self.cents = 0
        self.initUI()

    def initUI(self):
        """
        Initialize the Pitch Information Panel UI elements.
        """
        # UI elements
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setMaximumWidth(120)
        self.autopause_checkbox = QCheckBox("Auto-Pause", self)
        self.autopause_checkbox.setChecked(True)
        self.note_label = QLabel("-", self)
        self.freq_label = QLabel("Freq: -", self)
        self.cents_label = QLabel("Cents: -", self)
        
        # Set only font sizes, let CSS handle font families
        note_font = QFont()
        note_font.setPixelSize(defaults.FONT_SIZE_NOTE_LABEL)
        self.note_label.setFont(note_font)
        
        freq_font = QFont()
        freq_font.setPixelSize(defaults.FONT_SIZE_FREQ_LABEL)
        self.freq_label.setFont(freq_font)
        
        cents_font = QFont()
        cents_font.setPixelSize(defaults.FONT_SIZE_CENTS_LABEL)
        self.cents_label.setFont(cents_font)
        
        self.note_label.setTextFormat(Qt.RichText)

        # Create value labels with small font
        value_font = QFont()
        value_font.setPixelSize(8)

        # Scroll Speed Slider
        scroll_layout = QVBoxLayout()
        self.scroll_speed_label = QLabel(f"Scroll Speed: {defaults.DEFAULT_SCROLL_SPEED}")
        self.scroll_speed_label.setFont(value_font)
        self.scroll_speed_label.setAlignment(Qt.AlignLeft)
        scroll_layout.addWidget(self.scroll_speed_label)

        scroll_slider_layout = QHBoxLayout()
        self.scroll_speed_slider = QSlider(Qt.Horizontal)
        self.scroll_speed_slider.setRange(defaults.MIN_SCROLL_SPEED, defaults.MAX_SCROLL_SPEED)
        self.scroll_speed_slider.setValue(defaults.DEFAULT_SCROLL_SPEED)
        self.scroll_speed_slider.setMinimumWidth(120)
        self.scroll_speed_slider.setMaximumWidth(120)
        self.scroll_speed_slider.valueChanged.connect(
            lambda v: self.scroll_speed_label.setText(f"Scroll Speed: {v}"))
        scroll_slider_layout.addWidget(self.scroll_speed_slider)
        scroll_layout.addLayout(scroll_slider_layout)

        # Gain Slider Layout Block
        gain_layout = QVBoxLayout()
        self.gain_label = QLabel(f"Gain: {defaults.DEFAULT_VOLUME_DB}")
        self.gain_label.setFont(value_font)
        self.gain_label.setAlignment(Qt.AlignLeft)
        gain_layout.addWidget(self.gain_label)

        gain_slider_layout = QHBoxLayout()
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(-10, 60)  # Set the slider range to -10 to 60
        self.gain_slider.setValue(defaults.DEFAULT_VOLUME_DB)
        self.gain_slider.setMinimumWidth(120)
        self.gain_slider.setMaximumWidth(120)
        # Re-add connection so that the gain label updates with the slider's value
        self.gain_slider.valueChanged.connect(lambda v: self.gain_label.setText(f"Gain: {v}"))
        gain_slider_layout.addWidget(self.gain_slider)
        gain_layout.addLayout(gain_slider_layout)

        # Noise Slider Layout Block
        noise_layout = QVBoxLayout()
        self.noise_label = QLabel(f"Noise Reject: {defaults.DEFAULT_SENSITIVITY}")
        self.noise_label.setFont(value_font)
        self.noise_label.setAlignment(Qt.AlignLeft)
        noise_layout.addWidget(self.noise_label)

        noise_slider_layout = QHBoxLayout()
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(defaults.DEFAULT_SENSITIVITY)
        self.noise_slider.setMinimumWidth(120)
        self.noise_slider.setMaximumWidth(120)
        self.noise_slider.valueChanged.connect(
            lambda v: self.noise_label.setText(f"Noise Reject: {v}"))
        noise_slider_layout.addWidget(self.noise_slider)
        noise_layout.addLayout(noise_slider_layout)


        self.noise_slider.valueChanged.connect(self.sensitivityChanged.emit)
        self.gain_slider.valueChanged.connect(self.volumeChanged.emit)


        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.pause_button)
        layout.addWidget(self.autopause_checkbox)
        layout.addLayout(scroll_layout)

        # --- New Separator between Scroll Speed and Gain ---
        separator_between = QFrame()
        separator_between.setFrameShape(QFrame.HLine)
        separator_between.setFrameShadow(QFrame.Sunken)
        separator_between.setStyleSheet("background-color: #444444;")
        layout.addSpacing(15)
        layout.addWidget(separator_between)
        layout.addSpacing(15)

        layout.addLayout(gain_layout)
        layout.addLayout(noise_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #444444;")
        layout.addSpacing(15)
        layout.addWidget(separator)
        layout.addSpacing(15)
        layout.addWidget(self.note_label)
        layout.addSpacing(5)
        layout.addWidget(self.freq_label)
        layout.addWidget(self.cents_label)

        layout.addWidget(separator)
        layout.addStretch()
        self.stats = QTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setFixedHeight(100)
        self.stats.setFixedWidth(140)
        self.stats.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stats.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stats.setStyleSheet("background-color: transparent; color: white; font-size: 12px; border: none;")
        layout.addWidget(self.stats)

        self.setLayout(layout)

        # Group labels by function
        self.note_label.setProperty("class", "pitch-display")
        self.freq_label.setProperty("class", "freq-display")
        self.cents_label.setProperty("class", "cents-display")

        self.scroll_speed_label.setProperty("class", "control-label")
        self.gain_label.setProperty("class", "control-label")
        self.noise_label.setProperty("class", "control-label")

    def _ensure_in_scale_label_exists(self):
        """Make sure the in_scale_label exists, create it if it doesn't."""
        # This method is no longer needed, but we keep it empty to avoid breaking existing code
        pass
    
    def update_pitch(self, full_note, frequency, cents):
        """
        Update the pitch display with new pitch information.
        
        Args:
            full_note (str): The detected note name with octave.
            frequency (float): The detected frequency in Hz.
            cents (int): The cent deviation from the equal-tempered pitch.
        """
        # No longer need to ensure in_scale_label exists
        
        if frequency <= 0:
            self.note_label.setText("--")
            self.freq_label.setText("-- Hz")
            self.cents_label.setText("--")
            self.staff_note = None
            self.frequency = 0.0  # Add this line to update the frequency
            self.update()
            return
            
        # Store the frequency for use in paintEvent
        self.frequency = frequency  # Add this line to update the frequency
        
        # Extract the note name without octave
        note_parts = full_note.split("<sub>")
        note_name = note_parts[0].strip()
        octave_val = note_parts[1].split("</sub>")[0] if len(note_parts) > 1 else ""
        
        # Extract just the letter part for scale comparison (remove solfege if present)
        if " " in note_name:
            parts = note_name.split()
            solfege_part = parts[0]
            letter_part = parts[1]
        else:
            solfege_part = ""
            letter_part = note_name
        
        # Format cents display
        if cents > 0:
            self.cents_label.setText(f"+{cents}")
        else:
            self.cents_label.setText(f"{cents}")
        
        # Set default text color
        note_color = QColor(255, 255, 255)  # Default white
        scale_match = False
        
        # Default to showing the original note name
        updated_note = note_name
        
        # Get the MIDI value of the detected note for comparison
        detected_midi = int(round(69 + 12 * math.log2(frequency / defaults.TUNING_FREQUENCY)))
        detected_note_index = detected_midi % 12
            
        # Get the current scale for highlighting
        if hasattr(defaults, 'generate_scale') and defaults.current_scale != "-":
            # Check if this is a chromatic scale
            is_chromatic = (defaults.current_scale == "Chromatic")
            
            # Get the current scale with proper enharmonic handling
            current_scale = defaults.generate_scale()
            
            # Get canonical versions for comparison
            canonical_detected = canonical_note(letter_part)
            canonical_scale = [canonical_note(n) for n in current_scale]
            
            # Calculate how close the detected note is to a scale note
            if canonical_detected in canonical_scale:
                # The note is exactly in the scale, use the PITCH_LINE_DRAW_COLOR
                note_color = QColor(defaults.PITCH_LINE_DRAW_COLOR)
                scale_match = True
            else:
                # Check if we're close to a scale note (based on cents deviation)
                # Get the exact frequency of the detected note without cents deviation
                exact_detected_freq = defaults.TUNING_FREQUENCY * (2 ** ((detected_midi - 69) / 12))
                
                # Find the closest scale note
                closest_distance = float('inf')
                closest_midi = None
                closest_scale_note = None
                closest_index = None
                
                for i, scale_note in enumerate(current_scale):
                    scale_midi = None
                    # Convert each scale note to MIDI
                    for j, sharp_note in enumerate(SHARP_LETTER_NAMES):
                        if canonical_note(scale_note) == canonical_note(sharp_note):
                            scale_midi = detected_midi - (detected_note_index - j) % 12
                            break
                    
                    if scale_midi is not None:
                        # Calculate distance to this scale note
                        scale_freq = defaults.TUNING_FREQUENCY * (2 ** ((scale_midi - 69) / 12))
                        cents_diff = abs(1200 * math.log2(frequency / scale_freq))
                        
                        if cents_diff < closest_distance:
                            closest_distance = cents_diff
                            closest_midi = scale_midi
                            closest_scale_note = scale_note
                            closest_index = i
                
                # Color based on proximity to closest scale note
                if closest_distance <= 50:  # Within 50 cents of a scale note
                    # Interpolate color: white -> PITCH_LINE_DRAW_COLOR based on proximity
                    proximity = 1.0 - (closest_distance / 50.0)  # 1.0 = exact match, 0.0 = 50 cents away
                    target_color = QColor(defaults.PITCH_LINE_DRAW_COLOR)
                    
                    r = int(255 + proximity * (target_color.red() - 255))
                    g = int(255 + proximity * (target_color.green() - 255))
                    b = int(255 + proximity * (target_color.blue() - 255))
                    
                    note_color = QColor(r, g, b)
                    
                    # Show the closest scale note if we're very close (within 25 cents)
                    if closest_distance <= 25 and closest_scale_note is not None:
                        scale_match = True
                        # We'll set the matching_scale_note below
                        matching_scale_note = closest_scale_note
            
            # Update note display based on whether it's in the scale
            if scale_match:
                if is_chromatic:
                    prefer_sharps = use_sharps_for_key()
                    
                    if defaults.PITCH_LABEL_SOLFEGE:
                        if prefer_sharps:
                            correct_solfege = SHARP_SOLFEGE_NAMES[detected_note_index]
                        else:
                            correct_solfege = FLAT_SOLFEGE_NAMES[detected_note_index]
                        
                        if defaults.PITCH_LABEL_LETTERS:
                            # Show both solfege and letter name
                            if prefer_sharps:
                                updated_note = f"{correct_solfege} {SHARP_LETTER_NAMES[detected_note_index]}"
                            else:
                                updated_note = f"{correct_solfege} {FLAT_LETTER_NAMES[detected_note_index]}"
                        else:
                            # Show just solfege
                            updated_note = correct_solfege
                        
                        # Add the octave if needed
                        if defaults.PITCH_LABEL_OCTAVE and octave_val:
                            updated_note += f"<sub>{octave_val}</sub>"
                            
                        self.note_label.setText(format_note_with_html(updated_note))
                    else:
                        # Show just the letter name for natural notes mode
                        if prefer_sharps:
                            updated_note = SHARP_LETTER_NAMES[detected_note_index]
                        else:
                            updated_note = FLAT_LETTER_NAMES[detected_note_index]
                            
                        # Add the octave if needed
                        if defaults.PITCH_LABEL_OCTAVE and octave_val:
                            updated_note += f"<sub>{octave_val}</sub>"
                            
                        self.note_label.setText(format_note_with_html(updated_note))
                else:
                    # Non-chromatic scale - find the matching scale note
                    if 'matching_scale_note' not in locals():
                        matching_scale_note = None
                        for i, scale_canon in enumerate(canonical_scale):
                            if scale_canon == canonical_detected:
                                matching_scale_note = current_scale[i]
                                break
                    
                    # Update note display with correct notation from the scale
                    if defaults.PITCH_LABEL_SOLFEGE and matching_scale_note:
                        correct_solfege = note_to_solfege(matching_scale_note)
                        
                        if defaults.PITCH_LABEL_LETTERS:
                            # Replace the solfege part but keep the letter part from the scale
                            updated_note = f"{correct_solfege} {matching_scale_note}"
                        else:
                            # Just show the correct solfege
                            updated_note = correct_solfege
                        
                        # Add the octave if needed
                        if defaults.PITCH_LABEL_OCTAVE and octave_val:
                            updated_note += f"<sub>{octave_val}</sub>"
                            
                        self.note_label.setText(format_note_with_html(updated_note))
                    elif not defaults.PITCH_LABEL_SOLFEGE and matching_scale_note:
                        # Show just the letter part from the scale when in natural notes mode
                        updated_note = matching_scale_note
                        
                        # Add the octave if needed
                        if defaults.PITCH_LABEL_OCTAVE and octave_val:
                            updated_note += f"<sub>{octave_val}</sub>"
                            
                        self.note_label.setText(format_note_with_html(updated_note))

            else:
                # For non-matching notes, we have options:
                # 1. Show the original detected note (previous behavior)
                # 2. Show the closest diatonic note but with a different color
                
                # Option 2: Always show notes with proper diatonic spelling
                if not is_chromatic and defaults.current_scale != "-":
                    # Find the appropriate enharmonic spelling based on the key signature
                    prefer_sharps = use_sharps_for_key()
                    
                    if prefer_sharps:
                        diatonic_letter = SHARP_LETTER_NAMES[detected_note_index]
                        diatonic_solfege = SHARP_SOLFEGE_NAMES[detected_note_index]
                    else:
                        diatonic_letter = FLAT_LETTER_NAMES[detected_note_index]
                        diatonic_solfege = FLAT_SOLFEGE_NAMES[detected_note_index]
                    
                    if defaults.PITCH_LABEL_SOLFEGE:
                        if defaults.PITCH_LABEL_LETTERS:
                            updated_note = f"{diatonic_solfege} {diatonic_letter}"
                        else:
                            updated_note = diatonic_solfege
                            
                        if defaults.PITCH_LABEL_OCTAVE and octave_val:
                            updated_note += f"<sub>{octave_val}</sub>"
                            
                        self.note_label.setText(format_note_with_html(updated_note))
                    else:
                        # Update with proper diatonic letter when solfege is disabled
                        updated_note = diatonic_letter
                        if defaults.PITCH_LABEL_OCTAVE and octave_val:
                            updated_note += f"<sub>{octave_val}</sub>"
                        self.note_label.setText(format_note_with_html(updated_note))
                else:
                    # Just update with the detected note
                    self.note_label.setText(format_note_with_html(updated_note))
        else:
            # No scale selected, just use default display
             self.note_label.setText(format_note_with_html(updated_note))
        
        # Apply the calculated color to the note label
        self.note_label.setStyleSheet(f"color: rgb({note_color.red()}, {note_color.green()}, {note_color.blue()}); font-size: {defaults.FONT_SIZE_NOTE_LABEL}px; font-weight: bold;")
        
        # Update frequency display
        self.freq_label.setText(f"{frequency:.1f} Hz")
        
        # Update the staff display
        self.staff_note = note_name
        self.update()

    def update_info_message(self, message):
        self.stats.setHtml(message)
        #self.stats.moveCursor(QTextCursor.End)

    def paintEvent(self, event):
        """
        Custom paint event to draw the musical staff and note on the panel.
        """
        super(PitchInfoPanel, self).paintEvent(event)
        upper_staff_top = defaults.DEFAULT_STAFF_TOP_OFFSET
        upper_staff_height = defaults.DEFAULT_STAFF_HEIGHT

        painter = QPainter(self)
        rect = self.rect()
        painter.setPen(QPen(defaults.STAFF_MAIN_COLOR))
        for i in range(5):
            y_line = int(upper_staff_top + i * (upper_staff_height / 4))
            painter.setPen(QPen(defaults.STAFF_MAIN_COLOR, 1))
            painter.drawLine(10, y_line, self.width() - 10, y_line)

        staff_middle_width = self.width() / 2 + 7
        ledger_y = int(upper_staff_top + upper_staff_height + (upper_staff_height / 4))
        f_staff_height = defaults.DEFAULT_STAFF_HEIGHT
        spacing_bass = f_staff_height / 8
        f_staff_top = ledger_y + 8 * spacing_bass - 0.75 * f_staff_height
        for i in range(5):
            y_line = int(f_staff_top + i * (f_staff_height / 4))
            painter.setPen(QPen(defaults.STAFF_MAIN_COLOR, 1))
            painter.drawLine(10, y_line, self.width() - 10, y_line)

        painter.setPen(QPen(defaults.STAFF_CLEF_COLOR))
        treble_font = QFont()
        treble_font.setPixelSize(38)
        painter.setFont(treble_font)
        painter.drawText(15, int(upper_staff_top + (upper_staff_height / 4 * 3) + 8), "𝄞")
        
        bass_font = QFont()
        bass_font.setPixelSize(30)
        painter.setFont(bass_font)
        painter.drawText(17, int(f_staff_top + (f_staff_height / 4) + 14), "𝄢")

        try:
            _, _, midi, _ = frequency_to_note(self.frequency, defaults.global_prefer_sharps)
        except Exception:
            midi = 85

        if midi > 0:
            spacing_treble = upper_staff_height / 8
            spacing_bass = f_staff_height / 8
            staff_accidental = ""
            if midi >= 60:
                semitone_count = (midi - 60) % 12
                octave_count = (midi - 60) // 12
            else:
                semitone_count = 11 - (abs(midi - 59) % 12)
                octave_count = abs(midi - 59) // 12 + 1

            if semitone_count == 0:
                staff_spaces = 0
            if semitone_count == 1:
                if defaults.global_prefer_sharps:
                    staff_spaces = 0
                    staff_accidental = "#"
                else:
                    staff_spaces = 1
                    staff_accidental = "♭"
            if semitone_count == 2:
                staff_spaces = 1
            if semitone_count == 3:
                if defaults.global_prefer_sharps:
                    staff_spaces = 1
                    staff_accidental = "#"
                else:
                    staff_spaces = 2
                    staff_accidental = "♭"
            if semitone_count == 4:
                staff_spaces = 2
            if semitone_count == 5:
                staff_spaces = 3
            if semitone_count == 6:
                if defaults.global_prefer_sharps:
                    staff_spaces = 3
                    staff_accidental = "#"
                else:
                    staff_spaces = 4
                    staff_accidental = "♭"
            if semitone_count == 7:
                staff_spaces = 4
            if semitone_count == 8:
                if defaults.global_prefer_sharps:
                    staff_spaces = 4
                    staff_accidental = "#"
                else:
                    staff_spaces = 5
                    staff_accidental = "♭"
            if semitone_count == 9:
                staff_spaces = 5
            if semitone_count == 10:
                if defaults.global_prefer_sharps:
                    staff_spaces = 5
                    staff_accidental = "#"
                else:
                    staff_spaces = 6
                    staff_accidental = "♭"
            if semitone_count == 11:
                staff_spaces = 6

            if midi >= 60:
                staff_spaces += octave_count * 7
                y_note = (upper_staff_top + upper_staff_height + upper_staff_height / 4) - staff_spaces * spacing_treble
            else:
                staff_spaces = ((octave_count * 7) * -1) + staff_spaces
                y_note = (upper_staff_top + upper_staff_height + upper_staff_height / 4) - staff_spaces * spacing_bass

            painter.setBrush(defaults.STAFF_MAIN_COLOR)
            painter.setPen(defaults.STAFF_MAIN_COLOR)

            if midi >= 60:
                if y_note < upper_staff_top:
                    num_ledgers = int((upper_staff_top - y_note) / (upper_staff_height / 4))
                    for i in range(num_ledgers):
                        ledger_line_y = int(upper_staff_top - (i + 1) * (upper_staff_height / 4))
                        painter.drawLine(int(staff_middle_width - 10), ledger_line_y, int(staff_middle_width + 10), ledger_line_y)
                elif y_note > upper_staff_top + upper_staff_height:
                    num_ledgers = int((y_note - (upper_staff_top + upper_staff_height)) / (upper_staff_height / 4))
                    for i in range(num_ledgers):
                        ledger_line_y = int(upper_staff_top + upper_staff_height + (i + 1) * (upper_staff_height / 4))
                        painter.drawLine(int(staff_middle_width - 10), ledger_line_y, int(staff_middle_width + 10), ledger_line_y)
            else:
                if y_note < f_staff_top:
                    num_ledgers = int((f_staff_top - y_note) / (f_staff_height / 4))
                    for i in range(num_ledgers):
                        ledger_line_y = int(f_staff_top - (i + 1) * (f_staff_height / 4))
                        painter.drawLine(int(staff_middle_width - 10), ledger_line_y, int(staff_middle_width + 10), ledger_line_y)
                elif y_note > f_staff_top + f_staff_height:
                    num_ledgers = int((y_note - (f_staff_top + f_staff_height)) / (f_staff_height / 4))
                    for i in range(num_ledgers):
                        ledger_line_y = int(f_staff_top + f_staff_height + (i + 1) * (f_staff_height / 4))
                        painter.drawLine(int(staff_middle_width - 10), ledger_line_y, int(staff_middle_width + 10), ledger_line_y)

            painter.setBrush(defaults.STAFF_NOTE_COLOR)
            # Draw a rotated note head (ellipse) to simulate a quarter note head rotated to the left
            painter.save()
            # Translate to the center of the note head (which is at staff_middle_width, y_note)
            painter.translate(staff_middle_width, y_note)
            painter.rotate(-15)  # Rotate 15 degrees to the left
            # Draw the ellipse centered at (0,0): starting at (-6, -4) yields the correct center
            painter.drawEllipse(-6, -4, 12, 8)
            painter.restore()

            # Draw note stem (tail) to appear like a crotchet (quarter note)
            stem_width = 2
            stem_length = 30  # Adjust this value to change the stem length
            # Determine the vertical center of the staff
            if midi >= 60:
                staff_center = upper_staff_top + upper_staff_height / 2
            else:
                staff_center = f_staff_top + f_staff_height / 2

            painter.setPen(QPen(defaults.STAFF_NOTE_COLOR, stem_width))
            if y_note > staff_center:
                # Note is below the staff center: draw stem upward from the right side of the note head
                start_x = int(staff_middle_width + 5)
                start_y = int(y_note)
                end_y = start_y - stem_length
                painter.drawLine(start_x, start_y, start_x, end_y)
            else:
                # Note is above the staff center: draw stem downward from the left side of the note head
                start_x = int(staff_middle_width - 5)
                start_y = int(y_note)
                end_y = start_y + stem_length
                painter.drawLine(start_x, start_y, start_x, end_y)

            painter.setPen(defaults.STAFF_NOTE_COLOR)
            accidental_font = QFont()
            accidental_font.setPixelSize(16)
            painter.setFont(accidental_font)
            painter.drawText(int(staff_middle_width - 22), int(y_note + 7), staff_accidental)

        painter.end()


# --------------------------- Main Application Window ---------------------------
class MainWindow(QMainWindow):
    """
    The main window for the application. It combines the Piano Roll display,
    Pitch Information Panel, and the control widgets. Also responsible for initializing
    and managing the audio processing thread.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Real-Time Vocal Pitch Monitor")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "icons/diapason.ico")))
        stylesheet = load_stylesheet("style.css")
        # Append custom styling for QComboBox and QPushButton:
        custom_style = """
		"""
        if stylesheet:
            self.setStyleSheet(stylesheet + custom_style)
        else:
            self.setStyleSheet(custom_style)

        # Apply window geometry from defaults (loaded from config if available)
        #self.resize(defaults.DEFAULT_WINDOW_WIDTH, defaults.DEFAULT_WINDOW_HEIGHT)
        #self.move(defaults.DEFAULT_WINDOW_POS_X, defaults.DEFAULT_WINDOW_POS_Y)
        self.setGeometry(
            defaults.DEFAULT_WINDOW_POS_X,
            defaults.DEFAULT_WINDOW_POS_Y,
            defaults.DEFAULT_WINDOW_WIDTH,
            defaults.DEFAULT_WINDOW_HEIGHT
        )

        self.initUI()
        self.initAudioWorker()

    def log_debug_message(self, message):
        """
        Logs a debug message. The message is converted to a string, appended with a newline,
        printed to the console, and appended to the debug log (if enabled).
        """
        msg = str(message)
        print(msg)
        if self.debug_log is not None:
            self.debug_log.append(msg)
            self.debug_log.moveCursor(QTextCursor.End)

    def initUI(self):
        """
        Initialize the main user interface, layout widgets, and connect signals.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Create debug log widget early so that log_debug_message() works in populate_input_devices().
        if defaults.ENABLE_DEBUG:
            self.debug_log = QTextEdit()
            self.debug_log.setReadOnly(True)
            self.debug_log.setProperty("class", "debug_log")
            self.debug_log.setFixedHeight(150)
        else:
            self.debug_log = None

        self.log_debug_message(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%S')}")

        top_layout = QHBoxLayout()
        self.piano_roll = PianoRollWidget()
        self.pitch_info = PitchInfoPanel()
        top_layout.addWidget(self.piano_roll, 1)
        top_layout.addWidget(self.pitch_info)

        main_layout.addLayout(top_layout)

        controls_layout = QHBoxLayout()

        # ------ Left Controls Layout ------
        left_controls = QHBoxLayout()
        self.input_source_combo = QComboBox()
        self.input_source_combo.setMaximumWidth(200)
        self.input_source_combo.view().setMinimumWidth(400)
        self.input_source_combo.setMinimumContentsLength(100)
        self.populate_input_devices()  # This now logs debug messages.
        left_controls.addWidget(self.input_source_combo)

        self.pitch_method_combo = QComboBox()
        self.pitch_method_combo.addItems(["YIN", "PYIN", "MPM", "SWIPE", "Cepstrum", "HPS", "Autocorrelation"])
        index = self.pitch_method_combo.findText(defaults.current_pitch_method)
        if index >= 0:
            self.pitch_method_combo.setCurrentIndex(index)
        left_controls.addWidget(self.pitch_method_combo)

        controls_layout.addLayout(left_controls)

        #separator = QFrame()
        #separator.setFrameShape(QFrame.VLine)
        #separator.setFixedWidth(1)
        #separator.setStyleSheet("background-color: #444444;")
        #controls_layout.addWidget(separator)

        controls_layout.addStretch(1)
        controls_layout.addStretch(2)
        controls_layout.addStretch(3)

        # ------ Right Controls Layout ------
        right_controls = QHBoxLayout()
        # --- New: Note Format ComboBox ---
        #right_controls.addWidget(QLabel("Notes"))
        self.note_format_combo = QComboBox()
        self.note_format_combo.addItems(["Solfege", "Natural Notes"])
        # Set default based on the current configuration:
        if defaults.PITCH_LABEL_SOLFEGE:
            self.note_format_combo.setCurrentText("Solfege")
        else:
            self.note_format_combo.setCurrentText("Natural Notes")
        right_controls.addWidget(self.note_format_combo)

        self.enharmonic_combo = QComboBox()
        self.enharmonic_combo.addItems(["Sharps", "Flats"])
        index = self.enharmonic_combo.findText(defaults.DEFAULT_ENHARMONIC_NOTATION)
        if index >= 0:
            self.enharmonic_combo.setCurrentIndex(index)
        #right_controls.addWidget(QLabel("Enharmonic"))
        right_controls.addWidget(self.enharmonic_combo)

        self.root_note_combo = QComboBox()
        self.root_note_combo.addItems(list(defaults.NOTE_TO_INDEX.keys()))
        self.root_note_combo.setCurrentText(defaults.current_root_note)
        self.root_label = QLabel("Root", self)
        self.root_label.setProperty("class", "root-label")
        right_controls.addWidget(self.root_label)
        right_controls.addWidget(self.root_note_combo)


        self.scale_combo = QComboBox()
        self.scale_combo.addItems(defaults.SCALE_OPTIONS)
        self.scale_combo.setCurrentText(defaults.current_scale)
        self.scale_label = QLabel("Scale", self)
        self.scale_label.setProperty("class", "scale-label")
        right_controls.addWidget(self.scale_label)
        right_controls.addWidget(self.scale_combo)

        controls_layout.addLayout(right_controls)

        main_layout.addLayout(controls_layout)
        if defaults.ENABLE_DEBUG:
            main_layout.addWidget(self.debug_log)

        # ----- Signal connections -----
        # Only connect signals for visible widgets.
        # self.sensitivity_slider.valueChanged.connect(self.change_sensitivity)
        # self.volume_slider.valueChanged.connect(self.change_volume)
        self.pitch_info.scroll_speed_slider.valueChanged.connect(self.change_scroll_speed)
        self.input_source_combo.currentIndexChanged.connect(self.change_input_source)
        self.enharmonic_combo.currentIndexChanged.connect(self.change_enharmonic)
        # self.boost_checkbox.toggled.connect(self.change_boost_mode)
        # self.extra_sens_checkbox.toggled.connect(self.change_extra_sensitivity)
        self.pitch_method_combo.currentIndexChanged.connect(self.change_pitch_method)
        self.root_note_combo.currentIndexChanged.connect(self.change_root_note)
        self.scale_combo.currentIndexChanged.connect(self.change_scale)
        self.pitch_info.volumeChanged.connect(self.change_volume)
        self.pitch_info.sensitivityChanged.connect(self.change_sensitivity)
        self.note_format_combo.currentIndexChanged.connect(self.change_note_format)

        self.pitch_info.pause_button.clicked.connect(self.toggle_pause)
        self.pitch_info.autopause_checkbox.toggled.connect(self.change_autopause)

        # Add global spacebar shortcut for pause toggle
        self.pause_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.pause_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)  # Make it work anywhere in the application
        self.pause_shortcut.activated.connect(self.toggle_pause)

    def log_debug_message(self, message):
        """
        Logs a debug message. The message is converted to a string, appended with a newline,
        printed to the console, and appended to the debug log (if enabled).
        """
        msg = str(message)
        print(msg)
        if self.debug_log is not None:
            self.debug_log.append(msg)
            self.debug_log.moveCursor(QTextCursor.End)

    def initAudioWorker(self):
        """
        Initialize and start the audio processing thread (AudioStreamWorker).
        """
        self.log_debug_message("Initializing audio worker...")

        default_index = self.input_source_combo.itemData(self.input_source_combo.currentIndex())
        try:
            # Create the callback handler (which opens the stream) on the main thread.
            callback_handler = AudioStreamCallbackHandler(
                default_index,
                defaults.SAMPLE_RATE,
                defaults.CHUNK_SIZE,
                self.log_debug_message
            )
        except Exception as e:
            self.log_debug_message(f"Error opening stream: {e}")
            callback_handler = None

        if callback_handler is None:
            self.log_debug_message("No audio input device available. Audio input will be disabled.")
            # Create a dummy audio worker that does nothing to keep the app running without audio
            class DummyAudioWorker(QThread):
                def start(self): pass
                def stop(self): pass
            self.audio_worker = DummyAudioWorker()
        else:
            self.audio_worker = AudioStreamWorker(callback_handler)
            # Set the desired pitch detection method.
            self.audio_worker.set_pitch_method(defaults.current_pitch_method)
            self.log_debug_message("Pitch detection algorithm set to: " + str(defaults.current_pitch_method))
            self.audio_worker.debugMessage.connect(self.log_debug_message)
            self.audio_worker.statsText.connect(self.update_stats_message)
            self.audio_worker.pitchDetected.connect(self.handle_pitch_detected)
            self.audio_worker.stateChanged.connect(self.handle_state_changed)
            self.audio_worker.vuMeterUpdate.connect(self.piano_roll.update_vu_meter)
            self.audio_worker.vuMeterPostUpdate.connect(self.piano_roll.update_vu_meter_post)
            self.audio_worker.errorOccurred.connect(self.handle_error)
            self.audio_worker.start()

    def populate_input_devices(self):
        """
        Populate the input device combo box with available audio input devices.
        Uses sounddevice to detect available devices and prepends [ASIO] or [WASAPI]
        to the device name if applicable. Only devices with host API names containing "ASIO" or "WASAPI" are listed.
        Also logs detailed device and host API information.
        """
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as e:
            self.log_debug_message(f"Error querying devices: {e}")
            return

        self.log_debug_message("Host APIs:")
        for idx, api in enumerate(hostapis):
            self.log_debug_message(f"  {idx}: {api.get('name', 'Unknown')}")

        self.input_source_combo.clear()

        default_device_index = defaults.DEFAULT_INPUT_SOURCE_INDEX
        if default_device_index is None:
            if sys.platform.startswith("win"):
                # In Windows, attempt to set default input device using sd.default.device if available…
                if sd.default.device is not None and isinstance(sd.default.device, (list, tuple)) and len(sd.default.device) > 0:
                    default_device_index = sd.default.device[0]
                else:
                    try:
                        default_info = sd.query_devices(None, 'input')
                        default_device_index = default_info['index']
                    except Exception as e:
                        self.log_debug_message("Could not get default input device: " + str(e))
            else:
                if sd.default.device is not None and isinstance(sd.default.device, (list, tuple)) and len(sd.default.device) > 0:
                    default_device_index = sd.default.device[0]

        found_devices = False
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            hostapi_index = dev.get("hostapi", None)
            if hostapi_index is None or hostapi_index >= len(hostapis):
                continue
            api_name = hostapis[hostapi_index].get("name", "").upper()


            #if "ASIO" not in api_name and "WASAPI" not in api_name: continue
            #if "MME" not in api_name: continue
            if \
                "ASIO" not in api_name and \
                "MME" not in api_name and \
                "WASAPI" not in api_name: continue
                #"WDM-KS" not in api_name and \

            if "ASIO" in api_name:
                device_name = f"[ASIO] {dev.get('name', f'Device {i}')}"
            elif "WASAPI" in api_name:
                device_name = f"[WASAPI] {dev.get('name', f'Device {i}')}"
            elif "WDM-KS" in api_name:
                device_name = f"[WDM-KS] {dev.get('name', f'Device {i}')}"
            elif "MME" in api_name:
                device_name = f"[MME] {dev.get('name', f'Device {i}')}"
            else:
                device_name = f"{dev.get('name', f'Device {i}')}"

            self.input_source_combo.addItem(device_name, i)
            found_devices = True
            #self.log_debug_message(f"  {i}: {device_name}")

        if not found_devices:
            self.input_source_combo.addItem("No ASIO/WASAPI input devices found", -1)
            self.log_debug_message("No ASIO/WASAPI input devices found.")

        elif default_device_index is not None:
            idx = self.input_source_combo.findData(default_device_index)
            if idx >= 0:
                self.input_source_combo.setCurrentIndex(idx)
                self.log_debug_message(f"Default device set to index: {default_device_index}")

    def change_pitch_method(self, index):
        """
        Handle changes to the pitch detection method.
        """
        method = self.pitch_method_combo.currentText()
        defaults.current_pitch_method = method
        self.audio_worker.set_pitch_method(method)
        self.log_debug_message(f"Pitch method changed to: {method}")

    def toggle_pause(self):
        """
        Toggle pause/resume of the audio processing.
        """
        if self.audio_worker.user_paused:
            self.audio_worker.resume()
            self.pitch_info.pause_button.setText("Pause")
            # Remove manual pause override to revert to the default styling.
            self.pitch_info.pause_button.setStyleSheet("")
        else:
            self.audio_worker.pause()
            self.pitch_info.pause_button.setText("Resume")
            # Set background to the same magenta as the scrollbar handles.
            self.pitch_info.pause_button.setStyleSheet("background-color: #D6336C;")

    def change_sensitivity(self, value):
        # Update default sensitivity so changes persist in config file
        defaults.DEFAULT_SENSITIVITY = value
        self.audio_worker.set_sensitivity(value)

    def change_volume(self, value):
        # Update default volume so changes persist in config file
        defaults.DEFAULT_VOLUME_DB = value
        self.audio_worker.set_volume_db(value)

    def change_scroll_speed(self, value):
        # Update default scroll speed so changes persist in config file
        defaults.DEFAULT_SCROLL_SPEED = value
        self.piano_roll.set_scroll_speed(value)

    def change_input_source(self, index):
        """
        Handle changes in the selected input audio device.
        """
        device_index = self.input_source_combo.itemData(index)
        self.log_debug_message(f"Changing input device to index: {device_index}")
        try:
            # Create a new callback handler on the main thread.
            new_callback_handler = AudioStreamCallbackHandler(
                device_index,
                defaults.SAMPLE_RATE,
                defaults.CHUNK_SIZE,
                self.log_debug_message
            )
            # Replace the existing handler in the audio worker.
            self.audio_worker.replace_callback_handler(new_callback_handler)
            defaults.DEFAULT_INPUT_SOURCE_INDEX = device_index
        except Exception as e:
            self.log_debug_message(f"Error setting input device: {e}")

    def change_enharmonic(self, index):
        """
        Handle changes to the enharmonic notation style (sharps/flats).
        """
        defaults.global_prefer_sharps = (self.enharmonic_combo.currentText() == "Sharps")

    def change_boost_mode(self, checked):
        """
        Enable or disable boost mode.
        """
        self.audio_worker.set_boost_mode(checked)

    def change_extra_sensitivity(self, checked):
        """
        Enable or disable extra sensitivity mode.
        """
        self.audio_worker.set_extra_sensitivity(checked)

    def change_root_note(self, index):
        """
        Update the root note used for scale highlighting.
        """
        defaults.current_root_note = self.root_note_combo.currentText()
        # Update any UI elements that show the scale
        self.update_scale_display()
    
    def change_scale(self, index):
        """
        Update the scale/mode used for highlighting the piano roll.
        """
        defaults.current_scale = self.scale_combo.currentText()
        # Update any UI elements that show the scale
        self.update_scale_display()
    
    def update_scale_display(self):
        """Update any UI elements that display the current scale."""
        if defaults.current_scale == "-":
            self.log_debug_message("Scale display: No scale selected")
            return

        scale_notes = defaults.generate_scale()
        scale_solfege = defaults.generate_scale(with_solfege=True)

        root_note = defaults.current_root_note
        scale_type = defaults.current_scale

        self.log_debug_message(f"Current scale: {root_note} {scale_type}")
        self.log_debug_message(f"Notes: {', '.join(scale_notes)}")
        self.log_debug_message(f"Solfege: {', '.join(scale_solfege)}")

    def change_autopause(self, checked):
        """
        Enable or disable auto-pause functionality based on the checkbox.
        """
        self.audio_worker.auto_pause_enabled = checked

    def handle_pitch_detected(self, effective_time, pitch, rms):
        """
        Slot to handle detected pitch events.

        Args:
            effective_time (float): The time at which the pitch was detected.
            pitch (float): The detected pitch frequency.
            rms (float): The post-gain RMS level.
        """
        if pitch > 0:
            full_note, note_no_octave, midi, cents = frequency_to_note(pitch, defaults.global_prefer_sharps)
            self.pitch_info.update_pitch(full_note, pitch, cents)
            self.piano_roll.add_pitch_point(effective_time, pitch, note_no_octave, cents)
        elif pitch == -1.0:
            self.piano_roll.add_break_point(effective_time)
        else:
            # Only update piano roll's latest time if auto pause is NOT enabled.
            if not self.audio_worker.auto_pause_enabled:
                if effective_time > self.piano_roll.latest_time:
                    self.piano_roll.latest_time = effective_time
                self.piano_roll.update()

    def handle_state_changed(self, time_stamp, paused):
        # time_stamp (float): The effective time of the state change.
        # paused (bool): The new pause state.
        self.piano_roll.add_marker(time_stamp, paused)
        if not paused:
            self.piano_roll.add_break_point(time_stamp)

    def handle_error(self, error_message):
        """
        Display error messages received from the audio worker in a dialog.
        """
        print("Audio Worker Error:", error_message)
        QMessageBox.critical(self, "Audio Error", f"An error occurred while opening the audio stream:\n\n{error_message}")

    def closeEvent(self, event):
        """
        Ensure proper shutdown by stopping the audio processing thread and saving window geometry.
        """
        self.audio_worker.stop()
        try:
            defaults.updateWindowGeometry(self)
            defaults.SaveConfigToFile()
        except Exception as e:
            print(f"Error saving configuration: {e}")
        event.accept()

    def update_stats_message(self, audio_stats):
        if defaults.ENABLE_DEBUG:
            pr_fps = self.piano_roll.get_pr_fps()
            combined_stats = f"{audio_stats}<br>DrawRate: {pr_fps:.1f} Hz"
            self.pitch_info.update_info_message(combined_stats)

    def change_note_format(self, index):
        """
        Update the note format used in pitch display based on the combo box selection.
        If 'Solfege' is selected, use solfege notation.
        If 'Natural' is selected, use natural (letter) notation.
        """
        format_choice = self.note_format_combo.currentText()
        if format_choice == "Solfege":
            defaults.PITCH_LABEL_SOLFEGE = True
            defaults.PITCH_LABEL_LETTERS = False
        elif format_choice == "Natural Notes":
            defaults.PITCH_LABEL_SOLFEGE = False
            defaults.PITCH_LABEL_LETTERS = True

        # Force an update of UI components so that changes are reflected immediately.
        self.piano_roll.update()
        self.pitch_info.update()
        self.log_debug_message(f"Note format changed to: {format_choice}")





# --------------------------- Main Entry Point ---------------------------
if __name__ == '__main__':
    # If hardware acceleration is disabled, force Qt to use software OpenGL.
    if not defaults.ENABLE_HW_ACCELERATION:
        from PySide6.QtCore import Qt
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)

    defaults.LoadConfigFromFile()

    window = MainWindow()
    window.show()

    exit_code = app.exec()
    try:
        defaults.updateWindowGeometry(window)
        defaults.SaveConfigToFile()
    except Exception as e:
        print(f"Error during exit configuration save: {e}")
    sys.exit(exit_code)


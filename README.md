# 🐱 Mood Swings

A real-time webcam app that reads your facial expression and overlays a matching cat reaction image. Built with MediaPipe Face Landmarker and OpenCV.

---

## What it does

- Detects your face with MediaPipe Face Detection
- Reads 52 facial **blendshapes** (jaw open, smile, brow raise, etc.) using MediaPipe Face Landmarker
- Compares them to your **calibrated neutral face** using z-scores
- Classifies the expression into a pose: `happy`, `surprised`, `angry`, `sad`, `sleepy`, or `neutral`
- Overlays a cat image on your face for ~0.8 seconds when you make an expression

---

## Why calibration?

Every face is different — a big smile for one person is a slight grin for another. Instead of hardcoded thresholds, this app records your neutral face once, then measures how far your expressions deviate from it (in standard deviations). Detection becomes personal and consistent.

---

## Requirements

- Python 3.10–3.12
- A webcam
- Windows, macOS, or Linux

### Dependencies

See `requirements.txt`:
```
opencv-python
mediapipe==0.10.21
numpy
```

**Note:** MediaPipe must be `<= 0.10.21`. Newer versions removed the `mp.solutions` API this project relies on.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/mood-swings.git
cd mood-swings

python -m venv kitty
kitty\Scripts\activate         # Windows
# source kitty/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

---

## Usage

```bash
cd src
python main.py
```

### First run

1. The app auto-downloads `face_landmarker.task` (~3.7 MB) into `src/models/`
2. Press **`C`** to calibrate
3. Hold a relaxed, neutral face for ~2 seconds while the yellow countdown runs
4. Terminal prints `Calibration done.`
5. Try expressions — the cat overlay appears for a moment

### Keyboard controls

| Key | Action |
|---|---|
| `C` | Start calibration (neutral face, ~2 seconds) |
| `S` | Save a screenshot to `src/captures/` |
| `Q` | Quit |

---

## Project structure

```
mood-swings/
├── src/
│   ├── main.py                     # Loop, keys, orchestration
│   ├── detector.py                 # Face detection + crop
│   ├── expressions.py              # Blendshapes, calibration, classification
│   ├── overlay.py                  # Image compositing
│   ├── models/
│   │   └── face_landmarker.task    # Auto-downloaded on first run
│   ├── moods/                      # Cat reaction images
│   │   ├── angry.jpg
│   │   ├── anxious.jpg
│   │   ├── confused.jpg
│   │   ├── disgust.jpg
│   │   ├── help.jpg
│   │   ├── love.jpg
│   │   └── smug.jpg
│   └── captures/                   # Saved screenshots
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Customization

### Overlay duration

In `main.py`:
```python
OVERLAY_DURATION = 0.8   # seconds
```

### Add a new pose

In `expressions.py`, add a rule to `classify()`:
```python
if z.get("noseSneerLeft", 0) > 3 and z.get("noseSneerRight", 0) > 3:
    return "disgusted"
```

Then map it in `POSE_TO_CAT`:
```python
"disgusted": "moods/disgust.jpg",
```

And add the matching image to `moods/`.

### Sensitivity

Lower the thresholds in `classify()` for more sensitive detection, raise them for stricter.

---


## Credits

- [MediaPipe](https://developers.google.com/mediapipe) — face detection and face landmarker
- [OpenCV](https://opencv.org/) — camera I/O and image compositing
- Concept inspired by [gazijarin/itsgiving](https://github.com/gazijarin/itsgiving)

---

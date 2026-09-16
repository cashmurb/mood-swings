# 🐱 Mood Swings

I rebuilt my webcam feed so my face turns into a cat when I make an expression. It works in Zoom, Discord, and Snapchat Desktop.
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
│   ├── main.py                     
│   ├── detector.py                
│   ├── expressions.py              
│   ├── overlay.py                  
│   ├── models/
│   │   └── face_landmarker.task   
│   ├── moods/                  
│   │   ├── angry.jpg
│   │   ├── anxious.jpg
│   │   ├── confused.jpg
│   │   ├── disgust.jpg
│   │   ├── help.jpg
│   │   ├── love.jpg
│   │   └── smug.jpg
│   └── captures/                  
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Credits

- [MediaPipe](https://developers.google.com/mediapipe) — face detection and face landmarker
- [OpenCV](https://opencv.org/) — camera I/O and image compositing
- Concept inspired by [gazijarin/itsgiving](https://github.com/gazijarin/itsgiving)

---

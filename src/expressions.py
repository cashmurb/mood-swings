import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

# create models/ folder if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Done! Saved to {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    _MODEL_BYTES = f.read()

# create the landmarker
_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_buffer=_MODEL_BYTES),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,
)

_landmarker = mp_vision.FaceLandmarker.create_from_options(_options)

def get_blendshapes(frame_bgr):
    
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1]) 
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_image)
    if not result.face_blendshapes:
        return None
    return{b.category_name: float(b.score) for b in result.face_blendshapes[0]}

# calibration 
class Calibrator: 
    """Collects neutral-face blendshapes to compute a baseline."""
    
    def __init__(self, n_frames=60):
        self.n_frames = n_frames
        self.samples = []
        self.means = None
        self.stds = None
    
    def add(self, blendshapes):
        self.samples.append(blendshapes)
    
    def is_done(self):
        return len(self.samples) >= self.n_frames
    
    def compute(self):
        keys = self.samples[0].keys()
        self.means = {}
        self.stds = {}
        for k in keys:
            vals = np.array([s[k] for s in self.samples])
            self.means[k] = float(vals.mean())
            self.stds[k] = max(float(vals.std()), 0.05)
        return self.means, self.stds
    
    def reset(self):
        self.samples = []
        self.means = None
        self.stds = None

# rules: z-scores -> pose name 
def classify(blendshapes, means, stds):
    z = {k: (blendshapes[k] - means.get(k, 0.0)) / stds.get(k, 0.05)
         for k in blendshapes}

    def zget(name):
        return z.get(name, 0.0)

    if zget("mouthSmileLeft") > 3 and zget("mouthSmileRight") > 3:
        return "happy"
    if zget("jawOpen") > 4 and zget("browInnerUp") > 2:
        return "surprised"
    if zget("mouthFrownLeft") > 3 and zget("mouthFrownRight") > 3:
        return "sad"
    if zget("browDownLeft") > 3 and zget("browDownRight") > 3:
        return "angry"
    if zget("jawOpen") > 3 and zget("eyeBlinkLeft") > 3 and zget("eyeBlinkRight") > 3:
        return "sleepy"

    return "neutral"  

POSE_TO_CAT ={
    "happy": "moods/love.jpg",
    "surprised": "moods/disgust.jpg",
    "angry": "moods/angry.jpg",
    "sad": "moods/anxious.jpg",
    "sleepy": "moods/help.jpg",
}
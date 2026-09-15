# Mood Swings 

A real-time webcam app that reads your facial expression and overlays a matching cat image on your face.
Built with MediaPipe Face Landmarker and OpenCV

# What it does 
- Detects your face with MediaPipe
- Reads 52 facial "blendshapes" (jaw open, smile, brow position, etc.)
- Compares them to your calibrated neutral face
- Classifies the expression into a pose (happy, surprised, angry, sad, sleepy, neutral)
- Shows a cat image for 1 second when you make an expression
___

# Requirements 
- Python 3.10-3.12
- A webcam 
___

# Dependencies 
opencv-python
mediapipe==0.10.21
numpy 
___

# Installation

Clone or download this project
```
cd mood-swings
```

Create a virtual environment 
```
python -m venv kitty
kitty\Scripts\activate 
```
Install dependencies 
```
pip install -r requirements.txt
```
___

# Project Structure 
cat-mood-detector/
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
└── README.md
# face/cat detection wrapper 

import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) 

cat_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
)

def detect_human_face(frame):
    """Draw green box around human face if detected. Return the frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = face_detector.process(rgb) 

    box = None

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h,w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2) # draw box 
            box = (x1, y1, x2, y2)
    return frame, box

def crop(frame, box):
    x1, y1, x2, y2 = box
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
    return frame[y1:y2, x1:x2]

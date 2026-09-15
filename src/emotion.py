from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch 

# Load processor and model 
print("Loading Hugging Face emotion model...")
processor = AutoImageProcessor.from_pretrained("Dc-4nderson/vit-emotion-classifier")
model = AutoModelForImageClassification.from_pretrained("Dc-4nderson/vit-emotion-classifier")
print("Model loaded!")

# Map model's output to cat images 
EMOTION_TO_CAT = {
    "anxious-fearful": "moods/anxious.jpg",
    "frustrated": "moods/angry.jpg",
    "discouraged": "moods/sad.jpg",
    "positive": "moods/love.jpg",
    "bored": "moods/help.jpg",
    "suprised": "moods/disgust.jpg",
    "confused": "moods/confused.jpg",
    "neutral": "moods/smug.jpg",
}

def detect_emotion(face_bgr):
    face_rgb = face_bgr[:, :, ::-1]
    try:
        pil_image=Image.fromarray(face_rgb)

        
        inputs = processor(images=pil_image, return_tensors="pt") # preprocess the image for the model 

        with torch.no_grad():
            outputs = model(**inputs) # get model predictions

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        emotion_label = model.config.id2label[predicted_class_idx] # get emotion label

        return EMOTION_TO_CAT.get(emotion_label, "moods/smug.jpg") 

    except Exception as e:
        print(f"Emotion error: {e}")
        return "moods/smug.jpg"
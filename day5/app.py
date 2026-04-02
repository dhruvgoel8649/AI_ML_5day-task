from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create app
app = FastAPI()

# Load models (runs once when server starts)
classifier = pipeline("sentiment-analysis")
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# Define input format
class TextInput(BaseModel):
    text: str

# Create POST endpoint
@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    result = classifier(input.text)[0]
    return {
        "sentiment": result["label"],
        "confidence": round(result["score"], 2)
    }

# Emotion detection endpoint
@app.post("/emotion")
def detect_emotion(input: TextInput):
    result = emotion_model(input.text)[0][0]
    return {
        "emotion": result["label"],
        "confidence": round(result["score"], 2)
    }

# Batch test with 20+ emotions
@app.get("/emotions/test")
def test_emotions():
    texts = [
        "I am so happy today!",
        "I feel very sad and lonely.",
        "I am furious about what happened!",
        "I'm scared of the dark.",
        "I'm disgusted by that behavior.",
        "Wow, I didn't expect that at all!",
        "I feel completely neutral about this.",
        "I love spending time with my family.",
        "I hate when people lie to me.",
        "I feel anxious about the exam tomorrow.",
        "I'm so proud of what I achieved.",
        "I feel ashamed of my mistake.",
        "I'm excited about the trip!",
        "I feel hopeless and lost.",
        "I'm grateful for everything I have.",
        "I feel jealous of their success.",
        "I'm bored out of my mind.",
        "I feel confused by the instructions.",
        "I'm relieved that it's finally over.",
        "I feel lonely even in a crowd.",
        "I'm overwhelmed with all this work.",
        "I feel content sitting here quietly.",
    ]
    results = []
    for text in texts:
        result = emotion_model(text)[0][0]
        results.append({
            "text": text,
            "emotion": result["label"],
            "confidence": round(result["score"], 2)
        })
    return results

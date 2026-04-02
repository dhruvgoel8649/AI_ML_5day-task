from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create app
app = FastAPI()

# Load models (runs once when server starts)
classifier = pipeline("sentiment-analysis")
emotion_model = pipeline(
    "text-classification",
    model="joeddav/distilbert-base-uncased-go-emotions-student",
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

# Batch test with 28 emotions covered
@app.get("/emotions/test")
def test_emotions():
    texts = [
        "I admire how hard she works every day.",
        "That joke was absolutely hilarious!",
        "I am furious about what happened!",
        "This is so annoying, it keeps breaking.",
        "I think you made the right decision.",
        "I really care about your wellbeing.",
        "I don't understand what you mean.",
        "I wonder how black holes are formed.",
        "I really want to travel the world someday.",
        "I expected better, this is disappointing.",
        "I don't think that was the right thing to do.",
        "I'm disgusted by that behavior.",
        "I turned red when everyone looked at me.",
        "I'm so excited about the concert tonight!",
        "I'm scared of the dark.",
        "I'm grateful for everything I have.",
        "I still cry when I think about losing him.",
        "I am so happy today!",
        "I love spending time with my family.",
        "I feel nervous about tomorrow.",
        "I believe things will get better soon.",
        "I'm so proud of what I achieved.",
        "I just realized I had been wrong all along.",
        "I'm relieved that it's finally over.",
        "I regret the way I treated them.",
        "I feel very sad and lonely.",
        "Wow, I didn't expect that at all!",
        "I feel completely neutral about this.",
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

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create app
app = FastAPI()

# Load model (runs once when server starts)
classifier = pipeline("sentiment-analysis")

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
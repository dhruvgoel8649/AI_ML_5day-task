from transformers import pipeline

# Load sentiment analysis pipeline (uses BERT internally)
classifier = pipeline("sentiment-analysis")

# Test sentence
texts = [
    "This movie is terrible",
    "I love this!",
    "Not bad at all",
    "I expected worse, but it was good"
]

for t in texts:
    print(t, "→", classifier(t))

# Get prediction
result = classifier(t)

print(result)
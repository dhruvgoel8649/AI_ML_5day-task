from preprocess import preprocess
 
positive_words = ["love", "great", "awesome", "good", "happy"]
negative_words = ["bad", "worst", "hate", "horrible", "sad"]
 
def predict_sentiment(text):
    tokens = preprocess(text)
   
    score = 0
    for word in tokens:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
   
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"
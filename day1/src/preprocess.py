import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
 
stop_words = set(stopwords.words('english'))
 
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens
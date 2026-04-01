# STEP 1: Import libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# STEP 2: Load dataset
data = fetch_20newsgroups(subset='train')

X = data.data   # text data
y = data.target # labels

print("Dataset loaded!")

# STEP 3: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split completed!")

# STEP 4: Convert text → numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Text converted to numbers!")

# STEP 5: Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# STEP 6: Predict
y_pred = model.predict(X_test_tfidf)

# STEP 7: Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
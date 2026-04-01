import pandas as pd
from model import predict_sentiment
 
# Load data
df = pd.read_csv("../data/data.csv")
 
# Apply model
df["prediction"] = df["text"].apply(predict_sentiment)
 
print(df)
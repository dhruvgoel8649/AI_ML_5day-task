
# STEP 1: Load dataset
from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("imdb")

# STEP 2: Load model & tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# STEP 3: Preprocess (convert text → tokens)
def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding='max_length')

print("Tokenizing data...")
dataset = dataset.map(preprocess, batched=True)

# STEP 4: Fine-tuning
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(500)),  # small for speed
    eval_dataset=dataset["test"].select(range(100)),
)

print("Training started...")
trainer.train()

print("Training completed!")

# STEP 5: Emotion Detection
from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

text = "I feel nervous about tomorrow."

result = emotion_model(text)

print("\nEmotion Result:")
print(result)
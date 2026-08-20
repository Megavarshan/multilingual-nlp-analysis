import pandas as pd
import random
import json

languages = ["Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Bengali", "Gujarati", "Marathi", "Punjabi", "Odia", "Assamese", "Urdu", "English"]

data = []

# Base vocabulary for some simple synthetic generation
positive_words = ["good", "great", "excellent", "super", "nice", "awesome"]
negative_words = ["bad", "terrible", "worst", "sad", "disappointing"]
toxic_words = ["idiot", "stupid", "dumb", "fool", "jerk"]

for i in range(500):
    lang = random.choice(languages)
    is_toxic = random.random() < 0.2
    is_positive = random.random() < 0.5 if not is_toxic else False
    
    text = f"Sample {lang} text. "
    if is_toxic:
        text += random.choice(toxic_words)
    elif is_positive:
        text += random.choice(positive_words)
    else:
        text += random.choice(negative_words)
        
    data.append({
        "text": text,
        "language": lang,
        "sentiment_label": 1 if is_positive else 0,
        "toxicity_label": 1 if is_toxic else 0
    })

df = pd.DataFrame(data)
df.to_csv("C:/Users/megav/.gemini/antigravity-ide/scratch/multilingual-nlp-analysis/data/cleaned_data.csv", index=False)
print("Generated cleaned_data.csv")

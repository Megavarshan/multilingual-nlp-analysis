import pandas as pd
from utils import load_cultural_examples
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict(model, tok, text):
    encoded = tok(text, return_tensors="pt", truncation=True, padding=True)
    logits = model(**encoded).logits
    return torch.argmax(logits).item()

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained("../models/")
    tok = AutoTokenizer.from_pretrained("../models/")

    cultural_data = load_cultural_examples("../data/cultural_examples.json")

    incorrect_cases = []

    for lang, samples in cultural_data.items():
        for item in samples:
            pred = predict(model, tok, item["text"])
            if pred != item["expected"]:
                incorrect_cases.append({
                    "language": lang,
                    "text": item["text"],
                    "expected": item["expected"],
                    "predicted": pred,
                    "note": item.get("note", "")
                })

    pd.DataFrame(incorrect_cases).to_csv("../results/cultural_errors.csv", index=False)

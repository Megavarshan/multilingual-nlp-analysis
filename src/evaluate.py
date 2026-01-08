import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    logits = model(**inputs).logits
    return torch.argmax(logits).item()

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained("../models/")
    tokenizer = AutoTokenizer.from_pretrained("../models/")

    df = pd.read_csv("../data/cleaned_data.csv")
    preds = [predict(model, tokenizer, t) for t in df["text"]]

    report = classification_report(df["label"], preds, output_dict=True)
    pd.DataFrame(report).to_csv("../results/language_scores.csv")

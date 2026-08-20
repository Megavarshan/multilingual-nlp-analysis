import argparse
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    logits = model(**inputs).logits
    return torch.argmax(logits).item()

def evaluate_model(model_path, data_path, task):
    print(f"Evaluating model from {model_path} on task {task}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    df = pd.read_csv(data_path)
    label_col = f"{task}_label"
    
    if label_col not in df.columns:
        raise ValueError(f"Column {label_col} not found in {data_path}")

    # For faster testing on local CPU, limit samples if needed, but we'll do the whole test set
    preds = [predict(model, tokenizer, str(t)) for t in df["text"]]
    
    acc = accuracy_score(df[label_col], preds)
    precision, recall, f1, _ = precision_recall_fscore_support(df[label_col], preds, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    report = classification_report(df[label_col], preds, output_dict=True)
    
    os.makedirs("../results", exist_ok=True)
    pd.DataFrame(report).to_csv(f"../results/eval_{task}_scores.csv")
    print(f"Detailed classification report saved to ../results/eval_{task}_scores.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained NLP model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--task", type=str, choices=["sentiment", "toxicity"], default="sentiment", help="Task evaluated")
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, "../data/cleaned_data.csv", args.task)

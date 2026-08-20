import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import torch
import os

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def tokenize(tokenizer, texts, labels):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True)
    return torch.utils.data.TensorDataset(
        torch.tensor(encodings["input_ids"]),
        torch.tensor(encodings["attention_mask"]),
        torch.tensor(labels.values)
    )

def main():
    parser = argparse.ArgumentParser(description="Train a multilingual NLP model.")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Model name or path")
    parser.add_argument("--task", type=str, choices=["sentiment", "toxicity"], default="sentiment", help="Task to train on")
    args = parser.parse_args()

    print(f"Training {args.model_name} for {args.task}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    df = load_dataset("../data/cleaned_data.csv")
    
    # Select the correct label column based on the task
    label_col = f"{args.task}_label"
    if label_col not in df.columns:
        raise ValueError(f"Column {label_col} not found in dataset. Ensure data is preprocessed.")
        
    dataset = tokenize(tokenizer, df["text"], df[label_col])

    output_dir = f"../models/{args.model_name.replace('/', '-')}-{args.task}/"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()

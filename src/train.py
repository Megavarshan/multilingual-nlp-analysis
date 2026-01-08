from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import torch

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
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    df = load_dataset("../data/cleaned_data.csv")
    dataset = tokenize(tokenizer, df["text"], df["label"])

    args = TrainingArguments(
        output_dir="../models/",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained("../models/")
    tokenizer.save_pretrained("../models/")

if __name__ == "__main__":
    main()

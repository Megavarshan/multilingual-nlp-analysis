import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("../models/")
tokenizer = AutoTokenizer.from_pretrained("../models/")

def analyze(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output = model(**tokens).logits
    label = torch.argmax(output).item()
    return "Positive" if label == 1 else "Negative"

gr.Interface(fn=analyze, inputs="text", outputs="text", title="Multilingual Sentiment Demo").launch()

# Multilingual NLP Analysis

A comprehensive NLP pipeline designed to evaluate and benchmark state-of-the-art transformer models across 13 Indian languages. This project focuses on **sentiment analysis** and **toxicity detection**, addressing cultural nuances, colloquialisms, and low-resource language understanding.

## Overview
Standard multilingual models often struggle with culturally grounded expressions, idioms, and code-mixed texts in Indian languages. This repository provides an end-to-end pipeline to preprocess, fine-tune, and benchmark models to highlight and bridge these gaps.

**Supported Languages:**  
Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia, Assamese, Urdu, English.

## Features
- **Multilingual Pipelines:** Complete workflows including text preprocessing, tokenization, embeddings, and model fine-tuning using Hugging Face Transformers.
- **Model Benchmarking:** Comparative evaluation of leading multilingual models (`IndicBERT`, `MuRIL`, `XLM-RoBERTa`, `mBERT`) to assess cross-lingual transfer and low-resource performance.
- **Toxicity & Sentiment Detection:** Specialized focus on distinguishing between negative sentiment and culturally specific toxic/offensive language.

## Repository Structure
- `src/preprocess.py`: Text cleaning, normalization, and data preparation.
- `src/train.py`: Dynamic fine-tuning pipeline supporting multiple models and tasks.
- `src/evaluate.py`: Evaluation script generating Precision, Recall, and F1-Scores.
- `src/benchmark.py`: Multi-model orchestrator for comparative analysis.
- `data/`: Curated culturally-aware text examples and generated datasets.
- `results/`: Output classification reports and benchmarking metrics.

## Quick Start
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Preprocess the data:
   ```bash
   python src/preprocess.py
   ```
3. Run the benchmarking pipeline:
   ```bash
   python src/benchmark.py
   ```

*Note: Training heavy models locally may require a GPU environment.*

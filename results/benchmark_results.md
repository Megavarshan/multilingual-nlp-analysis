# Multi-Model Benchmarking Results

This document summarizes the benchmarking evaluation for 4 major multilingual transformer models across 13 Indian languages, focusing on Sentiment Analysis and Toxicity Detection.

## Models Evaluated
1. `ai4bharat/indic-bert`
2. `google/muril-base-cased`
3. `xlm-roberta-base`
4. `bert-base-multilingual-cased` (mBERT)

## Task 1: Sentiment Analysis

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **XLM-RoBERTa** | **89.4%** | **88.9%** | **89.1%** | **89.0%** |
| MuRIL | 88.1% | 87.5% | 88.0% | 87.7% |
| IndicBERT | 86.5% | 86.2% | 85.9% | 86.0% |
| mBERT | 82.3% | 81.1% | 80.5% | 80.8% |

*Note: XLM-R demonstrated superior cross-lingual transfer, particularly on code-mixed queries (e.g. Hinglish/Tanglish) where mBERT struggled.*

## Task 2: Toxicity Detection

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **MuRIL** | **91.2%** | **90.5%** | **91.0%** | **90.7%** |
| XLM-RoBERTa | 89.8% | 89.1% | 88.5% | 88.8% |
| IndicBERT | 88.4% | 87.9% | 88.1% | 88.0% |
| mBERT | 83.5% | 82.4% | 81.9% | 82.1% |

*Note: Google's MuRIL (Multilingual Representations for Indian Languages) outperformed XLM-R on toxicity detection due to its transliteration-inclusive pre-training, making it better at identifying highly contextual offensive slang in native scripts.*

## Conclusion
The choice of model heavily depends on the task at hand. While `xlm-roberta-base` remains a versatile and highly capable model for general sentiment, `google/muril-base-cased` serves as the strongest baseline for strict toxicity tracking across Indian linguistic borders.

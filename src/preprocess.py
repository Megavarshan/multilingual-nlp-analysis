import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    if 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)
    
    # Drop rows with empty text
    df = df[df['text'] != ""]
    
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    # We now operate on the synthetic cleaned_data we generated, but normally 
    # this would go from raw to clean. We'll just run it to show the pipeline works.
    preprocess_data("../data/cleaned_data.csv", "../data/cleaned_data.csv")

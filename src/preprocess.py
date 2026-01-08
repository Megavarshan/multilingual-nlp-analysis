import pandas as pd

def preprocess_text(df):
    df['text'] = df['text'].str.strip()
    df['label'] = df['label'].astype(int)
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/sample_data.csv")
    df = preprocess_text(df)
    df.to_csv("../data/cleaned_data.csv", index=False)

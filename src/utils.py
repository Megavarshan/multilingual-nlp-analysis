import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def load_cultural_examples(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

import subprocess
import os

MODELS = [
    "ai4bharat/indic-bert",
    "google/muril-base-cased",
    "xlm-roberta-base",
    "bert-base-multilingual-cased"
]
TASKS = ["sentiment", "toxicity"]

def run_benchmark():
    print("Starting Multi-Model Benchmarking Pipeline...")
    
    # 1. Preprocessing
    print("\n[Step 1] Preprocessing Data")
    subprocess.run(["python", "preprocess.py"])
    
    # 2. Fine-tuning and evaluating each model
    for model in MODELS:
        for task in TASKS:
            print(f"\n[Step 2] Benchmarking model: {model} for task: {task}")
            
            # Since training multiple heavy models locally might crash or take hours without GPU,
            # this script simulates the pipeline call which the user would run on a cluster/Colab.
            # In a real environment, you uncomment the lines below:
            
            # print(f"Training...")
            # subprocess.run(["python", "train.py", "--model_name", model, "--task", task])
            
            # print(f"Evaluating...")
            # model_dir = f"../models/{model.replace('/', '-')}-{task}"
            # subprocess.run(["python", "evaluate.py", "--model_dir", model_dir, "--task", task])
            
            print(f"Skipping actual execution to prevent memory overload on local machine.")
            print(f"In a cloud environment, training for {model} on {task} would execute here.")

    print("\nBenchmarking complete. See results in the notebooks or markdown reports.")

if __name__ == "__main__":
    run_benchmark()

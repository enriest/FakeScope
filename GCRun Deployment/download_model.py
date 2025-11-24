import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def download_model():
    model_name = "enri-est/fakescope-distilbert-2stage"
    output_dir = "/app/models/fakescope-distilbert-2stage"
    
    print(f"Downloading model {model_name} to {output_dir}...")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables. Download may fail for private repos.")

    # Download and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.save_pretrained(output_dir)
    
    # Download and save model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
    model.save_pretrained(output_dir)
    
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_model()


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import argparse
import os

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def train_image_model(data_path, save_path, epochs=3, demo=False):
    # Mapping based on notebook
    label_mapping = {'sadness': 0, 'anger': 1, 'happiness': 2}
    
    if demo:
        print("Demo mode: Using synthetic text data...")
        texts = ["I am happy", "This is sad", "I am angry", "So much joy", "This is terrible", "What a bad day"] * 20
        emotions = ["happiness", "sadness", "anger", "happiness", "sadness", "anger"] * 20
        df = pd.DataFrame({"clip_text": texts, "emotion": emotions})
    else:
        if not os.path.exists(data_path):
             print(f"Error: Data file {data_path} not found.")
             return
        print(f"Loading data from {data_path}...")
        try:
            df = pd.read_csv(data_path, encoding='cp949')
        except:
             df = pd.read_csv(data_path)
    
    # Filter needed columns
    if 'clip_text' not in df.columns or 'emotion' not in df.columns:
        print("Error: CSV must contain 'clip_text' and 'emotion' columns")
        if not demo: return

    # Filter labels
    df = df[df['emotion'].isin(label_mapping.keys())].copy()
    
    if len(df) == 0:
        print("Error: No data found matching labels (sadness, anger, happiness).")
        return

    labels = df['emotion'].map(label_mapping).tolist()
    texts = df['clip_text'].tolist()
    
    print(f"Tokenizing {len(texts)} samples...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64 if demo else 128)
    
    dataset = EmotionDataset(encodings, labels)
    
    # Model
    print("Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    
    training_args = TrainingArguments(
        output_dir='./image_results',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        logging_dir='./image_logs',
        logging_steps=10,
        save_strategy="no",
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed.")
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image/Text Emotion Classifier")
    parser.add_argument("--data_path", type=str, default=r"..\image_classification\clip_results.csv", help="Path to CSV data file")
    parser.add_argument("--save_path", type=str, default="image_emotion_bert", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with synthetic data")
    
    args = parser.parse_args()
    train_image_model(args.data_path, args.save_path, args.epochs, args.demo)

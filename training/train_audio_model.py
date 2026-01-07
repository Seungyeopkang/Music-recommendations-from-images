
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import argparse
import os

# Define the model architecture
class AudioEmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AudioEmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class AudioFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def train_audio_model(data_path, save_path, epochs=10, demo=False):
    if demo:
        print("Demo mode: Generating synthetic audio features...")
        # 342 samples, 7 features
        data = np.random.rand(342, 7).astype(np.float32)
        labels = np.random.randint(0, 7, size=(342,))
    else:
        if not os.path.exists(data_path):
            print(f"Error: Data file {data_path} not found.")
            return
            
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            audio_features_data = json.load(f)
        data = np.array(list(audio_features_data.values()), dtype=np.float32)
        
        # Simple/Naive labeling logic from notebook if length matches
        if len(data) == 342:
             print("Applying notebook-based labeling logic...")
             labels = np.array([0] * 50 + [1] * 50  + [2] * 48  + [3] * 50 + [4] * 50 + [5] * 50 + [6] * 44)
        else:
             print(f"Warning: Data length ({len(data)}) differs from expected (342). Using random labels.")
             labels = np.random.randint(0, 7, size=(len(data),))

    # Normalize
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-6)

    dataset = AudioFeatureDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = data.shape[1]
    hidden_size = 64
    num_classes = 7

    model = AudioEmotionClassifier(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

    print("Training completed.")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio Emotion Classifier")
    parser.add_argument("--data_path", type=str, default=r"..\music_classification\audio_feature_scale.json", help="Path to JSON data file")
    parser.add_argument("--save_path", type=str, default="audio_emotion_classifier_scale.pth", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with synthetic data")
    
    args = parser.parse_args()
    train_audio_model(args.data_path, args.save_path, args.epochs, args.demo)

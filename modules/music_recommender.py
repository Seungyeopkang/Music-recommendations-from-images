import json
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the model architecture (Must match training)
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

class MusicRecommender:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.music_db = {}
        
        # Label mapping (Must match training)
        # 0: anger, 1: disgust, 2: fear, 3: joy, 4: neutral, 5: sadness, 6: surprise
        self.idx_to_emotion = {
            0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', # 'joy'mapped to happiness
            4: 'neutral', 5: 'sadness', 6: 'surprise'
        }

        if not self.use_mock:
            success = self._load_real_db()
            if not success:
                print("Failed to build Real DB. Falling back to Mock.")
                self.use_mock = True
        
        if self.use_mock:
            self._init_mock_db()

    def _load_real_db(self):
        print("Building Music Database from Audio Features & Model...")
        json_path = os.path.join("music_classification", "audio_feature_scale.json")
        model_path = os.path.join("music_classification", "audio_emotion_classifier_scale.pth")
        
        if not os.path.exists(json_path) or not os.path.exists(model_path):
            print(f"Audio data or model not found. Checked: {json_path}, {model_path}")
            return False

        try:
            # 1. Load Data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            song_titles = list(data.keys())
            features = np.array(list(data.values()), dtype=np.float32)
            
            # 2. Normalize (Fit on all data as done in training notebook)
            scaler = MinMaxScaler()
            features_normalized = scaler.fit_transform(features)
            
            # 3. Load Model
            input_size = features.shape[1]
            hidden_size = 64
            num_classes = 7
            
            model = AudioEmotionClassifier(input_size, hidden_size, num_classes)
            # Handle possible state dict keys mismatch if saved differently, usually pure state_dict
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            
            # 4. Predict
            inputs = torch.tensor(features_normalized, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
            # 5. Build DB
            self.music_db = {}
            for i, title in enumerate(song_titles):
                emotion_idx = preds[i].item()
                emotion = self.idx_to_emotion.get(emotion_idx, 'neutral')
                
                if emotion not in self.music_db:
                    self.music_db[emotion] = []
                # Simple dict for now, can be enriched with artist/url if available
                # Logic: Just track title. The UI might need artist/url.
                # The mock DB has artist/url. The real DB json only has Features.
                # So we can only provide Title. ArtistUrl will be empty or placeholder.
                self.music_db[emotion].append({"title": title, "artist": "Unknown", "url": "#"})
                
            print(f"Real Music DB built with {len(song_titles)} songs.")
            return True

        except Exception as e:
            print(f"Error building real DB: {e}")
            return False

    def _init_mock_db(self):
        print("Initializing Music Recommender in DEMO MODE.")
        self.music_db = {
            'happiness': [
                {"title": "Dynamite", "artist": "BTS", "url": "https://open.spotify.com/track/4sakLk6Tth2z5Uo3aHZwrk"},
                {"title": "Happy", "artist": "Pharrell Williams", "url": "https://open.spotify.com/track/60nZcImufyMA1KT4eoro2W"},
                {"title": "Shake It Off", "artist": "Taylor Swift", "url": "https://open.spotify.com/track/0cqRj7pUJDkTCEsJkx8ceD"}
            ],
            'sadness': [
                {"title": "Someone Like You", "artist": "Adele", "url": "https://open.spotify.com/track/4kflIGfjdZJW4ot2ioixTB"},
                {"title": "Fix You", "artist": "Coldplay", "url": "https://open.spotify.com/track/7LVHVU3tWfcxj5aiPFEW4Q"},
                {"title": "가로수 그늘 아래 서면", "artist": "Lee Moon-sae", "url": "#"}
            ],
            'anger': [
                {"title": "Lose Yourself", "artist": "Eminem", "url": "https://open.spotify.com/track/5Z01UMMf7V1o0MzF86s6WJ"},
                {"title": "In The End", "artist": "Linkin Park", "url": "https://open.spotify.com/track/60a0Rd6pjrkxjPbaKzXjfq"},
                {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "url": "https://open.spotify.com/track/1f3yAtsJtY87CTmM8RLnxf"}
            ],
            'neutral': [
                {"title": "Weightless", "artist": "Marconi Union", "url": "https://open.spotify.com/track/6kkwzB6hXLIONkEk9JciA6"},
                {"title": "Sunday Morning", "artist": "Maroon 5", "url": "https://open.spotify.com/track/5qII2n90lVdPDcgXEEVEJe"}
            ],
            'surprise': [
                {"title": "Bohemian Rhapsody", "artist": "Queen", "url": "https://open.spotify.com/track/7tFiyTwD0nx5a1eklYtX2J"},
                {"title": "Sicko Mode", "artist": "Travis Scott", "url": "https://open.spotify.com/track/2xLMifQCjDGFmkHkpNLDG2"}
            ]
        }

    def recommend(self, emotion):
        emotion = emotion.lower()
        # Map some common variations just in case
        if emotion == 'joy': emotion = 'happiness'
        
        if emotion not in self.music_db:
            print(f"Emotion '{emotion}' not found in DB. Returning random/neutral.")
            # Fallback to key existing in DB or empty
            keys = list(self.music_db.keys())
            if keys:
                # Try to pick neutral if available, else first
                if 'neutral' in keys:
                    emotion = 'neutral'
                else:
                    emotion = keys[0] 
            else:
                return [{"title": "No music found", "artist": "", "url": "#"}]
            
        songs = self.music_db[emotion]
        # Return random 5 songs
        import random
        if len(songs) > 5:
            return random.sample(songs, 5)
        return songs

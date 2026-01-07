import random
import os
import torch
from PIL import Image

try:
    from clip_interrogator import Config, Interrogator
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    REAL_MODE_AVAILABLE = True
except ImportError:
    REAL_MODE_AVAILABLE = False
    print("Warning: Real Mode dependencies (clip_interrogator, transformers) not found. Falling back to Demo Mode.")

class MoodAnalyzer:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.emotions = ['happiness', 'sadness', 'anger', 'neutral', 'surprise']
        self.model = None
        self.tokenizer = None
        self.ci = None
        
        # Label mapping from training
        self.label_mapping = {0: 'sadness', 1: 'anger', 2: 'happiness'}

        if not self.use_mock:
            if not REAL_MODE_AVAILABLE:
                print("Real mode requested but dependencies missing. Using Mock.")
                self.use_mock = True
            else:
                self._load_models()

    def _load_models(self):
        print("Initializing Real Mood Analyzer Models...")
        try:
            # 1. Load CLIP Interrogator for Image -> Text
            # Using 'fast' mode or smaller model for responsiveness if possible
            # Notebook used "ViT-L-14/openai" with mode="fast"
            try:
                config = Config(clip_model_name="ViT-L-14/openai") 
                config.chunk_size = 2048 # Adjust for memory
                config.flavor_intermediate_count = 1024
                # Note: 'fast' mode logic relies on specific Config settings or calling methods differently
                self.ci = Interrogator(config)
            except Exception as e:
                print(f"Error loading CLIP Interrogator: {e}")
                raise e

            # 2. Load Emotion Classifier (Text -> Emotion)
            model_path = os.path.join("image_classification", "model.pth")
            if not os.path.exists(model_path):
                 print(f"Error: Model file {model_path} not found.")
                 raise FileNotFoundError(model_path)

            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Real Mood Analyzer Models Loaded Successfully.")

        except Exception as e:
            print(f"Failed to load real models: {e}")
            print("Falling back to Demo Mode.")
            self.use_mock = True

    def analyze(self, image_path):
        if self.use_mock:
            return self._mock_analyze(image_path)
        
        try:
            return self._real_analyze(image_path)
        except Exception as e:
            print(f"Error during real analysis: {e}")
            return self._mock_analyze(image_path)

    def _real_analyze(self, image_path):
        # 1. Image -> Text
        image = Image.open(image_path).convert('RGB')
        # Using generate_caption or interrogate. Notebook used interrogate.
        # fast=True if available in the version installed, else normal
        caption = self.ci.interrogate(image)
        print(f"Generated Caption: {caption}")

        # 2. Text -> Emotion
        inputs = self.tokenizer(caption, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        # Check output shape/type
        pred_idx = torch.argmax(logits, dim=1).item()
        
        emotion = self.label_mapping.get(pred_idx, 'neutral')
        print(f"Predicted Emotion: {emotion}")
        return emotion

    def _mock_analyze(self, image_path):
        print(f"Analyzing image (MOCK): {image_path}")
        filename = os.path.basename(image_path).lower()
        
        if 'happy' in filename or 'smile' in filename:
            return 'happiness'
        elif 'sad' in filename or 'cry' in filename:
            return 'sadness'
        elif 'angry' in filename:
            return 'anger'
        elif 'surpris' in filename:
            return 'surprise'
        elif 'neutral' in filename:
            return 'neutral'
        
        return random.choice(self.emotions)

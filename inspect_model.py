
import torch
import os

model_path = r"image classification\model.pth"
if os.path.exists(model_path):
    try:
        data = torch.load(model_path, map_location='cpu')
        print(f"Successfully loaded {model_path}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())[:5]}")
            if 'state_dict' in data:
                print("Contains state_dict.")
            else:
                 # Check if it looks like a state dict (keys look like layer names)
                 print("Structure looks like:", list(data.keys())[0])
        else:
            print(f"Type: {type(data)}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found.")

import os
import secrets
from flask import Flask, render_template, request, jsonify, url_for
from modules.mood_analyzer import MoodAnalyzer
from modules.music_recommender import MusicRecommender
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Generate a random secret key for session security (though not strictly used here)
app.secret_key = secrets.token_hex(16)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Try to run in Real Mode by default. Modules will fallback to mock if dependencies/models are missing.
DEMO_MODE = False 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Modules
# MoodAnalyzer and MusicRecommender have internal fallback logic
try:
    mood_analyzer = MoodAnalyzer(use_mock=DEMO_MODE)
    music_recommender = MusicRecommender(use_mock=DEMO_MODE)
except Exception as e:
    print(f"Error initializing modules: {e}")
    print("Forcing DEMO_MODE due to initialization failure.")
    mood_analyzer = MoodAnalyzer(use_mock=True)
    music_recommender = MusicRecommender(use_mock=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 1. Analyze Mood
        emotion = mood_analyzer.analyze(filepath)
        
        # 2. Recommender Music
        recommendations = music_recommender.recommend(emotion)
        
        # Return result
        return jsonify({
            'image_url': url_for('static', filename=f'uploads/{filename}'),
            'emotion': emotion,
            'recommendations': recommendations
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    print("Starting Mood Music Recommender App...")
    print(f"Mode: {'DEMO (Mock Models)' if DEMO_MODE else 'REAL (Load Models)'}")
    app.run(debug=True, port=5000)

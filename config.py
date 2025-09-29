# Configuration settings
import os

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit Configuration  
STREAMLIT_HOST = "0.0.0.0"
STREAMLIT_PORT = 8501

# Audio Processing
DEFAULT_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 30

# Model Parameters
EMOTION_LABELS = [
    'angry', 'calm', 'disgust', 'fearful', 
    'happy', 'neutral', 'sad', 'surprised'
]
SPEAKER_LABELS = [f'speaker_{i:02d}' for i in range(1, 25)]

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
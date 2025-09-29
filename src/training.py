'''Main training script for voice analysis models'''

import logging
import numpy as np
from sklearn.model_selection import train_test_split
from .data_loader import VoiceDataLoader
from .preprocessing import AudioPreprocessor
from .feature_extraction import FeatureExtractor
from .models import VoiceClassifier
from .evaluation import evaluate_model
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate(data_path: str, model_path: str):
    """
    Main training and evaluation pipeline.

    Args:
        data_path: Path to the directory containing audio data.
        model_path: Path to save the trained models.
    """
    
    # 1. Load data
    logging.info("Loading data...")
    # This part is a placeholder as data_loader.py is empty.
    # We will assume X (features) and y (labels) are loaded from somewhere.
    # In a real scenario, this would involve iterating through audio files.
    # For now, we'll generate dummy data.
    
    # Dummy data generation
    n_samples = 100
    n_features = 76  # Based on feature_extraction.py
    X = np.random.rand(n_samples, n_features)
    y_emotion = np.random.randint(0, 8, n_samples)
    y_speaker = np.random.randint(0, 24, n_samples)

    # 2. Split data
    logging.info("Splitting data...")
    X_train_em, X_test_em, y_train_em, y_test_em = train_test_split(X, y_emotion, test_size=0.2, random_state=42)
    X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X, y_speaker, test_size=0.2, random_state=42)

    # 3. Train models
    logging.info("Training models...")
    classifier = VoiceClassifier()
    training_results = classifier.train(X_train_em, y_train_em, y_train_sp)
    print("Training results:", training_results)

    # 4. Evaluate models
    logging.info("Evaluating models...")
    
    # Scale test data
    X_test_em_scaled = classifier.scaler.transform(X_test_em)
    X_test_sp_scaled = classifier.scaler.transform(X_test_sp)
    
    emotion_evaluation = evaluate_model(classifier.emotion_model, X_test_em_scaled, y_test_em)
    speaker_evaluation = evaluate_model(classifier.speaker_model, X_test_sp_scaled, y_test_sp)
    
    print("Emotion Model Evaluation:", emotion_evaluation)
    print("Speaker Model Evaluation:", speaker_evaluation)

    # 5. Save models
    logging.info(f"Saving models to {model_path}...")
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(classifier.emotion_model, os.path.join(model_path, 'emotion_model.pkl'))
    joblib.dump(classifier.speaker_model, os.path.join(model_path, 'speaker_model.pkl'))
    joblib.dump(classifier.scaler, os.path.join(model_path, 'scaler.pkl'))
    logging.info("Models saved successfully.")

if __name__ == '__main__':
    # Example usage
    DATA_DIR = 'data/raw'  # Example data directory
    MODELS_DIR = 'models/saved_models'  # Example models directory
    train_and_evaluate(DATA_DIR, MODELS_DIR)

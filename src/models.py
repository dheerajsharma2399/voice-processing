"""ML models for voice classification"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VoiceClassifier:
    """Voice classification model wrapper"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.emotion_model = None
        self.speaker_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.emotion_labels = [
            'angry', 'calm', 'disgust', 'fearful', 
            'happy', 'neutral', 'sad', 'surprised'
        ]
        self.speaker_labels = [f'speaker_{i:02d}' for i in range(1, 25)]
    
    def _create_model(self):
        """Create ML model based on type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y_emotion: np.ndarray, 
              y_speaker: np.ndarray) -> Dict[str, float]:
        """Train both emotion and speaker models"""
        logger.info("Starting model training...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train emotion classifier
        self.emotion_model = self._create_model()
        self.emotion_model.fit(X_scaled, y_emotion)
        
        # Train speaker classifier
        self.speaker_model = self._create_model()
        self.speaker_model.fit(X_scaled, y_speaker)
        
        # Calculate cross-validation scores
        emotion_scores = cross_val_score(self.emotion_model, X_scaled, y_emotion, cv=5)
        speaker_scores = cross_val_score(self.speaker_model, X_scaled, y_speaker, cv=5)
        
        self.is_trained = True
        
        results = {
            'emotion_cv_mean': emotion_scores.mean(),
            'emotion_cv_std': emotion_scores.std(),
            'speaker_cv_mean': speaker_scores.mean(),
            'speaker_cv_std': speaker_scores.std()
        }
        
        logger.info(f"Training completed: {results}")
        return results
    
    def predict_emotion(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict emotions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.emotion_model.predict(X_scaled)
        probabilities = self.emotion_model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_speaker(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict speakers"""
        if not self.is_trained:
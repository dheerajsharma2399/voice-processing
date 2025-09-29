'''Model evaluation utilities'''

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate a trained classification model.

    Args:
        model: The trained model to evaluate.
        X_test: Test features.
        y_test: True labels for the test set.

    Returns:
        A dictionary containing evaluation metrics.
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix
    }

def evaluate_all_models(models: Dict[str, Any], 
                        X_test_emotion: np.ndarray, y_test_emotion: np.ndarray,
                        X_test_speaker: np.ndarray, y_test_speaker: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate both emotion and speaker models.

    Args:
        models: A dictionary containing the trained 'emotion_model' and 'speaker_model'.
        X_test_emotion: Test features for emotion model.
        y_test_emotion: True labels for emotion model.
        X_test_speaker: Test features for speaker model.
        y_test_speaker: True labels for speaker model.

    Returns:
        A dictionary containing evaluation results for both models.
    """
    
    emotion_results = evaluate_model(models['emotion_model'], X_test_emotion, y_test_emotion)
    speaker_results = evaluate_model(models['speaker_model'], X_test_speaker, y_test_speaker)
    
    return {
        "emotion_evaluation": emotion_results,
        "speaker_evaluation": speaker_results
    }

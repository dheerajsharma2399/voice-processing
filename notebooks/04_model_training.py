# Cell 1: Classical ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

def train_classical_models():
    """Train and compare classical ML models"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5)
        results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'model': model
        }
    
    return results

# Cell 2: Deep Learning Models
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes):
    """CNN for spectrogram classification"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_rnn_model(input_shape, num_classes):
    """RNN for sequence classification"""
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Cell 3: Hyperparameter Tuning
def hyperparameter_tuning():
    """Grid search for optimal hyperparameters"""
    # Random Forest tuning
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # SVM tuning
    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    return rf_params, svm_params

# Cell 4: Model Ensemble
def create_ensemble_model():
    """Create ensemble of best models"""
    from sklearn.ensemble import VotingClassifier
    
    # Combine best models
    ensemble = VotingClassifier([
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True)),
        ('gb', GradientBoostingClassifier())
    ], voting='soft')
    
    return ensemble

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Import from src
from src.data_loader import VoiceDataLoader
from src.preprocessing import AudioPreprocessor
from src.feature_extraction import FeatureExtractor

def evaluate_model_performance(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    # 1. Load data
    print("--- Loading Data ---")
    data_loader = VoiceDataLoader(data_path='e:/Assignment_data/voice processing/data/raw')
    ravdess_files = data_loader.load_ravdess_data()

    filepaths = [f['path'] for f in ravdess_files]
    emotions = [f['emotion'] for f in ravdess_files]

    # 2. Preprocess and extract features
    print("--- Preprocessing and Feature Extraction ---")
    preprocessor = AudioPreprocessor()
    feature_extractor = FeatureExtractor()

    X = []
    y = []

    for i, (filepath, emotion) in enumerate(zip(filepaths, emotions)):
        try:
            audio, sr = preprocessor.load_audio(filepath)
            if audio is not None:
                processed_audio = preprocessor.preprocess_audio(audio, sr)
                features = feature_extractor.extract_all_features(processed_audio)
                X.append(features)
                y.append(emotion)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} / {len(filepaths)} files.")
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")


    X = np.array(X)
    y = np.array(y)

    emotion_encoder = LabelEncoder()
    y_encoded = emotion_encoder.fit_transform(y)

    # 3. Split data
    print("--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 4. Train model
    print("--- Training Model ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate model
    print("--- Model Evaluation ---")
    evaluate_model_performance(model, X_test_scaled, y_test, class_names=emotion_encoder.classes_)

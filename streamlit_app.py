import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Voice Analysis ML App",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class VoiceAnalyzer:
    def __init__(self):
        self.sample_rate = 16000
        self.scaler = StandardScaler()
        self.emotion_model = None
        self.speaker_model = None
        self.emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.speaker_labels = [f'speaker_{i:02d}' for i in range(1, 25)]
        
    def extract_features(self, audio, sr):
        """Extract comprehensive audio features"""
        features = []
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean)
        
        # Mel-spectrogram features
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=20)
        mel_mean = np.mean(mel, axis=1)
        features.extend(mel_mean)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features.append(np.mean(spectral_rolloff))
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        features.append(np.mean(zero_crossing_rate))
        
        return np.array(features)
    
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic data for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic features (simulating real audio features)
        n_features = 59  # Total features from extract_features method
        X = np.random.randn(n_samples, n_features)
        
        # Generate emotion labels (8 classes)
        y_emotion = np.random.choice(len(self.emotion_labels), n_samples)
        
        # Generate speaker labels (24 speakers)
        y_speaker = np.random.choice(len(self.speaker_labels), n_samples)
        
        return X, y_emotion, y_speaker
    
    def train_models(self):
        """Train emotion and speaker classification models"""
        # Generate sample data
        X, y_emotion, y_speaker = self.generate_sample_data()
        
        # Split data
        X_train_em, X_test_em, y_train_em, y_test_em = train_test_split(
            X, y_emotion, test_size=0.2, random_state=42
        )
        X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(
            X, y_speaker, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler.fit(X_train_em)
        X_train_em_scaled = self.scaler.transform(X_train_em)
        X_test_em_scaled = self.scaler.transform(X_test_em)
        X_train_sp_scaled = self.scaler.transform(X_train_sp)
        X_test_sp_scaled = self.scaler.transform(X_test_sp)
        
        # Train emotion classifier
        self.emotion_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.emotion_model.fit(X_train_em_scaled, y_train_em)
        
        # Train speaker identifier
        self.speaker_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.speaker_model.fit(X_train_sp_scaled, y_train_sp)
        
        # Calculate accuracies
        emotion_acc = accuracy_score(y_test_em, self.emotion_model.predict(X_test_em_scaled))
        speaker_acc = accuracy_score(y_test_sp, self.speaker_model.predict(X_test_sp_scaled))
        
        return emotion_acc, speaker_acc, X_test_em_scaled, y_test_em, X_test_sp_scaled, y_test_sp
    
    def predict_emotion(self, audio, sr):
        """Predict emotion from audio"""
        features = self.extract_features(audio, sr)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.emotion_model.predict(features_scaled)[0]
        probabilities = self.emotion_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return self.emotion_labels[prediction], confidence, probabilities
    
    def predict_speaker(self, audio, sr):
        """Predict speaker from audio"""
        features = self.extract_features(audio, sr)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.speaker_model.predict(features_scaled)[0]
        probabilities = self.speaker_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return self.speaker_labels[prediction], confidence, probabilities

def main():
    # Title
    st.markdown('<h1 class="main-header">🎤 Voice Analysis ML Application</h1>', 
                unsafe_allow_html=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = VoiceAnalyzer()
        st.session_state.models_trained = False
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Model training section
        st.subheader("Model Training")
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                emotion_acc, speaker_acc, X_test_em, y_test_em, X_test_sp, y_test_sp = st.session_state.analyzer.train_models()
                st.session_state.models_trained = True
                st.session_state.emotion_acc = emotion_acc
                st.session_state.speaker_acc = speaker_acc
                st.session_state.test_data = {
                    'X_test_em': X_test_em, 'y_test_em': y_test_em,
                    'X_test_sp': X_test_sp, 'y_test_sp': y_test_sp
                }
            st.success("Models trained successfully!")
        
        if st.session_state.models_trained:
            st.metric("Emotion Accuracy", f"{st.session_state.emotion_acc:.2%}")
            st.metric("Speaker Accuracy", f"{st.session_state.speaker_acc:.2%}")
        
        # Analysis options
        st.subheader("Analysis Options")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Emotion Detection", "Speaker Identification", "Both"]
        )
        
        # Audio settings
        st.subheader("Audio Settings")
        sample_rate = st.selectbox("Sample Rate (Hz)", [16000, 22050, 44100], index=0)
        duration = st.slider("Max Duration (seconds)", 1, 30, 10)
    
    # Main content
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first using the sidebar!")
        st.info("Click 'Train Models' in the sidebar to get started.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["🎵 Audio Analysis", "📊 Model Performance", "🔍 Feature Analysis", "📈 Data Visualization"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Audio Input & Analysis</h2>', unsafe_allow_html=True)
        
        # Audio input methods
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_method = st.radio(
                "Choose input method:",
                ["Upload Audio File", "Generate Sample Audio", "Use Demo Audio"],
                horizontal=True
            )
            
            audio_data = None
            sr = sample_rate
            
            if input_method == "Upload Audio File":
                uploaded_file = st.file_uploader(
                    "Choose an audio file",
                    type=['wav', 'mp3', 'flac', 'm4a', 'ogg']
                )
                if uploaded_file is not None:
                    try:
                        audio_data, sr = librosa.load(uploaded_file, sr=sample_rate, duration=duration)
                        st.success("Audio file loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading audio file: {e}")
            
            elif input_method == "Generate Sample Audio":
                if st.button("Generate Synthetic Audio"):
                    # Generate synthetic audio signal
                    duration_samples = int(sample_rate * 3)  # 3 seconds
                    t = np.linspace(0, 3, duration_samples)
                    
                    # Create a complex waveform (mix of frequencies)
                    freq1, freq2, freq3 = 440, 880, 1320  # Musical notes
                    audio_data = (np.sin(2 * np.pi * freq1 * t) * 0.3 + 
                                 np.sin(2 * np.pi * freq2 * t) * 0.2 + 
                                 np.sin(2 * np.pi * freq3 * t) * 0.1 +
                                 np.random.normal(0, 0.05, duration_samples))  # Add some noise
                    sr = sample_rate
                    st.success("Synthetic audio generated!")
            
            elif input_method == "Use Demo Audio":
                if st.button("Load Demo Audio"):
                    # Generate a more realistic demo audio
                    duration_samples = int(sample_rate * 4)
                    t = np.linspace(0, 4, duration_samples)
                    
                    # Simulate speech-like patterns
                    formants = [800, 1200, 2400]  # Typical speech formants
                    audio_data = np.zeros(duration_samples)
                    
                    for formant in formants:
                        envelope = np.exp(-t * 0.5) * np.sin(2 * np.pi * 5 * t)  # Modulation
                        audio_data += np.sin(2 * np.pi * formant * t) * envelope * 0.2
                    
                    # Add realistic noise
                    audio_data += np.random.normal(0, 0.02, duration_samples)
                    sr = sample_rate
                    st.success("Demo audio loaded!")
            
            # Process audio if available
            if audio_data is not None:
                process_audio_analysis(audio_data, sr, analysis_type, st.session_state.analyzer)
        
        with col2:
            st.subheader("📋 Instructions")
            st.write("""
            1. **Train Models**: Use sidebar to train ML models
            2. **Choose Input**: Select audio input method
            3. **Upload/Generate**: Provide audio data
            4. **Analyze**: View predictions and visualizations
            """)
            
            st.subheader("🎯 Supported Emotions")
            emotions = st.session_state.analyzer.emotion_labels
            for i, emotion in enumerate(emotions):
                st.write(f"{i+1}. {emotion.title()}")
    
    with tab2:
        show_model_performance()
    
    with tab3:
        show_feature_analysis()
    
    with tab4:
        show_data_visualization()

def process_audio_analysis(audio_data, sr, analysis_type, analyzer):
    """Process and analyze audio data"""
    st.markdown('<h3 class="sub-header">Analysis Results</h3>', unsafe_allow_html=True)
    
    # Display audio player
    st.audio(audio_data, sample_rate=sr, format='audio/wav')
    
    # Audio visualizations
    create_audio_visualizations(audio_data, sr)
    
    # Make predictions
    col1, col2 = st.columns(2)
    
    if analysis_type in ["Emotion Detection", "Both"]:
        with col1:
            with st.spinner("Analyzing emotions..."):
                emotion, confidence, emotion_probs = analyzer.predict_emotion(audio_data, sr)
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f'<h3>🎭 Detected Emotion: {emotion.upper()}</h3>', unsafe_allow_html=True)
            st.markdown(f'<h4>Confidence: {confidence:.1%}</h4>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Emotion probability chart
            emotion_df = pd.DataFrame({
                'Emotion': analyzer.emotion_labels,
                'Probability': emotion_probs
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(emotion_df['Emotion'], emotion_df['Probability'], 
                         color=plt.cm.viridis(emotion_df['Probability']))
            ax.set_title('Emotion Prediction Probabilities', fontsize=16, fontweight='bold')
            ax.set_xlabel('Emotions')
            ax.set_ylabel('Probability')
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight the predicted emotion
            max_idx = np.argmax(emotion_probs)
            bars[max_idx].set_color('red')
            bars[max_idx].set_alpha(0.8)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    if analysis_type in ["Speaker Identification", "Both"]:
        with col2:
            with st.spinner("Identifying speaker..."):
                speaker, confidence, speaker_probs = analyzer.predict_speaker(audio_data, sr)
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f'<h3>👤 Identified Speaker: {speaker.upper()}</h3>', unsafe_allow_html=True)
            st.markdown(f'<h4>Confidence: {confidence:.1%}</h4>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show top 5 speaker predictions
            top_indices = np.argsort(speaker_probs)[-5:][::-1]
            top_speakers = [analyzer.speaker_labels[i] for i in top_indices]
            top_probs = [speaker_probs[i] for i in top_indices]
            
            speaker_df = pd.DataFrame({
                'Speaker': top_speakers,
                'Probability': top_probs
            })
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(speaker_df['Speaker'], speaker_df['Probability'], 
                         color=plt.cm.plasma(speaker_df['Probability']))
            ax.set_title('Top 5 Speaker Predictions', fontsize=16, fontweight='bold')
            ax.set_xlabel('Speaker ID')
            ax.set_ylabel('Probability')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)

def create_audio_visualizations(audio_data, sr):
    """Create comprehensive audio visualizations"""
    st.subheader("🎵 Audio Visualizations")
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Waveform
    time = np.linspace(0, len(audio_data)/sr, len(audio_data))
    axes[0, 0].plot(time, audio_data, color='blue', alpha=0.7)
    axes[0, 0].set_title('Waveform', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img1 = librosa.display.specshow(D, y_axis='hz', x_axis='time', ax=axes[0, 1])
    axes[0, 1].set_title('Spectrogram', fontsize=14, fontweight='bold')
    plt.colorbar(img1, ax=axes[0, 1])
    
    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)
    img2 = librosa.display.specshow(mel_db, y_axis='mel', x_axis='time', ax=axes[1, 0])
    axes[1, 0].set_title('Mel-Spectrogram', fontsize=14, fontweight='bold')
    plt.colorbar(img2, ax=axes[1, 0])
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    img3 = librosa.display.specshow(mfcc, x_axis='time', ax=axes[1, 1])
    axes[1, 1].set_title('MFCC Features', fontsize=14, fontweight='bold')
    plt.colorbar(img3, ax=axes[1, 1])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional feature visualization
    st.subheader("🔊 Spectral Features")
    
    # Calculate spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
    
    # Plot spectral features
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, spectral_centroids, label='Spectral Centroid', color='blue')
    ax.plot(t, spectral_rolloff, label='Spectral Rolloff', color='red')
    ax.plot(t, zero_crossing_rate * sr / 10, label='ZCR (scaled)', color='green')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Hz')
    ax.set_title('Spectral Features Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def show_model_performance():
    """Display model performance metrics"""
    st.markdown('<h2 class="sub-header">📊 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'test_data'):
        st.warning("Please train models first to see performance metrics!")
        return
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Emotion Classification Performance")
        st.metric("Model Accuracy", f"{st.session_state.emotion_acc:.1%}")
        
        # Confusion matrix for emotions
        analyzer = st.session_state.analyzer
        X_test_em = st.session_state.test_data['X_test_em']
        y_test_em = st.session_state.test_data['y_test_em']
        y_pred_em = analyzer.emotion_model.predict(X_test_em)
        
        cm_emotion = confusion_matrix(y_test_em, y_pred_em)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_emotion, annot=True, fmt='d', cmap='Blues',
                   xticklabels=analyzer.emotion_labels,
                   yticklabels=analyzer.emotion_labels, ax=ax)
        ax.set_title('Emotion Classification Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Classification report
        report_em = classification_report(y_test_em, y_pred_em, 
                                        target_names=analyzer.emotion_labels,
                                        output_dict=True)
        report_df_em = pd.DataFrame(report_em).transpose()
        st.subheader("Emotion Classification Report")
        st.dataframe(report_df_em.round(3))
    
    with col2:
        st.subheader("👤 Speaker Identification Performance")
        st.metric("Model Accuracy", f"{st.session_state.speaker_acc:.1%}")
        
        # Feature importance plot
        feature_names = [f'Feature_{i}' for i in range(len(analyzer.emotion_model.feature_importances_))]
        importance_em = analyzer.emotion_model.feature_importances_
        importance_sp = analyzer.speaker_model.feature_importances_
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top 20 features for emotion
        top_indices_em = np.argsort(importance_em)[-20:]
        ax1.barh(range(20), importance_em[top_indices_em])
        ax1.set_yticks(range(20))
        ax1.set_yticklabels([feature_names[i] for i in top_indices_em])
        ax1.set_title('Top 20 Features - Emotion Classification')
        ax1.set_xlabel('Feature Importance')
        
        # Top 20 features for speaker
        top_indices_sp = np.argsort(importance_sp)[-20:]
        ax2.barh(range(20), importance_sp[top_indices_sp])
        ax2.set_yticks(range(20))
        ax2.set_yticklabels([feature_names[i] for i in top_indices_sp])
        ax2.set_title('Top 20 Features - Speaker Identification')
        ax2.set_xlabel('Feature Importance')
        
        plt.tight_layout()
        st.pyplot(fig)

def show_feature_analysis():
    """Show feature extraction and analysis"""
    st.markdown('<h2 class="sub-header">🔍 Feature Analysis</h2>', unsafe_allow_html=True)
    
    analyzer = st.session_state.analyzer
    
    # Feature description
    st.subheader("📋 Extracted Features")
    feature_info = {
        'Feature Type': [
            'MFCC Mean', 'MFCC Std', 'Chroma', 'Mel-Spectrogram',
            'Spectral Centroid', 'Spectral Rolloff', 'Zero Crossing Rate'
        ],
        'Count': [13, 13, 12, 20, 1, 1, 1],
        'Description': [
            'Mean of Mel-Frequency Cepstral Coefficients',
            'Standard deviation of MFCCs',
            'Pitch class profiles',
            'Mel-scale power spectrogram features',
            'Spectral centroid mean',
            'Spectral rolloff frequency',
            'Rate of zero crossings'
        ]
    }
    
    feature_df = pd.DataFrame(feature_info)
    st.table(feature_df)
    
    # Generate sample features for visualization
    if st.button("Generate Feature Samples"):
        # Create sample audio
        sr = 16000
        duration = 3
        t = np.linspace(0, duration, sr * duration)
        
        # Different types of audio signals
        signals = {
            'Pure Tone (440Hz)': np.sin(2 * np.pi * 440 * t),
            'Chirp Signal': np.sin(2 * np.pi * (200 + 100 * t) * t),
            'White Noise': np.random.normal(0, 0.1, len(t)),
            'Speech-like': np.sin(2 * np.pi * 800 * t) * np.exp(-t * 0.5) + 
                          np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 0.3)
        }
        
        # Extract features for each signal
        features_data = []
        for name, signal in signals.items():
            features = analyzer.extract_features(signal, sr)
            features_data.append({
                'Signal Type': name,
                'MFCC_1': features[0],
                'MFCC_2': features[1],
                'MFCC_3': features[2],
                'Chroma_1': features[26],
                'Mel_1': features[38],
                'Spectral_Centroid': features[58],
                'ZCR': features[-1]
            })
        
        features_comparison = pd.DataFrame(features_data)
        
        st.subheader("🎵 Feature Comparison Across Signal Types")
        st.dataframe(features_comparison.round(4))
        
        # Visualize feature differences
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot some key features
        feature_cols = ['MFCC_1', 'Chroma_1', 'Spectral_Centroid', 'ZCR']
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (ax, col, color) in enumerate(zip(axes.flat, feature_cols, colors)):
            ax.bar(features_comparison['Signal Type'], features_comparison[col], 
                  color=color, alpha=0.7)
            ax.set_title(f'{col} Across Signal Types')
            ax.set_ylabel('Feature Value')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

def show_data_visualization():
    """Show various data visualizations"""
    st.markdown('<h2 class="sub-header">📈 Data Visualization Dashboard</h2>', unsafe_allow_html=True)
    
    # Generate synthetic dataset for visualization
    analyzer = st.session_state.analyzer
    X, y_emotion, y_speaker = analyzer.generate_sample_data(1000)
    
    # Create DataFrame for easier handling
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Emotion'] = [analyzer.emotion_labels[i] for i in y_emotion]
    df['Speaker'] = [analyzer.speaker_labels[i] for i in y_speaker]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Emotion Distribution")
        emotion_counts = df['Emotion'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_counts)))
        bars = ax.bar(emotion_counts.index, emotion_counts.values, color=colors)
        ax.set_title('Distribution of Emotions in Dataset')
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature correlation heatmap
        st.subheader("🔗 Feature Correlation Matrix")
        # Select first 20 features for readability
        correlation_features = df[feature_names[:20]]
        corr_matrix = correlation_features.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix (First 20 Features)')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("👥 Speaker Distribution")
        # Show top 10 speakers for readability
        speaker_counts = df['Speaker'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(speaker_counts)), speaker_counts.values, color='skyblue')
        ax.set_title('Top 10 Speakers in Dataset')
        ax.set_xlabel('Speaker ID')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(speaker_counts)))
        ax.set_xticklabels(speaker_counts.index, rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 3D scatter plot of first 3 features colored by emotion
        st.subheader("🎨 3D Feature Space Visualization")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        emotions = df['Emotion'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(emotions)))
        
        for emotion, color in zip(emotions, colors):
            mask = df['Emotion'] == emotion
            ax.scatter(df.loc[mask, 'Feature_0'], 
                      df.loc[mask, 'Feature_1'], 
                      df.loc[mask, 'Feature_2'],
                      c=[color], label=emotion, alpha=0.6, s=20)
        
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
        ax.set_zlabel('Feature 2')
        ax.set_title('3D Feature Space (First 3 Features)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Statistical summary
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Number of Features", len(feature_names))
    with col3:
        st.metric("Emotion Classes", len(analyzer.emotion_labels))
    with col4:
        st.metric("Speaker Classes", len(analyzer.speaker_labels))
    
    # Feature statistics table
    st.subheader("📋 Feature Statistics Summary")
    feature_stats = df[feature_names[:10]].describe()  # Show first 10 features
    st.dataframe(feature_stats.round(4))

if __name__ == "__main__":
    main()
"""Feature extraction utilities"""

import librosa
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract audio features for ML models"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract comprehensive feature set"""
        features = []
        
        # MFCC features
        mfcc_features = self._extract_mfcc(audio)
        features.extend(mfcc_features)
        
        # Chroma features
        chroma_features = self._extract_chroma(audio)
        features.extend(chroma_features)
        
        # Mel-spectrogram features
        mel_features = self._extract_mel_spectrogram(audio)
        features.extend(mel_features)
        
        # Spectral features
        spectral_features = self._extract_spectral_features(audio)
        features.extend(spectral_features)
        
        # Rhythmic features
        rhythmic_features = self._extract_rhythmic_features(audio)
        features.extend(rhythmic_features)
        
        return np.array(features)
    
    def _extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> List[float]:
        """Extract MFCC features"""
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
            
            # Statistical features
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Delta features
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            
            return np.concatenate([mfcc_mean, mfcc_std, mfcc_delta_mean]).tolist()
            
        except Exception as e:
            logger.error(f"MFCC extraction failed: {e}")
            return [0.0] * (n_mfcc * 3)
    
    def _extract_chroma(self, audio: np.ndarray) -> List[float]:
        """Extract chroma features"""
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
            chroma_mean = np.mean(chroma, axis=1)
            return chroma_mean.tolist()
        except Exception as e:
            logger.error(f"Chroma extraction failed: {e}")
            return [0.0] * 12
    
    def _extract_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 20) -> List[float]:
        """Extract mel-spectrogram features"""
        try:
            mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=n_mels)
            mel_mean = np.mean(mel, axis=1)
            return mel_mean.tolist()
        except Exception as e:
            logger.error(f"Mel-spectrogram extraction failed: {e}")
            return [0.0] * n_mels
    
    def _extract_spectral_features(self, audio: np.ndarray) -> List[float]:
        """Extract spectral features"""
        features = []
        
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
            features.append(np.mean(spectral_rolloff))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
            features.append(np.mean(spectral_bandwidth))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.append(np.mean(zcr))
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            features = [0.0] * 5
            
        return features
    
    def _extract_rhythmic_features(self, audio: np.ndarray) -> List[float]:
        """Extract rhythmic features"""
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
            onset_rate = len(onset_frames) / (len(audio) / self.sr)
            
            return [tempo, onset_rate]
            
        except Exception as e:
            logger.error(f"Rhythmic feature extraction failed: {e}")
            return [0.0, 0.0]
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features"""
        names = []
        
        # MFCC names
        for i in range(13):
            names.append(f'mfcc_{i}_mean')
        for i in range(13):
            names.append(f'mfcc_{i}_std')
        for i in range(13):
            names.append(f'mfcc_delta_{i}_mean')
        
        # Chroma names
        for i in range(12):
            names.append(f'chroma_{i}')
            
        # Mel-spectrogram names
        for i in range(20):
            names.append(f'mel_{i}')
            
        # Spectral feature names
        names.extend([
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_bandwidth_mean',
            'zero_crossing_rate_mean'
        ])
        
        # Rhythmic feature names
        names.extend(['tempo', 'onset_rate'])
        
        return names
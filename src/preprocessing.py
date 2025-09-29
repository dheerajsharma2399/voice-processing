"""Audio preprocessing utilities"""

import librosa
import numpy as np
from scipy import signal
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Audio preprocessing pipeline"""
    
    def __init__(self, target_sr: int = 16000, max_duration: float = 30):
        self.target_sr = target_sr
        self.max_duration = max_duration
    
    def load_audio(self, file_path: str, 
                   duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load and basic preprocess audio file"""
        try:
            audio, sr = librosa.load(
                file_path, 
                sr=self.target_sr,
                duration=duration or self.max_duration
            )
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Complete audio preprocessing pipeline"""
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize amplitude
        audio = self._normalize_audio(audio)
        
        # Apply noise reduction
        audio = self._reduce_noise(audio, sr)
        
        # Pad or truncate to fixed length
        audio = self._pad_or_truncate(audio, sr)
        
        return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Basic noise reduction"""
        # High-pass filter to remove low-frequency noise
        if sr > 300:
            sos = signal.butter(5, 300, btype='high', fs=sr, output='sos')
            audio = signal.sosfilt(sos, audio)
        return audio
    
    def _pad_or_truncate(self, audio: np.ndarray, sr: int, 
                        target_length: Optional[int] = None) -> np.ndarray:
        """Pad or truncate audio to target length"""
        if target_length is None:
            target_length = int(self.max_duration * sr)
        
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            pad_length = target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        return audio
    
    def augment_audio(self, audio: np.ndarray, sr: int) -> List[Tuple[str, np.ndarray]]:
        """Apply data augmentation techniques"""
        augmented = []
        
        # Time stretching
        try:
            stretched = librosa.effects.time_stretch(audio, rate=0.9)
            augmented.append(('time_stretch', stretched))
        except Exception as e:
            logger.warning(f"Time stretch failed: {e}")
        
        # Pitch shifting
        try:
            pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
            augmented.append(('pitch_shift', pitched))
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
        
        # Add noise
        noise = np.random.normal(0, 0.02, len(audio))
        noisy = audio + noise
        augmented.append(('add_noise', noisy))
        
        return augmented
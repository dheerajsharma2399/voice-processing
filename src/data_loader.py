import os
import librosa
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VoiceDataLoader:
    """
    A class to load voice data from specified directories.
    """

    def __init__(self, data_path: str):
        """
        Initializes the data loader with the path to the data.

        Args:
            data_path: The path to the data directory.
        """
        self.data_path = data_path
        self.emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }

    def load_ravdess_data(self) -> List[Dict[str, Any]]:
        """
        Loads audio files from the RAVDESS dataset and extracts metadata from filenames.

        Returns:
            A list of dictionaries, where each dictionary contains:
            - 'path': The full path to the audio file.
            - 'filename': The name of the audio file.
            - 'emotion': The emotion label.
            - 'actor': The actor ID.
            - 'gender': The gender of the actor ('male' or 'female').
            - 'intensity': The emotional intensity ('normal' or 'strong').
            - 'statement': The statement ID.
            - 'repetition': The repetition number.
            - 'vocal_channel': The vocal channel ('speech' or 'song').
        """
        ravdess_path = os.path.join(self.data_path, 'RAVDESS')
        if not os.path.isdir(ravdess_path):
            logger.error(f"RAVDESS directory not found at: {ravdess_path}")
            return []

        logger.info(f"Loading RAVDESS data from: {ravdess_path}")
        data = []
        for actor_dir in sorted(os.listdir(ravdess_path)):
            actor_path = os.path.join(ravdess_path, actor_dir)
            if os.path.isdir(actor_path):
                for filename in os.listdir(actor_path):
                    if filename.endswith('.wav'):
                        try:
                            parts = filename.split('.')[0].split('-')
                            if len(parts) == 7:
                                emotion = self.emotion_map.get(parts[2])
                                actor = int(parts[6])
                                gender = 'female' if actor % 2 == 0 else 'male'
                                intensity = 'strong' if parts[3] == '02' else 'normal'
                                
                                file_path = os.path.join(actor_path, filename)
                                data.append({
                                    'path': file_path,
                                    'filename': filename,
                                    'emotion': emotion,
                                    'actor': actor,
                                    'gender': gender,
                                    'intensity': intensity,
                                    'statement': parts[4],
                                    'repetition': parts[5],
                                    'vocal_channel': 'song' if parts[1] == '02' else 'speech'
                                })
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse filename {filename}: {e}")
        
        logger.info(f"Loaded {len(data)} audio files from RAVDESS.")
        return data

    def load_audio_file(self, file_path: str, sample_rate: int = 22050) -> tuple:
        """
        Loads a single audio file.

        Args:
            file_path: The path to the audio file.
            sample_rate: The desired sample rate.

        Returns:
            A tuple containing the audio time series and the sample rate.
        """
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return None, None

if __name__ == '__main__':
    # Example usage:
    DATA_PATH = 'e:/Assignment_data/voice processing/data/raw'
    loader = VoiceDataLoader(DATA_PATH)
    ravdess_files = loader.load_ravdess_data()
    
    if ravdess_files:
        print(f"Successfully loaded {len(ravdess_files)} RAVDESS files.")
        # Example of loading the first audio file
        first_file = ravdess_files[0]
        print(f"Loading first file: {first_file['path']}")
        audio, sr = loader.load_audio_file(first_file['path'])
        if audio is not None:
            print(f"Audio loaded successfully with sample rate {sr} and duration {len(audio)/sr:.2f}s.")
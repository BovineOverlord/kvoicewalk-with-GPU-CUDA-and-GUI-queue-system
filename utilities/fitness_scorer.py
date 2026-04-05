# from scipy.spatial.distance import cosine
from typing import Any

import librosa
import numpy as np
import scipy.stats
import soundfile as sf
from numpy._typing import NDArray
from resemblyzer import preprocess_wav, VoiceEncoder

from utilities.device_utils import resolve_device


class FitnessScorer:
    def __init__(self, target_path: str, device: str = "auto"):
        self.device = resolve_device(device)
        self.encoder = VoiceEncoder(device=self.device)
        self.target_audio, _ = sf.read(target_path,dtype="float32")
        self.target_wav = preprocess_wav(target_path,source_sr=24000)
        self.target_embed = self.encoder.embed_utterance(self.target_wav)
        self.target_features = self.extract_features(self.target_audio)

    def hybrid_similarity(self, audio: NDArray[np.float32], audio2: NDArray[np.float32],target_similarity: float):
        features = self.extract_features(audio)
        self_similarity = self.self_similarity(audio,audio2)
        target_features_pentalty = self.target_feature_penalty(features)

        #Normalize and make higher = better
        feature_similarity = (100.0 - target_features_pentalty) / 100.0
        if feature_similarity < 0.0:
            feature_similarity = 0.01

        values = [target_similarity, self_similarity, feature_similarity]
        # Playing around with the weights can greatly affect scoring and random walk behavior
        weights = [0.48,0.5,0.02]
        score = (np.sum(weights) / np.sum(np.array(weights) / np.array(values))) * 100.0

        return {
            "score": score,
            "target_similarity": target_similarity,
            "self_similarity": self_similarity,
            "feature_similarity": feature_similarity
        }

    def target_similarity(self,audio: NDArray[np.float32]) -> float:
        audio_wav = preprocess_wav(audio,source_sr=24000)
        audio_embed = self.encoder.embed_utterance(audio_wav)
        similarity = np.inner(audio_embed, self.target_embed)
        return similarity

    def target_feature_penalty(self,features: dict[str, Any]) -> float:
        """Penalizes for differences in audio features"""
        # Normalized feature difference compared to target features
        penalty = 0.0
        for key, value in features.items():
            diff = abs((value - self.target_features[key])/self.target_features[key])
            penalty += diff
        return penalty

    def self_similarity(self,audio1: NDArray[np.float32], audio2: NDArray[np.float32]) -> float:
        """Self similarity indicates model stability. Poor self similarity means different input makes different sounding voices"""
        audio_wav1 = preprocess_wav(audio1,source_sr=24000)
        audio_embed1 = self.encoder.embed_utterance(audio_wav1)

        audio_wav2 = preprocess_wav(audio2,source_sr=24000)
        audio_embed2 = self.encoder.embed_utterance(audio_wav2)
        return np.inner(audio_embed1, audio_embed2)

    def extract_features(self, audio: NDArray[np.float32] | NDArray[np.float64], sr: int = 24000) -> dict[str, Any]:
        """
        Extract a comprehensive set of audio features for fingerprinting speech segments.

        Args:
            audio: Audio signal as numpy array (np.float32)
            sr: Sample rate (fixed at 24000 Hz)

        Returns:
            Dictionary containing extracted features
        """
        # Ensure audio is the right shape (flatten stereo to mono if needed)
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)

        # Initialize features dictionary
        features = {}

        # Basic features
        # features["duration"] = len(audio) / sr
        features["rms_energy"] = float(np.sqrt(np.mean(audio**2)))
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

        # Spectral features
        # Compute STFT
        n_fft = 2048  # Window size
        hop_length = 512  # Hop length

        # Spectral centroid and bandwidth (where the "center" of the sound is)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["spectral_contrast_mean"] = float(np.mean(contrast))
        features["spectral_contrast_std"] = float(np.std(contrast))

        # MFCCs (Mel-frequency cepstral coefficients) - important for speech
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

        # Store each MFCC coefficient mean and std
        for i in range(len(mfccs)):
            features[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc{i+1}_std"] = float(np.std(mfccs[i]))

        # MFCC delta features (first derivative)
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(len(mfcc_delta)):
            features[f"mfcc{i+1}_delta_mean"] = float(np.mean(mfcc_delta[i]))
            features[f"mfcc{i+1}_delta_std"] = float(np.std(mfcc_delta[i]))

        # Chroma features - useful for characterizing harmonic content
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))

        # Store individual chroma features
        for i in range(len(chroma)):
            features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

        # Mel spectrogram (average across frequency bands)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["mel_spec_mean"] = float(np.mean(mel_spec))
        features["mel_spec_std"] = float(np.std(mel_spec))

        # Spectral flatness - measure of the noisiness of the signal
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_flatness_mean"] = float(np.mean(flatness))
        features["spectral_flatness_std"] = float(np.std(flatness))

        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features["tonnetz_mean"] = float(np.mean(tonnetz))
        features["tonnetz_std"] = float(np.std(tonnetz))

        # Rhythm features - tempo and beat strength
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        features["tempo"] = float(tempo)

        if len(beat_frames) > 0:
            # Calculate beat_stats only if beats are detected
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            if len(beat_times) > 1:
                beat_diffs = np.diff(beat_times)
                features["beat_mean"] = float(np.mean(beat_diffs))
                features["beat_std"] = float(np.std(beat_diffs))
            else:
                features["beat_mean"] = 0.0
                features["beat_std"] = 0.0
        else:
            features["beat_mean"] = 0.0
            features["beat_std"] = 0.0

        # Pitch and harmonics
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)

        # For each frame, find the highest magnitude pitch
        pitch_values = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:  # Exclude zero pitch
                pitch_values.append(pitch)

        if pitch_values:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"] = float(np.std(pitch_values))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0

        # Speech-specific features

        # Voice Activity Detection (simplified)
        # Higher energies typically indicate voice activity
        energy = np.array([sum(abs(audio[i:i+hop_length])) for i in range(0, len(audio), hop_length)])
        features["energy_mean"] = float(np.mean(energy))
        features["energy_std"] = float(np.std(energy))

        # Harmonics-to-noise ratio (simplified approximation)
        # Using the squared magnitude of the spectrogram
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_squared = S**2
        S_mean = np.mean(S_squared, axis=1)
        S_std = np.std(S_squared, axis=1)
        S_ratio = np.divide(S_mean, S_std, out=np.zeros_like(S_mean), where=S_std!=0)
        features["harmonic_ratio"] = float(np.mean(S_ratio))

        # Statistical features from the raw waveform
        features["audio_mean"] = float(np.mean(audio))
        features["audio_std"] = float(np.std(audio))
        features["audio_skew"] = float(scipy.stats.skew(audio))
        features["audio_kurtosis"] = float(scipy.stats.kurtosis(audio))

        return features



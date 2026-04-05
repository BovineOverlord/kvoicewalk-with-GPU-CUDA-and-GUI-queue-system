import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from utilities.device_utils import resolve_device
from utilities.path_router import TEXTS_DIR, CONVERTED_DIR


def convert_to_wav_mono_24k(audio_path: Path) -> Path:
    print(f"Converting {audio_path.name} to Mono Wav 24K...")
    try:
        with sf.SoundFile(audio_path, "r") as f:
            if f.format != "WAV" or f.samplerate != 24000 or f.channels != 1:
                # Create output filename with proper audio format
                converted_audio_file = Path(CONVERTED_DIR / str(audio_path.stem + ".wav"))

                # Read the audio data
                audio_data = f.read()

                # Convert to mono if needed
                if f.channels > 1:
                    converted_audio_data = np.mean(audio_data, axis=1)
                    print("Cenverted to Mono...")
                else:
                    converted_audio_data = audio_data

                # Resample if needed
                if f.samplerate != 24000:
                    converted_audio_data = librosa.resample(
                        converted_audio_data,
                        orig_sr=f.samplerate,
                        target_sr=24000,
                    )
                    print("Resampled to 24K...")

                # Save converted audio
                sf.write(converted_audio_file, converted_audio_data, samplerate=24000, format="WAV")
                print(f"{audio_path.name} successfully converted to Mono WAV 24K format: {converted_audio_file}")
                return converted_audio_file
            else:
                print(f"{audio_path.name} matches Mono WAV 24K format")
                return audio_path

    except Exception as e:
        print(f"Error converting {audio_path.name}: {e}\n")
        raise


class Transcriber:
    def __init__(self, device: str = "auto"):
        model_size = "large-v3"
        self.device = resolve_device(device)
        print(f"Starting Transcriber on {self.device}...")

        if self.device == "cuda":
            for compute_type in ("float16", "int8_float16", "float32"):
                try:
                    self.model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
                    print(f"Loaded faster-whisper with compute_type={compute_type}")
                    break
                except Exception as e:
                    print(f"Failed to load CUDA transcriber with compute_type={compute_type}: {e}")
            else:
                print("Falling back to CPU transcriber.")
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: Path):
        audio_file = audio_path
        start_time = datetime.datetime.now()

        try:
            print(f"Loading {audio_file.name}...")
            segments, info = self.model.transcribe(str(audio_file), beam_size=5)

            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            print(f"Transcribing {audio_file.name}...")

            transcription = ""
            for segment in segments:
                transcription += segment.text
                # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)) # Optional timestamps if parsing longer audio clips

            transcription_output = Path(TEXTS_DIR / str(f"{audio_file.stem}.txt"))
            with open(str(transcription_output), "w") as file:
                file.write(f"{transcription}")

            end_time = datetime.datetime.now()
            print(f"Transcription completed in {(end_time - start_time).total_seconds()} seconds")
            print(f"Transcription available at ./texts/{audio_file.name[:-4]}.txt")
            print(f"{audio_file.name} Transcription:\n{transcription}")
            return transcription

        except Exception as e:
            print(f"Transcription failed for {audio_file.name} - Error: {e}")
            return


# TODO: Integrate into automated workflows

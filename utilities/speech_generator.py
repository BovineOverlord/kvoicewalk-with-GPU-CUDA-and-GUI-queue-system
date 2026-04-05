import warnings

import numpy as np
import torch
from kokoro import KPipeline

from utilities.device_utils import resolve_device


class SpeechGenerator:
    def __init__(self, device: str = "auto"):
        surpressWarnings()
        self.device = resolve_device(device)
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=self.device)

    def generate_audio(self, text: str, voice: torch.Tensor | str, speed: float = 1.0) -> np.typing.NDArray[np.float32]:
        voice_arg = voice
        if isinstance(voice, torch.Tensor):
            # kokoro==0.9.x checks specifically for torch.FloatTensor in load_voice;
            # normalize any tensor (including CUDA tensors) to CPU float32 first.
            voice_arg = voice.detach().to(device="cpu", dtype=torch.float32)

        generator = self.pipeline(text, voice=voice_arg, speed=speed)
        audio = [chunk for _, _, chunk in generator]
        if not audio:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio)


def surpressWarnings():
    # Surpress all these warnings showing up from libraries cluttering the console
    warnings.filterwarnings(
        "ignore",
        message=".*RNN module weights are not part of single contiguous chunk of memory.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore", message=".*is deprecated in favor of*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*dropout option adds dropout after all but last recurrent layer*",
        category=UserWarning,
    )

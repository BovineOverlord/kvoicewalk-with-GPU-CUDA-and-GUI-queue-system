import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class SkipFileException(Exception):
    """Exception to skip loading a file."""


class VoiceLoader:
    """Handles safe loading of voice files with user controls."""

    def __init__(self, auto_allow_unsafe: bool = False, auto_deny_unsafe: bool = False):
        self.auto_allow_unsafe = auto_allow_unsafe
        self.auto_deny_unsafe = auto_deny_unsafe
        self.risky_files: List[str] = []

    def safe_load_pt_file(self, file_path: str) -> Union[torch.Tensor, Dict[str, Any]]:
        """Safely load a .pt file with user interaction for unsafe files."""
        try:
            data = torch.load(file_path, weights_only=True)
            print(f"OK: Safely loaded {file_path} with weights_only=True")
            return data
        except Exception as safe_error:
            print(f"WARN: Safe loading failed: {safe_error}")

        try:
            with torch.serialization.safe_globals(
                [
                    np.core.multiarray._reconstruct,
                    np.ndarray,
                    np.dtype,
                    np.core.multiarray.scalar,
                ]
            ):
                data = torch.load(file_path)
                print(f"OK: Safely loaded {file_path} with numpy globals allowed")
                return data
        except Exception as numpy_error:
            print(f"WARN: Numpy-safe loading failed: {numpy_error}")

        return self._handle_unsafe_loading(file_path)

    def _handle_unsafe_loading(self, file_path: str) -> Union[torch.Tensor, Dict[str, Any]]:
        """Handle unsafe file loading with user interaction."""
        print(f"RISK: File {file_path} requires unsafe loading")

        if self.auto_deny_unsafe:
            self.risky_files.append(file_path)
            raise SkipFileException(f"Auto-denied unsafe loading of {file_path}")

        if self.auto_allow_unsafe:
            self.risky_files.append(file_path)
            return self._unsafe_load(file_path)

        while True:
            choice = input(
                f"\nOptions for {os.path.basename(file_path)}:\n"
                f"[Y]es - Load unsafely\n"
                f"[N]o - Skip this file\n"
                f"[A]ll - Yes to all remaining files\n"
                f"[D]eny - No to all remaining files\n"
                f"[E]xit - Stop program\n"
                f"Choice: "
            ).upper().strip()

            if choice == "E":
                print("User stopping program")
                raise SystemExit
            if choice == "D":
                self.auto_deny_unsafe = True
                self.risky_files.append(file_path)
                raise SkipFileException(f"User denied loading {file_path}")
            if choice == "N":
                self.risky_files.append(file_path)
                raise SkipFileException(f"User declined loading {file_path}")
            if choice == "A":
                self.auto_allow_unsafe = True
                self.risky_files.append(file_path)
                return self._unsafe_load(file_path)
            if choice == "Y":
                self.risky_files.append(file_path)
                return self._unsafe_load(file_path)

            print("Invalid choice. Please try again.")

    def _unsafe_load(self, file_path: str) -> Union[torch.Tensor, Dict[str, Any]]:
        """Perform unsafe loading."""
        try:
            data = torch.load(file_path, weights_only=False)
            print(f"OK: Loaded {file_path} with weights_only=False")
            return data
        except Exception as final_error:
            raise RuntimeError(f"ERROR: Could not load {file_path}: {final_error}")

    def convert_loaded_data_to_tensor(self, data: Union[torch.Tensor, Dict, np.ndarray]) -> torch.Tensor:
        """Convert loaded data to a PyTorch tensor regardless of original format."""
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        if isinstance(data, dict):
            if "voice_vector" in data:
                return self.convert_loaded_data_to_tensor(data["voice_vector"])
            if "style" in data:
                return self.convert_loaded_data_to_tensor(data["style"])
            if "tensor" in data:
                return self.convert_loaded_data_to_tensor(data["tensor"])
            for key, value in data.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    print(f"Found tensor data in key: {key}")
                    return self.convert_loaded_data_to_tensor(value)
            raise ValueError(f"No tensor data found in dictionary. Keys: {list(data.keys())}")
        raise TypeError(f"Cannot convert {type(data)} to tensor")

    def load_voice_safely(self, file_path: str) -> Optional[torch.Tensor]:
        """Complete safe loading pipeline for voice files."""
        print(f"Loading voice file: {file_path}")

        try:
            raw_data = self.safe_load_pt_file(file_path)
            voice_tensor = self.convert_loaded_data_to_tensor(raw_data)
            if voice_tensor.numel() == 0:
                raise ValueError(f"Loaded tensor is empty from {file_path}")
            return voice_tensor
        except SkipFileException as e:
            print(f"SKIP: {e}")
            return None

    def get_risk_report(self) -> str:
        """Generate a report of risky files."""
        report: List[str] = []
        if self.risky_files:
            report.append("RISKY FILES LOADED:")
            report.append(
                "These files required unsafe loading. Consider redownloading or repairing them in a secure environment."
            )
            for file in self.risky_files:
                report.append(f"  - {file}")
        return "\n".join(report)


def load_voice_safely(
    file_path: str,
    auto_allow_unsafe: bool = False,
    auto_deny_unsafe: bool = False,
) -> Optional[torch.Tensor]:
    """Standalone function for loading a single voice file."""
    loader = VoiceLoader(auto_allow_unsafe, auto_deny_unsafe)
    result = loader.load_voice_safely(file_path)

    report = loader.get_risk_report()
    if report:
        print(f"\n{report}")

    return result


def load_multiple_voices(
    file_paths: List[str],
    auto_allow_unsafe: bool = False,
    auto_deny_unsafe: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load multiple voice files with shared user preferences."""
    loader = VoiceLoader(auto_allow_unsafe, auto_deny_unsafe)
    voices: Dict[str, torch.Tensor] = {}

    for file_path in file_paths:
        try:
            voice = loader.load_voice_safely(file_path)
            if voice is not None:
                voices[os.path.basename(file_path)] = voice
        except SystemExit:
            break

    print("\nLOADING SUMMARY:")
    print(f"Successfully loaded: {len(voices)} files")
    report = loader.get_risk_report()
    if report:
        print(report)

    return voices

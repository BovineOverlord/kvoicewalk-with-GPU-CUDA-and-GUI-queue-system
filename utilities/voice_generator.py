import torch

from utilities.device_utils import resolve_device


class VoiceGenerator:
    def __init__(self, voices: list[torch.Tensor], starting_voice: str | None, device: str = "auto"):
        self.device = torch.device(resolve_device(device))
        self.voices = [voice.detach().to(self.device) for voice in voices]

        self.stacked = torch.stack(self.voices, dim=0)
        self.mean = self.stacked.mean(dim=0)
        self.std = self.stacked.std(dim=0)
        self.min = self.stacked.min(dim=0)[0]
        self.max = self.stacked.max(dim=0)[0]

        if starting_voice:
            self.starting_voice = torch.load(starting_voice, map_location=self.device)
        else:
            self.starting_voice = self.mean

    def generate_voice(
        self,
        base_tensor: torch.Tensor | None,
        diversity: float = 1.0,
        device: str | None = None,
        clip: bool = False,
    ) -> torch.Tensor:
        """Generate a new voice tensor based on the base_tensor and diversity."""
        active_device = self.device if device is None else torch.device(resolve_device(device))

        if base_tensor is None:
            base_tensor = self.mean.to(active_device)
        else:
            base_tensor = base_tensor.clone().to(active_device)

        # Generate random noise with same shape
        noise = torch.randn_like(base_tensor, device=active_device)

        # Scale noise by standard deviation and the diversity factor
        scaled_noise = noise * self.std.to(active_device) * diversity

        # Add scaled noise to base tensor
        new_tensor = base_tensor + scaled_noise

        if clip:
            new_tensor = torch.clamp(new_tensor, self.min.to(active_device), self.max.to(active_device))

        return new_tensor

    # TODO: Make more voice generation functions

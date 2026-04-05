import torch


def resolve_device(preferred: str | None = "auto") -> str:
    requested = (preferred or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, cuda")

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    return requested

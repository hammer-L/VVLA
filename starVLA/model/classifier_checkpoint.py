"""Save and load compact language-classifier checkpoints.

The public helpers deliberately accept both a complete StarVLA state dict and
the module-local state dict written by current classifier training runs.  This
keeps old, multi-gigabyte checkpoints usable without perpetuating them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


_CLASSIFIER_MARKER = "language_classifier."


def load_state_dict_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a PyTorch or safetensors state dict on CPU."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError(f"Checkpoint {path} does not contain a tensor state dict")
    return state


def classifier_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    expected_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract a module-local classifier state dict from old or new layouts."""
    extracted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        marker_index = key.find(_CLASSIFIER_MARKER)
        if marker_index >= 0:
            local_key = key[marker_index + len(_CLASSIFIER_MARKER) :]
            extracted[local_key] = value

    if not extracted:
        # New checkpoints intentionally have no module prefix.  When the target
        # module is available, require every key to be one of its own keys so a
        # full/incorrect checkpoint cannot silently masquerade as a compact one.
        if expected_keys is not None:
            unexpected = set(state_dict) - expected_keys
            if unexpected:
                raise RuntimeError(
                    "Checkpoint has no language_classifier prefix and contains "
                    f"non-classifier keys: {sorted(unexpected)[:8]}"
                )
        extracted = dict(state_dict)

    if not extracted:
        raise RuntimeError("Checkpoint contains no language_classifier tensors")
    return extracted


def classifier_state_from_model_state(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract classifier tensors from a consolidated full-model state dict."""
    extracted = classifier_state_dict(state_dict)
    return {key: value.detach().cpu().contiguous() for key, value in extracted.items()}


def load_classifier_checkpoint(module: torch.nn.Module, path: str | Path) -> None:
    """Strictly restore ``module`` from any supported classifier checkpoint."""
    expected = set(module.state_dict())
    local_state = classifier_state_dict(load_state_dict_file(path), expected_keys=expected)
    module.load_state_dict(local_state, strict=True)


def save_classifier_checkpoint(
    state_dict: Mapping[str, torch.Tensor], path: str | Path, save_format: str
) -> Path:
    """Write only classifier tensors using the configured serialization format."""
    path = Path(path)
    local_state = classifier_state_from_model_state(state_dict)
    path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "safetensors":
        from safetensors.torch import save_file

        save_file(local_state, str(path))
    elif save_format == "pt":
        torch.save(local_state, path)
    else:
        raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
    return path


def classifier_checkpoint_path(directory: str | Path, stem: str, save_format: str) -> Path:
    suffix = ".safetensors" if save_format == "safetensors" else ".pt"
    if save_format not in {"pt", "safetensors"}:
        raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
    return Path(directory) / f"{stem}{suffix}"

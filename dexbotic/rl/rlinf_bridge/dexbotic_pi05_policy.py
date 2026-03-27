"""Pi05 RL policy for RLinf (placeholder until an RLinf-compatible policy is added)."""

from __future__ import annotations

from typing import Any, Optional

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype: Optional[Any] = None):
    raise NotImplementedError(
        "dexbotic_pi05 is registered in ModelRegistry but no RLinf Pi05 policy is wired yet. "
        "Implement loading (e.g. mirror dexbotic_pi) in this module when ready."
    )

"""CogAct RL policy for RLinf (placeholder until an RLinf-compatible policy is added)."""

from __future__ import annotations

from typing import Any, Optional

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype: Optional[Any] = None):
    raise NotImplementedError(
        "dexbotic_cogact is registered in ModelRegistry but no RLinf CogAct policy is wired yet. "
        "Implement loading in this module when ready."
    )

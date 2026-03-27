"""Register Dexbotic model loaders into RLinf ``ModelRegistry`` before training starts."""

from __future__ import annotations

from rlinf.models.registry import ModelRegistry


def _get_pi0_model(cfg, torch_dtype=None):
    from dexbotic.rl.rlinf_bridge.dexbotic_pi0_policy import get_model

    return get_model(cfg, torch_dtype)


def _get_dm0_model(cfg, torch_dtype=None):
    from dexbotic.rl.rlinf_bridge.dexbotic_dm0_policy import get_model

    return get_model(cfg, torch_dtype)


def _get_pi05_model(cfg, torch_dtype=None):
    from dexbotic.rl.rlinf_bridge.dexbotic_pi05_policy import get_model

    return get_model(cfg, torch_dtype)


def _get_cogact_model(cfg, torch_dtype=None):
    from dexbotic.rl.rlinf_bridge.dexbotic_cogact_policy import get_model

    return get_model(cfg, torch_dtype)


def register_all() -> None:
    """Register Dexbotic ``model_type`` strings with RLinf.

    Uses ``force=True`` so repeated calls or overlapping keys replace cleanly, and so
    Dexbotic loaders override any stale registry entries (including the legacy
    ``dexbotic_pi`` name used in RLinf YAML).
    """
    ModelRegistry.register("dexbotic_pi0", _get_pi0_model, force=True)
    ModelRegistry.register("dexbotic_pi", _get_pi0_model, force=True)
    ModelRegistry.register("dexbotic_dm0", _get_dm0_model, force=True)
    ModelRegistry.register("dexbotic_pi05", _get_pi05_model, force=True)
    ModelRegistry.register("dexbotic_cogact", _get_cogact_model, force=True)

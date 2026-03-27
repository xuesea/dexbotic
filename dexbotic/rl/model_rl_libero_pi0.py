"""RL training entry: Libero + Dexbotic Pi0 + PPO (cfg in ``dexbotic/config``)."""

from __future__ import annotations

# Register before any RLinf worker import (via _embodied_cli).
from dexbotic.rl.rlinf_registry import register_all

register_all()

import hydra

from dexbotic.rl._embodied_cli import run_embodied_rl


@hydra.main(
    version_base="1.1",
    config_path="../config",
    config_name="libero_pi0_ppo",
)
def main(cfg) -> None:
    run_embodied_rl(cfg)


if __name__ == "__main__":
    main()

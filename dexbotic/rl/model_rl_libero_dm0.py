"""RL training entry: Libero + Dexbotic DM0 + PPO."""

from __future__ import annotations

from dexbotic.rl.rlinf_registry import register_all

register_all()

import hydra

from dexbotic.rl._embodied_cli import run_embodied_rl


@hydra.main(
    version_base="1.1",
    config_path="../config",
    config_name="libero_dm0_ppo",
)
def main(cfg) -> None:
    run_embodied_rl(cfg)


if __name__ == "__main__":
    main()

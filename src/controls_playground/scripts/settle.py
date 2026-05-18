"""Settle the anchor keyframes by running zero-control physics steps and
print the result as YAML ready to paste back into configs/env.yaml.

Anchors with severe penetration may bounce rather than settle cleanly --
inspect the printed `torso_z` deltas before adopting the output.

  python -m controls_playground.settle --config configs/env.yaml --steps 50
"""
from __future__ import annotations
import argparse

import mujoco
import numpy as np

from mujoco_playground._src.dm_control_suite import common as _common
from mujoco_playground._src.dm_control_suite import humanoid as _hum

from ..env import INIT_NAMES, EnvConfig


def _build_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        _hum._XML_PATH.read_text(), _common.get_assets()
    )


def settle(qpos: np.ndarray, n_steps: int, model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(n_steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    return data.qpos.copy()


def _yaml_qpos(qpos: np.ndarray, indent: int = 6) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {v:+.6f}" for v in qpos)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/env.yaml")
    p.add_argument("--steps",  type=int, default=50,
                   help="physics steps with zero ctrl per anchor")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = EnvConfig.from_yaml(args.config)
    model = _build_model()

    print(f"# Settled anchors ({args.steps} mj_step with zero ctrl)")
    print(f"# Paste into {args.config} under `anchors:`\n")
    print("anchors:")
    for name in INIT_NAMES:
        if name == "STANDING":
            continue  # standing is model.qpos0, no need to settle
        key = name.lower()
        qpos0 = np.asarray(env_cfg.anchors[key])
        settled = settle(qpos0, args.steps, model)
        head_z_before = qpos0[2]  # torso z is a proxy for "is it standing"
        head_z_after  = settled[2]
        print(f"  {key}:    # torso_z {head_z_before:+.3f} -> {head_z_after:+.3f}")
        print(f"    qpos:")
        print(_yaml_qpos(settled))


if __name__ == "__main__":
    main()

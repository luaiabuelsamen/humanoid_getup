"""Plot the robustness sweep produced by motor_dr_eval.

Usage:
  python -m controls_playground.motor_dr_plot \\
      --ckpts v8=path/to/v8/ckpt.pkl v9=path/to/v9/ckpt.pkl \\
      --out analysis_out/motor_dr.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt

from ..env import EnvConfig
from ..motor_dr import MotorDRHumanoidGetUp
from .motor_dr_eval import SEVERITIES, rollout_summary
from ..policy import load_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="label=path pairs, e.g. v8=path1 v9=path2")
    p.add_argument("--out", type=str, default="analysis_out/motor_dr.png")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps",    type=int, default=400)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = EnvConfig.from_yaml("configs/env.yaml")

    severity_names = list(SEVERITIES.keys())
    x = np.arange(len(severity_names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for spec in args.ckpts:
        label, path = spec.split("=", 1)
        success = []
        wobble  = []
        for sev_name, motor_cfg in SEVERITIES.items():
            env = MotorDRHumanoidGetUp(env_cfg, motor_cfg=motor_cfg,
                                       mjx_config_overrides={"impl": "jax"})
            rng = jax.random.PRNGKey(0)
            obs_size = int(env.reset(rng).obs.shape[0])
            policy, _ = load_policy(path, obs_size, env.action_size)
            step_fn  = jax.jit(env.step)
            reset_fn = jax.jit(env.reset)

            summary = rollout_summary(env, policy, reset_fn, step_fn,
                                      args.episodes, args.steps)
            results = [r for init_list in summary.values() for r in init_list]
            success.append(np.mean([s for s, _, _ in results]))
            wobble.append(np.nanmean([w for _, w, _ in results]))
            print(f"[plot] {label:<10} {sev_name:<10} success={success[-1]:.2f}  "
                  f"wobble={wobble[-1]:.3f}")

        axes[0].plot(x, success, "o-", lw=2, ms=8, label=label)
        axes[1].plot(x, wobble,  "o-", lw=2, ms=8, label=label)

    for ax, ylabel, title in [
        (axes[0], "success rate", "Robustness: success vs motor DR severity"),
        (axes[1], "wobble  (|CoM_xy_v|)  [m/s]",
         "Stability: wobble vs motor DR severity"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(severity_names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylim(0, 1.05)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()

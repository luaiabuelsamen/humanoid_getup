"""Roll multiple checkpoints through HumanoidGetUp and report mean reward
components, broken down by init type.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict

import jax
import numpy as np

from ..env import EnvConfig, HumanoidGetUp, INIT_NAMES
from ..policy import load_policy


def rollout_metrics(ckpt: str, env, n_episodes: int, n_steps: int,
                    seed: int = 0) -> dict:
    rng = jax.random.PRNGKey(seed)
    obs_size = int(env.reset(rng).obs.shape[0])
    policy, _ = load_policy(ckpt, obs_size, env.action_size)

    step_fn  = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    per_init: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for _ in range(n_episodes):
        rng, reset_key = jax.random.split(rng)
        state = reset_fn(reset_key)
        idx = int(state.metrics["init/idx"])
        name = INIT_NAMES[idx]

        sums: dict[str, float] = defaultdict(float)
        for _ in range(n_steps):
            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            for k, v in state.metrics.items():
                if k.startswith("reward/"):
                    sums[k] += float(v)
            sums["total"] += float(state.reward)

        for k, v in sums.items():
            per_init[name][k].append(v)

    return per_init


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--steps",    type=int, default=400)
    p.add_argument("--seed",     type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    env_cfg = EnvConfig.from_yaml(args.env_config)
    env = HumanoidGetUp(env_cfg, mjx_config_overrides={"impl": "jax"})

    results = {}
    for ckpt in args.ckpts:
        name = Path(ckpt).parent.name
        print(f"[eval] rolling out {name}  ({args.episodes} episodes)")
        results[name] = rollout_metrics(ckpt, env, args.episodes, args.steps, args.seed)

    metric_keys = ["total", "reward/stand", "reward/head_bonus",
                   "reward/foot_flat", "reward/stillness"]

    print("\nper-init mean reward components (episode sums)")
    print(f"{'ckpt':<40} {'init':>10} {'n':>3}  " +
          "  ".join(f"{k.split('/')[-1]:>10}" for k in metric_keys))
    for name, per_init in results.items():
        for init_name in INIT_NAMES:
            if init_name not in per_init:
                continue
            episodes = per_init[init_name]
            n = len(next(iter(episodes.values())))
            means = [np.mean(episodes.get(k, [0.0])) for k in metric_keys]
            print(f"{name:<40} {init_name:>10} {n:>3}  " +
                  "  ".join(f"{m:>10.2f}" for m in means))

    print("\noverall mean (all inits weighted equally per ckpt)")
    for name, per_init in results.items():
        overall: dict[str, list[float]] = defaultdict(list)
        for init_name, episodes in per_init.items():
            for k in metric_keys:
                overall[k].extend(episodes.get(k, []))
        print(f"{name:<40} " + "  ".join(
            f"{k.split('/')[-1]}={np.mean(overall[k]):.2f}" for k in metric_keys))


if __name__ == "__main__":
    main()

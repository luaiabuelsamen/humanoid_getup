"""Empirical state-space landscape for the humanoid getup task.

For a 54-D nonlinear contact system there's no analytic "feasible region" or
"invariant set" to enumerate (unlike the linear MPC case). What you can do
is pick an interpretable 2D projection and show, on that slice:

  1. where the reward function lives  (sample a grid, compute reward)
  2. where the trained policy's rollouts travel  (overlay trajectories per
     init type)

The natural projection for this task is (head_z, torso_upright). Together
they answer 'is it standing?' -- standing is the corner (head_z ~ 1.5,
upright ~ 1.0).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import matplotlib.pyplot as plt

from ..env import EnvConfig, HumanoidGetUp, INIT_NAMES
from ..policy import load_policy


# ---------------------------------------------------------------- reward grid
def _reward_at(env, head_z_target: float, upright_target: float) -> float:
    """Sample the env reward at a synthetic state. We start from the standing
    qpos0 and only the head_z + torso orientation are altered, then we run
    one mj_forward to get xpos/xmat and call _get_reward directly.

    This isolates the reward landscape on the (head_z, upright) slice;
    everything else (joints, com vel) is held at the standing reference."""
    mj_model = env.mj_model
    qpos = np.asarray(mj_model.qpos0).copy()
    qpos[2] = head_z_target  # torso z (head height tracks this roughly)
    # Torso upright: rotate torso by angle alpha about world y. xmat[2,2] = cos(alpha).
    alpha = np.arccos(np.clip(upright_target, -1.0, 1.0))
    qpos[3:7] = [np.cos(alpha / 2), 0.0, np.sin(alpha / 2), 0.0]

    from mujoco import mjx as _mjx
    from mujoco_playground._src import mjx_env
    data = mjx_env.make_data(
        mj_model, qpos=jp.array(qpos),
        impl=env.mjx_model.impl.value,
        naconmax=env._config.naconmax, njmax=env._config.njmax,
    )
    data = _mjx.forward(env.mjx_model, data)
    metrics = {
        "reward/standing": jp.zeros(()), "reward/upright": jp.zeros(()),
        "reward/stand":    jp.zeros(()), "reward/small_control": jp.zeros(()),
        "reward/move":     jp.zeros(()), "reward/head_bonus": jp.zeros(()),
        "reward/foot_flat":jp.zeros(()), "reward/stillness": jp.zeros(()),
    }
    r = env._get_reward(data, jp.zeros(env.action_size), {}, metrics)
    return float(r)


def reward_grid(env, head_z_range: tuple, upright_range: tuple,
                n_h: int = 40, n_u: int = 40) -> tuple:
    hs = np.linspace(*head_z_range, n_h)
    us = np.linspace(*upright_range, n_u)
    grid = np.zeros((n_u, n_h))
    for i, u in enumerate(us):
        for j, h in enumerate(hs):
            grid[i, j] = _reward_at(env, h, u)
    return hs, us, grid


# ---------------------------------------------------------------- trajectories
def rollout_trace(env, policy, reset_fn, step_fn, rng,
                  n_steps: int) -> tuple[np.ndarray, np.ndarray, int]:
    rng, k = jax.random.split(rng)
    state = reset_fn(k)
    head_id = env._head_body_id
    torso_id = env._torso_body_id
    init_idx = int(state.metrics["init/idx"])

    hz = np.zeros(n_steps + 1)
    up = np.zeros(n_steps + 1)
    hz[0] = float(state.data.xpos[head_id, 2])
    up[0] = float(state.data.xmat[torso_id].reshape(3, 3)[2, 2])

    for t in range(n_steps):
        rng, ak = jax.random.split(rng)
        action, _ = policy(state.obs, ak)
        state = step_fn(state, action)
        hz[t + 1] = float(state.data.xpos[head_id, 2])
        up[t + 1] = float(state.data.xmat[torso_id].reshape(3, 3)[2, 2])
    return hz, up, init_idx


# ---------------------------------------------------------------- plot
def plot_landscape(hs: np.ndarray, us: np.ndarray, grid: np.ndarray,
                   traces: list[tuple[np.ndarray, np.ndarray, int]],
                   out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))

    pcm = ax.pcolormesh(hs, us, grid, shading="auto",
                        cmap="viridis", alpha=0.9)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("per-step reward at (head_z, upright) slice")

    # Standing target marker
    ax.axhline(1.0, color="w", ls=":", lw=0.7, alpha=0.5)
    ax.axvline(1.4, color="w", ls=":", lw=0.7, alpha=0.5)
    ax.scatter([1.4], [1.0], marker="*", s=200, color="w", edgecolors="k",
               linewidths=1.2, zorder=4, label="standing target")

    cmap = plt.get_cmap("tab10")
    seen = set()
    for hz, up, idx in traces:
        name = INIT_NAMES[idx]
        label = name if name not in seen else None
        seen.add(name)
        ax.plot(hz, up, "-", color=cmap(idx), lw=1.3, alpha=0.85, label=label)
        ax.scatter(hz[0], up[0], color=cmap(idx), s=40,
                   edgecolors="k", linewidths=0.6, zorder=3)
        ax.scatter(hz[-1], up[-1], color=cmap(idx), s=60, marker="X",
                   edgecolors="k", linewidths=0.6, zorder=3)

    ax.set_xlabel("head z [m]")
    ax.set_ylabel(r"torso upright projection  $\mathrm{xmat}[2,2]$")
    ax.set_xlim(hs.min(), hs.max())
    ax.set_ylim(us.min(), us.max())
    ax.set_title("Reward landscape on the (head_z, upright) slice\n"
                 "plus trained-policy rollouts from each init  "
                 "(circle = reset, X = end)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, required=True)
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--out",        type=str, default="analysis_out/landscape.png")
    p.add_argument("--episodes",   type=int, default=10)
    p.add_argument("--steps",      type=int, default=300)
    p.add_argument("--grid",       type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig.from_yaml(args.env_config)
    env = HumanoidGetUp(env_cfg, mjx_config_overrides={"impl": "jax"})

    rng = jax.random.PRNGKey(0)
    obs_size = int(env.reset(rng).obs.shape[0])
    policy, _ = load_policy(args.ckpt, obs_size, env.action_size)
    reset_fn = jax.jit(env.reset)
    step_fn  = jax.jit(env.step)

    print("computing reward grid ...")
    hs, us, grid = reward_grid(env, (0.05, 1.8), (-1.0, 1.0),
                                n_h=args.grid, n_u=args.grid)
    print("rolling out policy ...")
    traces = []
    for ep in range(args.episodes):
        rng, k = jax.random.split(rng)
        traces.append(rollout_trace(env, policy, reset_fn, step_fn, k, args.steps))
        idx = traces[-1][2]
        print(f"  ep {ep}  init={INIT_NAMES[idx]}  "
              f"end (head_z, upright) = ({traces[-1][0][-1]:.2f}, {traces[-1][1][-1]:.2f})")

    plot_landscape(hs, us, grid, traces, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

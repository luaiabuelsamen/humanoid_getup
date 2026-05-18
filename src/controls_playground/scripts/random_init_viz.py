"""Render per-episode clips for the truly-random-init stress test.

Same env as random_init.py (uniform SO(3) quat + random joint angles),
but produces an mp4 per episode so we can eyeball whether successes are
genuine getups or physics launches from bad initial configurations.

7-second real-time clips: ctrl_dt=0.025 means 280 env steps @ 40 fps = 7s.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import jax
import mujoco
import numpy as np

from ..env import EnvConfig
from ..policy import load_policy
from .random_init import make_random_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--out-dir", type=str, default="clips_random")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--steps",    type=int, default=280)   # 7 s @ ctrl_dt=0.025
    p.add_argument("--fps",      type=int, default=40)
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=480)
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args()

    import imageio
    env_cfg = EnvConfig.from_yaml(args.env_config)
    env = make_random_env(env_cfg)

    rng = jax.random.PRNGKey(args.seed)
    obs_size = int(env.reset(rng).obs.shape[0])
    policy, _ = load_policy(args.ckpt, obs_size, env.action_size)
    step_fn  = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    mj_model = env.mj_model
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    head_id  = env._head_body_id

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        rng, k = jax.random.split(rng)
        state = reset_fn(k)
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = np.asarray(state.data.qpos)
        mj_data.qvel[:] = np.asarray(state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        frames: list[np.ndarray] = []
        head_z_t: list[float] = []
        for _ in range(args.steps):
            rng, ak = jax.random.split(rng)
            action, _ = policy(state.obs, ak)
            state = step_fn(state, action)
            mj_data.qpos[:] = np.asarray(state.data.qpos)
            mj_data.qvel[:] = np.asarray(state.data.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=-1)
            frames.append(renderer.render())
            head_z_t.append(float(state.data.xpos[head_id, 2]))

        peak  = max(head_z_t)
        final = float(np.mean(head_z_t[-40:]))   # last ~1 s
        path = out_dir / f"ep{ep:02d}_peak{peak:.2f}_final{final:.2f}.mp4"
        imageio.mimsave(path, frames, fps=args.fps, codec="libx264", quality=8)
        print(f"[clip] {path.name}  (peak={peak:.2f}  final_mean={final:.2f})",
              flush=True)


if __name__ == "__main__":
    main()

"""Visualize a trained policy.

Three entrypoints:
- play_main   (cp-play):    interactive MuJoCo viewer
- render_main (cp-render):  single mp4 rendering N episodes back-to-back
- clips_main  (cp-clips):   per-episode clips (mp4 by default) to a directory

All three support --warmup-steps to run zero-ctrl mj_step's between reset and
policy start, so the initial-pose penetration resolves off-screen instead of
flashing as a wedge at the start of every clip.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import jax
import mujoco
import numpy as np

from ..env import EnvConfig, HumanoidGetUp, INIT_NAMES
from ..policy import load_policy


def _setup(ckpt: str, env_config: str, impl: str = "warp"):
    env_cfg = EnvConfig.from_yaml(env_config)
    env = HumanoidGetUp(env_cfg, mjx_config_overrides={"impl": impl})
    rng = jax.random.PRNGKey(0)
    obs_size = int(env.reset(rng).obs.shape[0])
    policy, _ = load_policy(ckpt, obs_size, env.action_size)
    return env, policy, jax.jit(env.reset), jax.jit(env.step), rng


def _apply_state_to_mjdata(mj_model, mj_data, state) -> None:
    mj_data.qpos[:] = np.asarray(state.data.qpos)
    mj_data.qvel[:] = np.asarray(state.data.qvel)
    mujoco.mj_forward(mj_model, mj_data)


def _warmup(mj_model, mj_data, n_steps: int) -> None:
    """Run zero-control physics steps so contact penetration resolves before
    the policy starts. Pure visual cleanup; doesn't affect what the policy sees
    at training time."""
    for _ in range(n_steps):
        mj_data.ctrl[:] = 0.0
        mujoco.mj_step(mj_model, mj_data)


def _init_name(state) -> str:
    idx = int(state.metrics.get("init/idx", -1))
    return INIT_NAMES[idx] if 0 <= idx < len(INIT_NAMES) else "?"


def play_main() -> None:
    import mujoco.viewer
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--frame-dt", type=float, default=0.025)
    args = p.parse_args()

    env, policy, reset_fn, step_fn, rng = _setup(args.ckpt, args.env_config, impl="jax")
    mj_model = env.mj_model
    mj_data  = mujoco.MjData(mj_model)

    first_rng, rng = jax.random.split(rng)
    state = reset_fn(first_rng)
    _apply_state_to_mjdata(mj_model, mj_data, state)
    _warmup(mj_model, mj_data, args.warmup_steps)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for ep in range(args.episodes):
            if ep > 0:
                rng, reset_key = jax.random.split(rng)
                state = reset_fn(reset_key)
                _apply_state_to_mjdata(mj_model, mj_data, state)
                _warmup(mj_model, mj_data, args.warmup_steps)
            viewer.sync()
            print(f"[play] ep {ep} init={_init_name(state)} torso_z={float(state.data.qpos[2]):.2f}")

            for _ in range(args.steps):
                if not viewer.is_running():
                    return
                rng, act_key = jax.random.split(rng)
                action, _ = policy(state.obs, act_key)
                state = step_fn(state, action)
                _apply_state_to_mjdata(mj_model, mj_data, state)
                viewer.sync()
                time.sleep(args.frame_dt)


def _rollout_frames(env, policy, reset_fn, step_fn, rng, renderer, mj_model,
                    n_steps: int, warmup_steps: int) -> tuple[list[np.ndarray], str]:
    rng, reset_key = jax.random.split(rng)
    state = reset_fn(reset_key)
    mj_data = mujoco.MjData(mj_model)
    _apply_state_to_mjdata(mj_model, mj_data, state)
    _warmup(mj_model, mj_data, warmup_steps)

    frames: list[np.ndarray] = []
    for _ in range(n_steps):
        rng, act_key = jax.random.split(rng)
        action, _ = policy(state.obs, act_key)
        state = step_fn(state, action)
        _apply_state_to_mjdata(mj_model, mj_data, state)
        renderer.update_scene(mj_data, camera=-1)
        frames.append(renderer.render())
    return frames, _init_name(state)


def render_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out",  type=str, default="")
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps",    type=int, default=600)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps",    type=int, default=40)
    args = p.parse_args()

    import imageio
    env, policy, reset_fn, step_fn, rng = _setup(args.ckpt, args.env_config)
    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "rollout.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(env.mj_model, height=args.height, width=args.width)
    all_frames: list[np.ndarray] = []
    for ep in range(args.episodes):
        rng, ep_rng = jax.random.split(rng)
        frames, init_name = _rollout_frames(env, policy, reset_fn, step_fn, ep_rng,
                                             renderer, env.mj_model,
                                             args.steps, args.warmup_steps)
        print(f"[render] ep {ep} init={init_name} ({len(frames)} frames)", flush=True)
        all_frames.extend(frames)

    print(f"[render] writing {out_path} ({len(all_frames)} frames @ {args.fps} fps)", flush=True)
    imageio.mimsave(out_path, all_frames, fps=args.fps)


def clips_main() -> None:
    """Produce N individual clips, one per episode, named by init type.
    Default format is mp4 (much smaller than gif at the same quality)."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="clips")
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    p.add_argument("--seed",   type=int, default=0)
    args = p.parse_args()

    import imageio
    env, policy, reset_fn, step_fn, _ = _setup(args.ckpt, args.env_config, impl="jax")
    rng = jax.random.PRNGKey(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.mj_model, height=args.height, width=args.width)

    for ep in range(args.episodes):
        rng, ep_rng = jax.random.split(rng)
        frames, init_name = _rollout_frames(env, policy, reset_fn, step_fn, ep_rng,
                                             renderer, env.mj_model,
                                             args.steps, args.warmup_steps)
        path = out_dir / f"{ep:02d}_{init_name}.{args.format}"
        if args.format == "gif":
            imageio.mimsave(path, frames, format="GIF",
                            duration=1.0 / args.fps, loop=0)
        else:
            imageio.mimsave(path, frames, fps=args.fps,
                            codec="libx264", quality=8)
        print(f"[clip] wrote {path}  ({len(frames)} frames)", flush=True)


# Backwards-compat alias for the older `cp-gifs` console script.
gifs_main = clips_main


if __name__ == "__main__":
    import sys
    cmd = "play"
    if len(sys.argv) > 1 and sys.argv[1] in ("render", "play", "clips", "gifs"):
        cmd = sys.argv.pop(1)
    {"play": play_main, "render": render_main,
     "clips": clips_main, "gifs": gifs_main}[cmd]()

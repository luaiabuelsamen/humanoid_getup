"""Out-of-distribution probe: run a trained policy from noisier init poses."""
from __future__ import annotations
import argparse
import time

import jax
import jax.numpy as jp
import mujoco

from mujoco import mjx as _mjx
from mujoco_playground._src import mjx_env

from ..env import EnvConfig, HumanoidGetUp, INIT_NAMES
from ..policy import load_policy


STAND_HEAD_Z = 1.3


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jp.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _random_yaw_quat(rng, yaw_range):
    yaw = jax.random.uniform(rng, (), minval=-yaw_range, maxval=yaw_range)
    return jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])


def make_noisy_env(env_cfg: EnvConfig,
                   joint_jitter: float, yaw_range: float, pos_jitter_xy: float):
    class HumanoidGetUpRandom(HumanoidGetUp):
        def reset(self, rng):
            rng, branch_key, jit_key, pos_key, yaw_key = jax.random.split(rng, 5)

            init_idx = jax.random.randint(branch_key, (), 0, self._anchor_qpos.shape[0])
            qpos = self._anchor_qpos[init_idx]

            jitter = jax.random.uniform(jit_key, (qpos.shape[0] - 7,),
                                        minval=-joint_jitter, maxval=joint_jitter)
            qpos = qpos.at[7:].add(jitter)

            pos_jit = jax.random.uniform(pos_key, (2,),
                                         minval=-pos_jitter_xy, maxval=pos_jitter_xy)
            qpos = qpos.at[0:2].add(pos_jit)

            base_quat = qpos[3:7]
            new_quat = _quat_mul(_random_yaw_quat(yaw_key, yaw_range), base_quat)
            qpos = qpos.at[3:7].set(new_quat)

            data = mjx_env.make_data(self.mj_model, qpos=qpos,
                                     impl=self.mjx_model.impl.value,
                                     naconmax=self._config.naconmax,
                                     njmax=self._config.njmax)
            data = _mjx.forward(self.mjx_model, data)

            metrics = {
                "reward/standing":      jp.zeros(()),
                "reward/upright":       jp.zeros(()),
                "reward/stand":         jp.zeros(()),
                "reward/small_control": jp.zeros(()),
                "reward/move":          jp.zeros(()),
                "reward/head_bonus":    jp.zeros(()),
                "init/idx":             init_idx.astype(jp.float32),
            }
            obs = self._get_obs(data, {"rng": rng})
            return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, {"rng": rng})

    return HumanoidGetUpRandom(env_cfg, mjx_config_overrides={"impl": "jax"})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--joint-jitter", type=float, default=0.30)
    p.add_argument("--yaw-range",    type=float, default=3.14159)
    p.add_argument("--pos-jitter",   type=float, default=0.10)
    p.add_argument("--no-viewer", action="store_true",
                   help="headless mode (no MuJoCo passive viewer)")
    args = p.parse_args()

    env_cfg = EnvConfig.from_yaml(args.env_config)
    env = make_noisy_env(env_cfg, args.joint_jitter, args.yaw_range, args.pos_jitter)

    rng = jax.random.PRNGKey(0)
    obs_size = int(env.reset(rng).obs.shape[0])
    policy, _ = load_policy(args.ckpt, obs_size, env.action_size)

    step_fn  = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    mj_model = env.mj_model
    mj_data  = mujoco.MjData(mj_model)

    viewer_ctx = None
    if not args.no_viewer:
        import mujoco.viewer
        viewer_ctx = mujoco.viewer.launch_passive(mj_model, mj_data)

    try:
        stood_count = 0
        for ep in range(args.episodes):
            rng, reset_key = jax.random.split(rng)
            state = reset_fn(reset_key)
            mj_data.qpos[:] = state.data.qpos
            mj_data.qvel[:] = state.data.qvel
            mujoco.mj_forward(mj_model, mj_data)
            if viewer_ctx is not None:
                viewer_ctx.sync()

            idx = int(state.metrics.get("init/idx", -1))
            name = INIT_NAMES[idx] if 0 <= idx < len(INIT_NAMES) else "?"
            head_max = float(state.data.xpos[env._head_body_id, 2])
            print(f"[probe] ep {ep} base={name} torso_z={float(state.data.qpos[2]):.2f}")

            for _ in range(args.steps):
                if viewer_ctx is not None and not viewer_ctx.is_running():
                    return
                rng, act_key = jax.random.split(rng)
                action, _ = policy(state.obs, act_key)
                state = step_fn(state, action)
                mj_data.qpos[:] = state.data.qpos
                mj_data.qvel[:] = state.data.qvel
                mujoco.mj_forward(mj_model, mj_data)
                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    time.sleep(0.025)
                head_max = max(head_max, float(state.data.xpos[env._head_body_id, 2]))

            stood = head_max > STAND_HEAD_Z
            stood_count += int(stood)
            print(f"        peak head_z={head_max:.2f}  -- {'STOOD' if stood else 'failed'}")

        print(f"\n[probe] {stood_count}/{args.episodes} stood "
              f"(threshold head_z > {STAND_HEAD_Z})")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()


if __name__ == "__main__":
    main()

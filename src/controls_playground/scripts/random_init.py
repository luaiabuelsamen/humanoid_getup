"""True-random-init stress test.

Samples truly random init poses (not perturbations of the 5 anchor keyframes
that the policy was trained on), runs a trained policy, reports success
rate and peak head height. Directly tests whether the multi-pose-anchor
training implicitly generalizes to arbitrary configurations.

Sampling:
  - root orientation: uniform on SO(3) (random unit quaternion)
  - joint angles:     uniform on each joint's [lo, hi] (from model.jnt_range)
  - root xy:          uniform small offset
  - root z:           sampled in [0.10, 0.40] (lying / sitting region) so the
                      body always starts below standing
  - velocities:       zero
After setting qpos, runs a few zero-control physics steps to resolve any
penetration before the policy takes over (same trick as the play-time
warmup in viz.py).
"""
from __future__ import annotations
import argparse

import jax
import jax.numpy as jp
import numpy as np
import mujoco

from mujoco import mjx as _mjx
from mujoco_playground._src import mjx_env

from ..env import EnvConfig, HumanoidGetUp
from ..policy import load_policy


STAND_HEAD_Z = 1.3


def _random_so3_quat(rng: jax.Array) -> jax.Array:
    """Uniform random unit quaternion -- samples uniformly on SO(3)."""
    v = jax.random.normal(rng, (4,))
    return v / jp.linalg.norm(v)


def _random_qpos(rng: jax.Array, model: mujoco.MjModel,
                 z_range=(0.10, 0.40), xy_range=0.15) -> jax.Array:
    rng, qk, zk, xyk = jax.random.split(rng, 4)
    quat = _random_so3_quat(qk)
    z = jax.random.uniform(zk, (), minval=z_range[0], maxval=z_range[1])
    xy = jax.random.uniform(xyk, (2,), minval=-xy_range, maxval=xy_range)
    # Joint angles: uniform within each joint's range. The freejoint occupies
    # the first 7 qpos entries; jnt_range covers hinge joints only.
    # Get jnt_range for hinge joints (n_joints = nq - 7).
    n_hinge = model.nq - 7
    rng, jk = jax.random.split(rng)
    # jnt_range[1:] skips the freejoint (first joint). The remaining joints
    # are hinges. Use the model's actual range; clamp wide ranges to ±1.6
    # rad (no-one's elbow goes a full revolution in our scenarios).
    lo = jp.array(np.clip(model.jnt_range[1:, 0], -1.6, 0.0))
    hi = jp.array(np.clip(model.jnt_range[1:, 1],  0.0, 1.6))
    joint_q = jax.random.uniform(jk, (n_hinge,), minval=lo, maxval=hi)

    qpos = jp.concatenate([xy, jp.array([z]), quat, joint_q])
    return qpos


def make_random_env(env_cfg: EnvConfig, settle_steps: int = 5):
    """HumanoidGetUp subclass with truly random reset."""

    class HumanoidGetUpRandom(HumanoidGetUp):
        def reset(self, rng):
            rng, q_key = jax.random.split(rng)
            qpos = _random_qpos(q_key, self.mj_model)

            data = mjx_env.make_data(
                self.mj_model, qpos=qpos,
                impl=self.mjx_model.impl.value,
                naconmax=self._config.naconmax,
                njmax=self._config.njmax,
            )
            data = _mjx.forward(self.mjx_model, data)
            # Run a few zero-ctrl steps to resolve any penetration.
            for _ in range(settle_steps):
                data = mjx_env.step(self.mjx_model, data, jp.zeros(self.action_size),
                                    self.n_substeps)

            metrics = {
                "reward/standing":      jp.zeros(()),
                "reward/upright":       jp.zeros(()),
                "reward/stand":         jp.zeros(()),
                "reward/small_control": jp.zeros(()),
                "reward/move":          jp.zeros(()),
                "reward/head_bonus":    jp.zeros(()),
                "reward/foot_flat":     jp.zeros(()),
                "reward/stillness":     jp.zeros(()),
                "init/idx":             jp.array(-1.0),  # random, no anchor index
            }
            obs = self._get_obs(data, {"rng": rng})
            return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()),
                                  metrics, {"rng": rng})

    return HumanoidGetUpRandom(env_cfg, mjx_config_overrides={"impl": "jax"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps",    type=int, default=400)
    p.add_argument("--seed",     type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = EnvConfig.from_yaml(args.env_config)
    env = make_random_env(env_cfg)

    rng = jax.random.PRNGKey(args.seed)
    obs_size = int(env.reset(rng).obs.shape[0])
    policy, _ = load_policy(args.ckpt, obs_size, env.action_size)
    step_fn  = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    head_id = env._head_body_id
    results: list[tuple[bool, float, float]] = []

    for ep in range(args.episodes):
        rng, k = jax.random.split(rng)
        state = reset_fn(k)
        head_z0 = float(state.data.xpos[head_id, 2])
        peak_head_z = head_z0

        for _ in range(args.steps):
            rng, ak = jax.random.split(rng)
            action, _ = policy(state.obs, ak)
            state = step_fn(state, action)
            head_z = float(state.data.xpos[head_id, 2])
            peak_head_z = max(peak_head_z, head_z)

        stood = peak_head_z > STAND_HEAD_Z
        results.append((stood, head_z0, peak_head_z))
        if ep % 10 == 0:
            print(f"  ep {ep}: head_z0={head_z0:.2f}  peak={peak_head_z:.2f}  "
                  f"{'STOOD' if stood else 'failed'}", flush=True)

    n = len(results)
    n_stood = sum(s for s, _, _ in results)
    peak_avg = float(np.mean([p for _, _, p in results]))
    # Wilson 95% CI for binomial
    p_hat = n_stood / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    print(f"\n[random-init] {n_stood}/{n} stood (peak head_z > {STAND_HEAD_Z})")
    print(f"              success = {p_hat:.2f}  Wilson 95% CI ≈ "
          f"[{max(0, center - half):.2f}, {min(1, center + half):.2f}]")
    print(f"              mean peak head_z = {peak_avg:.2f}")


if __name__ == "__main__":
    main()

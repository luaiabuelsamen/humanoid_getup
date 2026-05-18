"""Evaluate a trained policy under motor-model domain randomization.

Sweeps DR severity. For each severity level:
  - sample n_episodes init poses
  - sample new motor params at each reset
  - roll the policy
  - record: stood (peak head_z > 1.3), mean wobble while standing

Output: a table comparing nominal vs mild vs strong DR and the per-init
breakdown.
"""
from __future__ import annotations
import argparse

import jax
import jax.numpy as jp
import numpy as np

from ..env import EnvConfig, INIT_NAMES
from ..motor_dr import MotorDRHumanoidGetUp, MotorDRConfig
from ..policy import load_policy


SEVERITIES = {
    "nominal": MotorDRConfig(gain_range=(1.0, 1.0), tau_range=(0.005, 0.005),
                              umax_range=(1.0, 1.0)),
    "mild":    MotorDRConfig(gain_range=(0.85, 1.15), tau_range=(0.005, 0.025),
                              umax_range=(0.8, 1.2)),
    "strong":  MotorDRConfig(gain_range=(0.7, 1.3),  tau_range=(0.005, 0.04),
                              umax_range=(0.6, 1.4)),
    "extreme": MotorDRConfig(gain_range=(0.5, 1.5),  tau_range=(0.005, 0.08),
                              umax_range=(0.4, 1.6)),
}


def rollout_summary(env, policy, reset_fn, step_fn, n_episodes: int,
                    n_steps: int, seed: int = 0) -> dict:
    rng = jax.random.PRNGKey(seed)
    head_id = env._head_body_id
    by_init: dict[str, list] = {n: [] for n in INIT_NAMES}

    for _ in range(n_episodes):
        rng, k = jax.random.split(rng)
        state = reset_fn(k)
        init_name = INIT_NAMES[int(state.metrics["init/idx"])]

        peak_head_z = float(state.data.xpos[head_id, 2])
        wobble_samples = []
        for _ in range(n_steps):
            rng, ak = jax.random.split(rng)
            action, _ = policy(state.obs, ak)
            state = step_fn(state, action)
            head_z = float(state.data.xpos[head_id, 2])
            peak_head_z = max(peak_head_z, head_z)
            if head_z > 1.3:
                cv = np.asarray(env._center_of_mass_velocity(state.data))
                wobble_samples.append(float(np.linalg.norm(cv[:2])))

        stood = peak_head_z > 1.3
        wobble = float(np.mean(wobble_samples)) if wobble_samples else float("nan")
        by_init[init_name].append((stood, wobble, peak_head_z))

    return by_init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/humanoid_getup/getup-v8-long/ckpt_000412876800.pkl")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps",    type=int, default=400)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = EnvConfig.from_yaml("configs/env.yaml")

    rows = []
    for sev_name, motor_cfg in SEVERITIES.items():
        env = MotorDRHumanoidGetUp(env_cfg, motor_cfg=motor_cfg,
                                   mjx_config_overrides={"impl": "jax"})
        rng = jax.random.PRNGKey(0)
        obs_size = int(env.reset(rng).obs.shape[0])
        policy, _ = load_policy(args.ckpt, obs_size, env.action_size)
        step_fn  = jax.jit(env.step)
        reset_fn = jax.jit(env.reset)
        print(f"[motor-dr] severity={sev_name}: gain in {motor_cfg.gain_range}, "
              f"tau in {motor_cfg.tau_range}, u_max in {motor_cfg.umax_range}")
        summary = rollout_summary(env, policy, reset_fn, step_fn,
                                  args.episodes, args.steps)

        all_results = [r for init_list in summary.values() for r in init_list]
        success = np.mean([s for s, w, p in all_results])
        peak    = np.mean([p for s, w, p in all_results])
        wobble  = np.nanmean([w for s, w, p in all_results])
        rows.append((sev_name, success, peak, wobble, len(all_results)))

        for init_name, results in summary.items():
            if not results:
                continue
            s_rate = np.mean([s for s, _, _ in results])
            p_avg  = np.mean([p for _, _, p in results])
            print(f"           {init_name:>9}: n={len(results):>2}  "
                  f"success={s_rate:.2f}  peak_head_z={p_avg:.2f}")
        print()

    print("=== overall summary ===")
    print(f"{'severity':<10} {'success':>8} {'peak_head_z':>13} {'wobble':>8} {'n':>4}")
    for name, s, p, w, n in rows:
        print(f"{name:<10} {s:>8.2f} {p:>13.2f} {w:>8.3f} {n:>4}")


if __name__ == "__main__":
    main()

"""Hybrid LQR + RL controller.

Switching rule:
  state in standing region  ->  u = ctrl_eq - K @ (state - state_eq)   (LQR)
  state elsewhere           ->  u = pi_RL(obs)                          (PPO policy)

The standing region is defined by a head-height and torso-upright threshold
plus a velocity gate; hysteresis avoids chatter at the boundary.

Run:
  python -m controls_playground.hybrid --ckpt ... --episodes 5 --render
"""
from __future__ import annotations
import argparse
from pathlib import Path

import jax
import numpy as np
import mujoco
import scipy.linalg

from ..env import EnvConfig, HumanoidGetUp, INIT_NAMES
from ..policy import load_policy
from . import lqr as _lqr


# Switching thresholds (with hysteresis -> [enter, exit]).
HEAD_Z_BAND   = (1.30, 1.20)
UPRIGHT_BAND  = (0.92, 0.85)
QVEL_BAND     = (3.0,  6.0)


def _state_diff(qpos: np.ndarray, qpos_eq: np.ndarray,
                qvel: np.ndarray, qvel_eq: np.ndarray,
                ignore_root_xy: bool = True) -> np.ndarray:
    """State error in linearization tangent space.
    - Root xy position is not directly actuatable, so we zero it out
      (otherwise LQR tries to push the floating base laterally with
      joint torques and just shakes the body).
    - Root z position kept.
    - Root quaternion: small-angle subtraction -> rotation vector.
    - Joint angles subtract directly.
    - All generalized velocities included.
    """
    nv = qvel.size
    err = np.zeros(2 * nv)

    if not ignore_root_xy:
        err[0:2] = qpos[0:2] - qpos_eq[0:2]
    err[2] = qpos[2] - qpos_eq[2]

    q_cur = qpos[3:7]
    q_eq  = qpos_eq[3:7]
    q_eq_conj = np.array([q_eq[0], -q_eq[1], -q_eq[2], -q_eq[3]])
    q_err = np.array([
        q_cur[0]*q_eq_conj[0] - q_cur[1]*q_eq_conj[1] - q_cur[2]*q_eq_conj[2] - q_cur[3]*q_eq_conj[3],
        q_cur[0]*q_eq_conj[1] + q_cur[1]*q_eq_conj[0] + q_cur[2]*q_eq_conj[3] - q_cur[3]*q_eq_conj[2],
        q_cur[0]*q_eq_conj[2] - q_cur[1]*q_eq_conj[3] + q_cur[2]*q_eq_conj[0] + q_cur[3]*q_eq_conj[1],
        q_cur[0]*q_eq_conj[3] + q_cur[1]*q_eq_conj[2] - q_cur[2]*q_eq_conj[1] + q_cur[3]*q_eq_conj[0],
    ])
    if q_err[0] < 0:
        q_err = -q_err
    err[3:6] = 2.0 * q_err[1:4]

    err[6:nv] = qpos[7:] - qpos_eq[7:]
    err[nv:] = qvel - qvel_eq
    return err


class HybridController:
    """Three modes (selected at construction):
      - 'switch'    hard-switch LQR/RL with hysteresis (the original try)
      - 'additive'  u = u_RL + alpha(state) * (-K @ state_err)   (gentle nudge)
      - 'damping'   u = u_RL + alpha(state) * (-K_vel @ qvel)    (rate feedback)
    For 'additive' and 'damping' the LQR contribution is gated smoothly by
    alpha(state) which is 1 deep inside the standing region and decays to 0
    outside.
    """

    def __init__(self, K: np.ndarray, ctrl_eq: np.ndarray,
                 qpos_eq: np.ndarray, qvel_eq: np.ndarray,
                 action_size: int, nv: int, mode: str = "switch",
                 alpha_max: float = 0.2):
        self.K = K
        self.ctrl_eq = ctrl_eq
        self.qpos_eq = qpos_eq
        self.qvel_eq = qvel_eq
        self.action_size = action_size
        self.nv = nv
        self.mode = mode
        self.alpha_max = alpha_max
        self._in_lqr = False                      # for 'switch'
        self.mode_log: list[str] = []

    def _switch_to_lqr(self, head_z: float, upright: float, qvel_norm: float
                       ) -> bool:
        enter = (head_z > HEAD_Z_BAND[0] and upright > UPRIGHT_BAND[0]
                 and qvel_norm < QVEL_BAND[0])
        exit_ = (head_z < HEAD_Z_BAND[1] or upright < UPRIGHT_BAND[1]
                 or qvel_norm > QVEL_BAND[1])
        if not self._in_lqr and enter:
            self._in_lqr = True
        elif self._in_lqr and exit_:
            self._in_lqr = False
        return self._in_lqr

    def _alpha(self, head_z: float, upright: float, qvel_norm: float) -> float:
        """Smooth gate: 0 outside the standing band, ramps to alpha_max inside."""
        h = np.clip((head_z - 1.20) / 0.10, 0.0, 1.0)
        u = np.clip((upright - 0.85) / 0.10, 0.0, 1.0)
        v = np.clip((4.0 - qvel_norm) / 2.0, 0.0, 1.0)
        return self.alpha_max * h * u * v

    def action(self, head_z: float, upright: float,
               qpos: np.ndarray, qvel: np.ndarray,
               rl_action: np.ndarray) -> np.ndarray:
        qvel_norm = float(np.linalg.norm(qvel))

        if self.mode == "switch":
            if self._switch_to_lqr(head_z, upright, qvel_norm):
                err = _state_diff(qpos, self.qpos_eq, qvel, self.qvel_eq)
                u = self.ctrl_eq - self.K @ err
                self.mode_log.append("LQR")
                return np.clip(u, -1.0, 1.0)
            self.mode_log.append("RL")
            return np.asarray(rl_action)

        alpha = self._alpha(head_z, upright, qvel_norm)
        self.mode_log.append(f"a={alpha:.2f}")

        if self.mode == "additive":
            err = _state_diff(qpos, self.qpos_eq, qvel, self.qvel_eq)
            correction = -self.K @ err
        elif self.mode == "damping":
            # Use only the velocity columns of K -> pure rate feedback toward
            # qvel_eq. Treats LQR as a damper, not a regulator.
            K_v = self.K[:, self.nv:]
            correction = -K_v @ (qvel - self.qvel_eq)
        else:
            raise ValueError(self.mode)

        u = np.asarray(rl_action) + alpha * correction
        return np.clip(u, -1.0, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/humanoid_getup/getup-v8-long/ckpt_000412876800.pkl")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--steps",    type=int, default=400)
    p.add_argument("--out-dir",  type=str, default="analysis_out/hybrid")
    p.add_argument("--render",   action="store_true")
    p.add_argument("--q-pose-weight", type=float, default=10.0)
    p.add_argument("--q-vel-weight",  type=float, default=1.0)
    p.add_argument("--r-weight",      type=float, default=0.01)
    p.add_argument("--modes", nargs="+",
                   default=["switch", "additive", "damping"],
                   choices=("switch", "additive", "damping"))
    p.add_argument("--alpha-max", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. LQR design at the policy's operating point.
    model = _lqr.load_playground_humanoid()
    nv = model.nv
    nu = model.nu

    qpos_eq, qvel_eq, ctrl_eq = _lqr.find_operating_point(model, args.ckpt)
    A, B = _lqr.linearize(model, qpos_eq, qvel_eq, ctrl_eq)
    Q = np.eye(2 * nv)
    Q[:nv, :nv] *= args.q_pose_weight
    Q[nv:, nv:] *= args.q_vel_weight
    R = args.r_weight * np.eye(nu)
    K, _ = _lqr.lqr_gain(A, B, Q, R)
    eig_cl = np.linalg.eigvals(A - B @ K)
    print(f"[hybrid] LQR closed-loop max |eig| = {np.abs(eig_cl).max():.4f}")

    # 2. Env + RL policy
    env_cfg = EnvConfig.from_yaml("configs/env.yaml")
    env = HumanoidGetUp(env_cfg, mjx_config_overrides={"impl": "jax"})
    rng = jax.random.PRNGKey(0)
    obs_size = int(env.reset(rng).obs.shape[0])
    rl_policy, _ = load_policy(args.ckpt, obs_size, env.action_size)

    step_fn  = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    head_id = env._head_body_id
    torso_id = env._torso_body_id

    def torso_upright(state) -> float:
        return float(state.data.xmat[torso_id, 2, 2])

    # 3. Per-policy rollouts. Always run RL-only + each requested mode.
    import jax.numpy as jp
    summary = []
    labels = ["RL-only"] + [f"Hybrid({m})" for m in args.modes]
    for label in labels:
        mode = label.replace("Hybrid(", "").rstrip(")") if label != "RL-only" else None
        wobble_per_ep = []
        rng2 = jax.random.PRNGKey(0)
        for ep in range(args.episodes):
            rng2, reset_key = jax.random.split(rng2)
            state = reset_fn(reset_key)
            controller = HybridController(K, ctrl_eq, qpos_eq, qvel_eq,
                                          env.action_size, nv,
                                          mode=mode or "switch",
                                          alpha_max=args.alpha_max)

            wobble_samples = []
            for _ in range(args.steps):
                rng2, ak = jax.random.split(rng2)
                rl_action, _ = rl_policy(state.obs, ak)
                rl_action_np = np.asarray(rl_action)

                if mode is not None:
                    u = controller.action(
                        head_z=float(state.data.xpos[head_id, 2]),
                        upright=torso_upright(state),
                        qpos=np.asarray(state.data.qpos),
                        qvel=np.asarray(state.data.qvel),
                        rl_action=rl_action_np,
                    )
                else:
                    u = rl_action_np

                state = step_fn(state, jp.asarray(u))

                head_z = float(state.data.xpos[head_id, 2])
                if head_z > 1.3:
                    com_vel = np.asarray(env._center_of_mass_velocity(state.data))
                    wobble_samples.append(float(np.linalg.norm(com_vel[:2])))

            wobble = float(np.mean(wobble_samples)) if wobble_samples else float("nan")
            wobble_per_ep.append(wobble)
        summary.append((label, np.nanmean(wobble_per_ep)))
        print(f"[hybrid] {label:<22}  mean wobble = {summary[-1][1]:.3f}")

    print("\n=== summary (mean |CoM_xy_v| while standing) ===")
    for label, w in summary:
        delta = ""
        if label != "RL-only":
            delta = f"  ({w / summary[0][1]:+.2f}x vs RL-only)"
        print(f"  {label:<22}  {w:.3f}{delta}")


if __name__ == "__main__":
    main()

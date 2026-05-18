"""LQR design around the standing equilibrium of the dm_control humanoid.

  1. Solve for ctrl_eq: actuator commands that produce ~zero joint acceleration
     at the standing keyframe with qvel=0 (the freejoint residual is left to
     contact forces).
  2. Linearize the discrete dynamics around (q_stand, 0, ctrl_eq) via
     mjd_transitionFD.
  3. Solve the discrete algebraic Riccati equation for the infinite-horizon
     LQR gain K with diagonal Q, R weights.
  4. Print closed-loop eigenvalues of A - B K.

Run:
  python -m controls_playground.lqr --keyframe stand_on_left_leg
"""
from __future__ import annotations
import argparse

import numpy as np
import mujoco
import scipy.linalg
from scipy.optimize import least_squares

from mujoco_playground._src.dm_control_suite import common as _dmcs_common
from mujoco_playground._src.dm_control_suite import humanoid as _dmcs_hum


def load_playground_humanoid() -> mujoco.MjModel:
    """The exact humanoid XML the trained RL policy uses."""
    return mujoco.MjModel.from_xml_string(
        _dmcs_hum._XML_PATH.read_text(), _dmcs_common.get_assets()
    )


def find_operating_point(model: mujoco.MjModel, ckpt_path: str,
                         n_warmup_steps: int = 800
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the trained RL policy from standing init, return the (qpos,
    qvel, ctrl) at the step where |qvel| is smallest in the standing region.
    This is the policy's empirical operating point - linearizing here gives
    LQR a sensible local stabilization target."""
    import jax
    from .env import EnvConfig, HumanoidGetUp
    from .policy import load_policy

    env_cfg = EnvConfig.from_yaml("configs/env.yaml")
    env = HumanoidGetUp(env_cfg, mjx_config_overrides={"impl": "jax"})

    obs_size = int(env.reset(jax.random.PRNGKey(0)).obs.shape[0])
    policy, _ = load_policy(ckpt_path, obs_size, env.action_size)
    step_fn  = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    rng = jax.random.PRNGKey(0)
    state = reset_fn(rng)
    while int(state.metrics["init/idx"]) != 0:
        rng, k = jax.random.split(rng)
        state = reset_fn(k)

    best = None
    for _ in range(n_warmup_steps):
        rng, ak = jax.random.split(rng)
        action, _ = policy(state.obs, ak)
        state = step_fn(state, action)
        head_z = float(state.data.xpos[env._head_body_id, 2])
        if head_z < 1.3:
            continue
        qvel_norm = float(np.linalg.norm(state.data.qvel))
        if best is None or qvel_norm < best[0]:
            best = (qvel_norm, np.asarray(state.data.qpos),
                    np.asarray(state.data.qvel), np.asarray(action))

    if best is None:
        raise RuntimeError("policy never reached standing region in warmup")
    print(f"[lqr] operating point: |qvel| = {best[0]:.3f},  "
          f"torso_z = {best[1][2]:.3f}")
    return best[1], best[2], best[3]


def find_ctrl_eq(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray
                 ) -> tuple[np.ndarray, float]:
    """Solve for ctrl that makes actuated-joint accelerations zero at (qpos, qvel).

    Uses scipy.optimize.least_squares with a finite-difference Jacobian and a
    trust-region solver so it actually moves off the zero initial guess on
    this stiff, contact-rich system.
    """
    data = mujoco.MjData(model)

    def residual(ctrl: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        return data.qacc[6:]      # only the actuated DOFs

    res = least_squares(
        residual, np.zeros(model.nu),
        method="trf", max_nfev=2000, diff_step=1e-3, x_scale="jac",
    )
    return res.x, float(np.linalg.norm(res.fun))


def linearize(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray,
              ctrl: np.ndarray, eps: float = 1e-6
              ) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    nv = model.nv
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, model.nu))
    mujoco.mjd_transitionFD(model, data, eps, True, A, B, None, None)
    return A, B


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Discrete infinite-horizon LQR: solve DARE, return (K, P)."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/humanoid_getup/getup-v8-long/ckpt_000412876800.pkl",
                   help="RL policy to use as the operating-point oracle "
                        "(--mode policy)")
    p.add_argument("--mode", choices=("qpos0", "policy"), default="policy",
                   help="qpos0: linearize at model.qpos0 + ctrl_eq via inverse "
                        "dynamics. policy: linearize at the state the RL "
                        "policy actually visits while standing.")
    p.add_argument("--q-pose-weight",  type=float, default=10.0)
    p.add_argument("--q-vel-weight",   type=float, default=1.0)
    p.add_argument("--r-weight",       type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = load_playground_humanoid()
    nv = model.nv
    nu = model.nu

    if args.mode == "policy":
        qpos0, qvel0, ctrl_eq = find_operating_point(model, args.ckpt)
        residual_norm = None
    else:
        qpos0 = model.qpos0.copy()
        qvel0 = np.zeros(nv)
        ctrl_eq, residual_norm = find_ctrl_eq(model, qpos0, qvel0)
        print(f"[lqr] ctrl_eq solved: ||qacc[6:]|| residual = "
              f"{residual_norm:.3e}")

    print(f"[lqr] operating torso_z = {qpos0[2]:.3f}, "
          f"|qvel| = {np.linalg.norm(qvel0):.3f}, "
          f"||ctrl_eq|| = {np.linalg.norm(ctrl_eq):.3f}")

    # 50-step open-loop drift check (apply constant ctrl_eq)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.ctrl[:] = ctrl_eq
    for _ in range(50):
        mujoco.mj_step(model, data)
    drift_z = abs(float(data.qpos[2]) - qpos0[2])
    drift_v = float(np.linalg.norm(data.qvel))
    print(f"[lqr] 50-step open-loop drift at ctrl_eq: |dz|={drift_z:.3f}, "
          f"|qvel|={drift_v:.3f}")

    # 2. Linearize
    A, B = linearize(model, qpos0, qvel0, ctrl_eq)
    eig_ol = np.linalg.eigvals(A)
    n_unstable = int((np.abs(eig_ol) > 1.0 + 1e-6).sum())
    print(f"[lqr] open-loop max |eig| = {np.abs(eig_ol).max():.4f}, "
          f"unstable modes = {n_unstable} / {2*nv}")

    # 3. LQR weights and DARE
    Q = np.eye(2 * nv)
    Q[:nv, :nv]     *= args.q_pose_weight
    Q[nv:, nv:]     *= args.q_vel_weight
    R = args.r_weight * np.eye(nu)

    try:
        K, _ = lqr_gain(A, B, Q, R)
    except Exception as e:
        print(f"[lqr] DARE solve failed: {e}")
        return

    eig_cl = np.linalg.eigvals(A - B @ K)
    n_unstable_cl = int((np.abs(eig_cl) > 1.0 + 1e-6).sum())
    print(f"[lqr] closed-loop max |eig| = {np.abs(eig_cl).max():.4f}, "
          f"unstable modes = {n_unstable_cl} / {2*nv}")
    print(f"[lqr] K shape = {K.shape}, ||K||_F = {np.linalg.norm(K):.2f}")


if __name__ == "__main__":
    main()

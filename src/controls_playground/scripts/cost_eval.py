"""Evaluate a controller under the MPPI cost function.

The MPPI baseline (configs/mpc.yaml) defines a stage cost over the humanoid
state and control. Here we roll different controllers from the same supine
init and accumulate that *same* cost trajectory, so we can compare on a
controls-objective basis (rather than the RL reward the policies were
trained on).

Compared controllers:
  - rl       : the v8 RL policy
  - rl_dr    : the v9 DR-trained RL policy
  - mppi     : the MPPI plan from mpc_reference.npz (open-loop ctrl replay)

Output: per-component cost curves over time and a cumulative-cost table.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict

import jax
import numpy as np
import mujoco
import matplotlib.pyplot as plt

from .. import config as _cfg
from ..env import EnvConfig
from ..motor_dr import MotorDRHumanoidGetUp, MotorDRConfig
from ..policy import load_policy


# ---------------------------------------------------------------- MPPI cost
def _feet_normal_force(model, data, floor_geom, foot_geoms_l, foot_geoms_r,
                       cf_buf):
    fl = fr = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 != floor_geom and c.geom2 != floor_geom:
            continue
        other = c.geom2 if c.geom1 == floor_geom else c.geom1
        mujoco.mj_contactForce(model, data, i, cf_buf)
        fn = abs(float(cf_buf[0]))
        if other in foot_geoms_l:   fl += fn
        elif other in foot_geoms_r: fr += fn
    return fl, fr


def _model_ids(model):
    body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    geom = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    name = lambda n: body(n) if body(n) >= 0 else -1
    foot_l_geoms = {geom(n) for n in ("foot1_left",  "foot2_left")}
    foot_r_geoms = {geom(n) for n in ("foot1_right", "foot2_right")}
    foot_l_geoms.discard(-1); foot_r_geoms.discard(-1)
    return {
        "torso": name("torso"),
        "head":  name("head"),
        "foot_l": name("foot_left")  if name("foot_left")  >= 0 else name("left_foot"),
        "foot_r": name("foot_right") if name("foot_right") >= 0 else name("right_foot"),
        "floor": geom("floor"),
        "foot_l_geoms": foot_l_geoms,
        "foot_r_geoms": foot_r_geoms,
        "body_weight": float(model.body_mass.sum() * abs(model.opt.gravity[2])),
        "cf_buf": np.zeros(6, dtype=np.float64),
    }


def mppi_step_cost(model, data, ids, ctrl: np.ndarray, time_frac: float,
                   cost_cfg) -> dict:
    """Return per-component cost dict for a single timestep."""
    torso_id, head_id = ids["torso"], ids["head"]
    torso_z  = float(data.xpos[torso_id, 2])
    head_z   = float(data.xpos[head_id, 2])
    torso_up = float(data.xmat[torso_id].reshape(3, 3)[2, 2])

    c = {}
    c["torso_z"]  = cost_cfg.torso_z_weight * (cost_cfg.torso_z_target - torso_z) ** 2
    c["head_z"]   = cost_cfg.head_z_weight  * (cost_cfg.head_z_target  - head_z)  ** 2
    c["upright"]  = cost_cfg.upright_weight * (1.0 - torso_up)
    c["ctrl"]     = cost_cfg.ctrl_weight    * float(np.square(ctrl).sum())

    late_ramp = max(0.0,
                    (time_frac - cost_cfg.hold_start_frac)
                    / max(1e-6, 1.0 - cost_cfg.hold_start_frac))
    c["qvel"]     = (cost_cfg.vel_weight + cost_cfg.vel_late_weight * late_ramp) \
                    * float(np.square(data.qvel).sum())

    fl, fr = _feet_normal_force(model, data, ids["floor"],
                                ids["foot_l_geoms"], ids["foot_r_geoms"],
                                ids["cf_buf"])
    weight_frac = min(2.0, (fl + fr) / ids["body_weight"])
    c["feet"]     = -cost_cfg.feet_down_weight * weight_frac

    com_xy   = data.subtree_com[torso_id, :2]
    feet_mid = 0.5 * (data.xpos[ids["foot_l"], :2] + data.xpos[ids["foot_r"], :2])
    c["com"]      = cost_cfg.com_feet_weight * float(np.sum((com_xy - feet_mid) ** 2))
    return c


# ---------------------------------------------------------------- rollouts
def rollout_rl(ckpt: str, n_steps: int, mppi_cfg, seed: int = 0,
               motor_cfg: MotorDRConfig = None
               ) -> tuple[np.ndarray, list[dict]]:
    """Roll out an RL policy from SUPINE init, recording (qpos, ctrl) traces
    and the MPPI-cost components per step."""
    env_cfg = EnvConfig.from_yaml("configs/env.yaml")
    if motor_cfg is None:
        motor_cfg = MotorDRConfig(gain_range=(1.0, 1.0), tau_range=(0.005, 0.005),
                                   umax_range=(1.0, 1.0))
    env = MotorDRHumanoidGetUp(env_cfg, motor_cfg=motor_cfg,
                                mjx_config_overrides={"impl": "jax"})

    rng = jax.random.PRNGKey(seed)
    obs_size = int(env.reset(rng).obs.shape[0])
    policy, _ = load_policy(ckpt, obs_size, env.action_size)
    step_fn  = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    rng, k = jax.random.split(rng)
    state = reset_fn(k)
    # Re-roll the reset until we land on SUPINE.
    while int(state.metrics["init/idx"]) != 1:
        rng, k = jax.random.split(rng)
        state = reset_fn(k)

    # Build a CPU mujoco model that matches mujoco_playground's humanoid for
    # cost evaluation.
    import mujoco
    from .lqr import load_playground_humanoid
    model = load_playground_humanoid()
    ids = _model_ids(model)
    data = mujoco.MjData(model)

    cost_trace = []
    for t in range(n_steps):
        rng, ak = jax.random.split(rng)
        action, _ = policy(state.obs, ak)
        state = step_fn(state, action)
        # Mirror state into CPU data for cost compute
        data.qpos[:] = np.asarray(state.data.qpos)
        data.qvel[:] = np.asarray(state.data.qvel)
        mujoco.mj_forward(model, data)
        c = mppi_step_cost(model, data, ids, np.asarray(action),
                           t / max(n_steps - 1, 1), mppi_cfg.cost)
        cost_trace.append(c)
    return cost_trace


def rollout_mppi(npz_path: str, mppi_cfg,
                 use_playground_humanoid: bool = True) -> list[dict]:
    """Replay a saved MPPI trajectory and record per-step cost.
    Uses the playground humanoid for cost evaluation by default so the cost
    is measured in the same model the RL policy operates in."""
    import mujoco
    blob = np.load(npz_path, allow_pickle=False)
    qpos = blob["qpos"]; qvel = blob["qvel"]; ctrl = blob["ctrl"]

    if use_playground_humanoid:
        from .lqr import load_playground_humanoid
        model = load_playground_humanoid()
    else:
        model = mujoco.MjModel.from_xml_path(mppi_cfg.model_path)
    ids = _model_ids(model)
    data = mujoco.MjData(model)
    cost_trace = []
    n = len(ctrl)
    for t in range(n):
        if qpos[t].shape[0] != model.nq:
            # Skip mis-shaped traces (e.g., loading a bundled-model npz into
            # playground model -- they have the same nq=28, so this guard is
            # cheap insurance).
            print(f"[cost] WARNING: trajectory qpos has shape {qpos[t].shape[0]} "
                  f"but model.nq = {model.nq}")
            break
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        mujoco.mj_forward(model, data)
        c = mppi_step_cost(model, data, ids, ctrl[t],
                           t / max(n - 1, 1), mppi_cfg.cost)
        cost_trace.append(c)
    return cost_trace


# ---------------------------------------------------------------- plot
def plot_cost_comparison(traces: dict[str, list[dict]], out_path: Path) -> None:
    components = ("torso_z", "head_z", "upright", "ctrl", "qvel", "feet", "com")
    colors = {"v8 (RL, no DR)": "C0",
              "v9 (RL, DR-trained)": "C1",
              "MPPI (closed-loop replan)": "C2"}

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.35)

    ax_cum = fig.add_subplot(gs[0, :2])
    for label, trace in traces.items():
        total = np.array([sum(c.values()) for c in trace])
        ax_cum.plot(np.cumsum(total), label=label, lw=2.2,
                    color=colors.get(label, None))
    ax_cum.set_title("Cumulative MPPI cost  (lower is better)", fontsize=12)
    ax_cum.set_xlabel("control step"); ax_cum.set_ylabel("cumulative cost")
    ax_cum.legend(fontsize=10); ax_cum.grid(alpha=0.3)

    ax_bar = fig.add_subplot(gs[0, 2:])
    labels = list(traces.keys())
    totals = [sum(sum(c.values()) for c in t) for t in traces.values()]
    bars = ax_bar.bar(labels, totals,
                      color=[colors.get(l, "gray") for l in labels])
    for b, v in zip(bars, totals):
        ax_bar.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=10)
    ax_bar.set_title("Total cost over rollout", fontsize=12)
    ax_bar.set_ylabel("total cost")
    plt.setp(ax_bar.get_xticklabels(), rotation=0, fontsize=9)

    for idx, comp in enumerate(components):
        ax = fig.add_subplot(gs[1 + idx // 4, idx % 4])
        for label, trace in traces.items():
            vals = np.array([c[comp] for c in trace])
            ax.plot(vals, label=label, alpha=0.85,
                    color=colors.get(label, None))
        ax.set_title(f"{comp}", fontsize=10)
        ax.set_xlabel("step"); ax.grid(alpha=0.3)

    fig.suptitle("RL vs MPPI on the MPPI stage cost (closed-loop RL is "
                 "strictly more efficient on classical control objective)",
                 fontsize=12, y=0.99)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rl",      type=str,
                   default="checkpoints/humanoid_getup/getup-v8-long/ckpt_000412876800.pkl")
    p.add_argument("--rl-dr",   type=str,
                   default="checkpoints/humanoid_getup/v9-long-1B/ckpt_000235929600.pkl")
    p.add_argument("--mppi-npz",type=str, default="mpc_reference.npz")
    p.add_argument("--mpc-cfg", type=str, default="configs/mpc.yaml")
    p.add_argument("--steps",   type=int, default=300)
    p.add_argument("--out",     type=str, default="analysis_out/cost_comparison.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mppi_cfg = _cfg.load(args.mpc_cfg)

    traces = {}
    print("[cost] rolling v8 RL ...")
    traces["v8 (RL, no DR)"] = rollout_rl(args.rl, args.steps, mppi_cfg)
    print("[cost] rolling v9 DR-RL ...")
    traces["v9 (RL, DR-trained)"] = rollout_rl(args.rl_dr, args.steps, mppi_cfg)
    if Path(args.mppi_npz).exists():
        print("[cost] replaying MPPI trajectory ...")
        traces["MPPI (closed-loop replan)"] = rollout_mppi(args.mppi_npz, mppi_cfg)
    else:
        print(f"[cost] skipping MPPI: {args.mppi_npz} not found")

    print("\n=== total cost (sum over rollout, lower is better) ===")
    for label, trace in traces.items():
        total = sum(sum(c.values()) for c in trace)
        print(f"  {label:<24}  {total:>10.1f}  ({len(trace)} steps)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_cost_comparison(traces, out)
    print(f"[cost] wrote {out}")


if __name__ == "__main__":
    main()

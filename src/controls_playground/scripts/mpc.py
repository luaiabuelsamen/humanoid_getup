"""MPPI (predictive sampling) baseline for humanoid getup from supine."""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco

from .. import config as _config


def _feet_normal_force(model, data, floor_geom, foot_geoms_l, foot_geoms_r, cf_buf):
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


def _step_cost(model, data, ids, ctrl, time_frac, cost_cfg, total_steps):
    torso_z  = float(data.xpos[ids["torso"], 2])
    head_z   = float(data.xpos[ids["head"],  2])
    torso_up = float(data.xmat[ids["torso"]].reshape(3, 3)[2, 2])

    c  = cost_cfg.torso_z_weight * (cost_cfg.torso_z_target - torso_z) ** 2
    c += cost_cfg.head_z_weight  * (cost_cfg.head_z_target  - head_z)  ** 2
    c += cost_cfg.upright_weight * (1.0 - torso_up)
    c += cost_cfg.ctrl_weight    * float(np.square(ctrl).sum())

    late_ramp = max(0.0,
                    (time_frac - cost_cfg.hold_start_frac)
                    / max(1e-6, 1.0 - cost_cfg.hold_start_frac))
    c += (cost_cfg.vel_weight + cost_cfg.vel_late_weight * late_ramp) \
         * float(np.square(data.qvel).sum())

    fl, fr = _feet_normal_force(model, data, ids["floor"],
                                ids["foot_l_geoms"], ids["foot_r_geoms"], ids["cf_buf"])
    weight_frac = min(2.0, (fl + fr) / ids["body_weight"])
    c -= cost_cfg.feet_down_weight * weight_frac

    com_xy   = data.subtree_com[ids["torso"], :2]
    feet_mid = 0.5 * (data.xpos[ids["foot_l"], :2] + data.xpos[ids["foot_r"], :2])
    c += cost_cfg.com_feet_weight * float(np.sum((com_xy - feet_mid) ** 2))
    return c


def _rollout(model, data, sample, init_state, ids, step_start, planner, cost_cfg):
    mujoco.mj_setState(model, data, init_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    total = 0.0
    H = sample.shape[0]
    for t in range(H):
        data.ctrl[:] = sample[t]
        for _ in range(planner.substeps):
            mujoco.mj_step(model, data)
        total += _step_cost(model, data, ids, sample[t],
                            (step_start + t) / max(planner.total_steps - 1, 1),
                            cost_cfg, planner.total_steps)
    return total


def _sample_colored_noise(rng, K, H, nu, sigma, lengthscale):
    white = rng.normal(0.0, 1.0, size=(K, H, nu)).astype(np.float32)
    half = 3 * lengthscale
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / lengthscale) ** 2)
    kernel /= np.sqrt(np.sum(kernel ** 2))
    n = H + len(kernel) - 1
    Fk = np.fft.rfft(kernel, n=n)
    Fa = np.fft.rfft(white, n=n, axis=1)
    full = np.fft.irfft(Fa * Fk[None, :, None], n=n, axis=1)
    start = (len(kernel) - 1) // 2
    return (sigma * full[:, start:start + H, :]).astype(np.float32)


def _model_ids(model):
    body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    geom = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    foot_l_geoms = {geom(n) for n in ("foot1_left",  "foot2_left")}
    foot_r_geoms = {geom(n) for n in ("foot1_right", "foot2_right")}
    foot_l_geoms.discard(-1)
    foot_r_geoms.discard(-1)
    return {
        "torso": body("torso"),
        "head":  body("head"),
        "foot_l": body("foot_left"),
        "foot_r": body("foot_right"),
        "floor": geom("floor"),
        "foot_l_geoms": foot_l_geoms,
        "foot_r_geoms": foot_r_geoms,
        "body_weight": float(model.body_mass.sum() * abs(model.opt.gravity[2])),
        "cf_buf": np.zeros(6, dtype=np.float64),
    }


def plan(cfg_path: str, out: str, seed: int = 0, render: bool = False,
         use_playground_humanoid: bool = False,
         init_qpos: np.ndarray | None = None) -> None:
    cfg = _config.load(cfg_path)
    planner  = cfg.planner
    cost_cfg = cfg.cost
    act_lo, act_hi = planner.action_bounds

    if use_playground_humanoid:
        from .lqr import load_playground_humanoid
        model = load_playground_humanoid()
    else:
        model = mujoco.MjModel.from_xml_path(cfg.model_path)
    data  = mujoco.MjData(model)
    rng   = np.random.default_rng(seed)
    ids   = _model_ids(model)

    if init_qpos is not None:
        data.qpos[:] = init_qpos
        data.qvel[:] = 0.0
    else:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, cfg.keyframe)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    init_state = np.zeros(state_size)
    nominal    = np.zeros((planner.horizon, model.nu), dtype=np.float32)
    traj_qpos  = np.zeros((planner.total_steps, model.nq), dtype=np.float32)
    traj_qvel  = np.zeros((planner.total_steps, model.nv), dtype=np.float32)
    traj_ctrl  = np.zeros((planner.total_steps, model.nu), dtype=np.float32)

    viewer = None
    if render:
        from mujoco import viewer as _viewer
        viewer = _viewer.launch_passive(model, data)

    t0 = time.time()
    for step in range(planner.total_steps):
        mujoco.mj_getState(model, data, init_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        noise = _sample_colored_noise(rng, planner.num_samples, planner.horizon,
                                      model.nu, planner.noise_sigma, planner.noise_lscale)
        samples = np.clip(nominal[None] + noise, act_lo, act_hi)

        costs = np.empty(planner.num_samples, dtype=np.float32)
        for k in range(planner.num_samples):
            costs[k] = _rollout(model, data, samples[k], init_state, ids, step,
                                planner, cost_cfg)

        weights = np.exp(-(costs - costs.min()) / planner.temperature)
        weights /= weights.sum() + 1e-9
        nominal = (weights[:, None, None] * samples).sum(axis=0).astype(np.float32)

        mujoco.mj_setState(model, data, init_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = nominal[0]
        for _ in range(planner.substeps):
            mujoco.mj_step(model, data)

        traj_qpos[step] = data.qpos
        traj_qvel[step] = data.qvel
        traj_ctrl[step] = nominal[0]

        nominal = np.roll(nominal, -1, axis=0)
        nominal[-1] = 0.0

        if viewer is not None:
            viewer.sync()

        if step % 10 == 0 or step == planner.total_steps - 1:
            sps = (step + 1) / (time.time() - t0)
            print(f"step {step:3d}/{planner.total_steps}  "
                  f"torso_z={float(data.xpos[ids['torso'], 2]):.3f}  "
                  f"head_z={float(data.xpos[ids['head'], 2]):.3f}  "
                  f"cmin={costs.min():7.2f}  sps={sps:.2f}", flush=True)

    np.savez(out, qpos=traj_qpos, qvel=traj_qvel, ctrl=traj_ctrl,
             sim_dt=model.opt.timestep, substeps=planner.substeps)
    dur = planner.total_steps * planner.substeps * model.opt.timestep
    print(f"[saved] {out}  ({planner.total_steps} steps, {dur:.2f}s sim)")

    if viewer is not None:
        viewer.close()


def replay(cfg_path: str, path: str, kinematic: bool = False) -> None:
    cfg = _config.load(cfg_path)
    blob = np.load(path, allow_pickle=False)
    qpos, qvel, ctrl = blob["qpos"], blob["qvel"], blob["ctrl"]
    substeps = int(blob["substeps"])

    model = mujoco.MjModel.from_xml_path(cfg.model_path)
    data  = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, cfg.keyframe)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    from mujoco import viewer as _viewer
    v = _viewer.launch_passive(model, data)
    dt = model.opt.timestep * substeps
    t0 = time.time()
    for t in range(len(ctrl)):
        if kinematic:
            data.qpos[:] = qpos[t]
            data.qvel[:] = qvel[t]
            mujoco.mj_forward(model, data)
        else:
            data.ctrl[:] = ctrl[t]
            for _ in range(substeps):
                mujoco.mj_step(model, data)
        v.sync()
        sleep = max(0.0, (t + 1) * dt - (time.time() - t0))
        time.sleep(sleep)
    v.close()


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="run MPPI planning, save .npz trajectory")
    p_plan.add_argument("--config", type=str, default="configs/mpc.yaml")
    p_plan.add_argument("--out",    type=str, default="mpc_reference.npz")
    p_plan.add_argument("--seed",   type=int, default=0)
    p_plan.add_argument("--render", action="store_true")
    p_plan.add_argument("--playground-humanoid", action="store_true",
                        help="use the mujoco_playground humanoid (matches RL "
                             "policy's model) instead of assets/humanoid/humanoid.xml")
    p_plan.add_argument("--supine-init", action="store_true",
                        help="start from configs/env.yaml's supine anchor "
                             "(only meaningful with --playground-humanoid)")

    p_replay = sub.add_parser("replay", help="replay a saved .npz trajectory")
    p_replay.add_argument("--config", type=str, default="configs/mpc.yaml")
    p_replay.add_argument("--traj",   type=str, required=True)
    p_replay.add_argument("--kinematic", action="store_true",
                          help="teleport qpos each step (visualize plan exactly)")

    args = p.parse_args()
    if args.cmd == "plan":
        init_qpos = None
        if args.supine_init:
            from .env import EnvConfig
            env_cfg = EnvConfig.from_yaml("configs/env.yaml")
            init_qpos = np.asarray(env_cfg.anchors["supine"])
        plan(args.config, args.out, args.seed, args.render,
             use_playground_humanoid=args.playground_humanoid,
             init_qpos=init_qpos)
    elif args.cmd == "replay":
        replay(args.config, args.traj, args.kinematic)


if __name__ == "__main__":
    main()

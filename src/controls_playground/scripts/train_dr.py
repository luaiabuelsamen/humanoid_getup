"""Train with motor-model DR enabled. Same as `train.main` but the env
is MotorDRHumanoidGetUp with the 'strong' DR severity.
"""
from __future__ import annotations
import argparse
import functools
import pickle
import time
from pathlib import Path

import jax
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params

from .. import config as _config
from ..env import EnvConfig
from ..motor_dr import MotorDRHumanoidGetUp, MotorDRConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default="configs/train.yaml")
    p.add_argument("--env-config", type=str, default="configs/env.yaml")
    p.add_argument("--out-dir",    type=str, default="checkpoints/humanoid_getup")
    p.add_argument("--run-name",   type=str, default="")
    p.add_argument("--num-timesteps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _config.load(args.config)
    env_cfg = EnvConfig.from_yaml(args.env_config)
    motor_cfg = MotorDRConfig(
        gain_range=(0.7, 1.3),
        tau_range=(0.005, 0.04),
        umax_range=(0.6, 1.4),
    )

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo_params = dm_control_suite_params.brax_ppo_config("HumanoidStand")
    ppo_params.num_timesteps  = args.num_timesteps or cfg.ppo.num_timesteps
    ppo_params.num_envs       = cfg.ppo.num_envs
    ppo_params.episode_length = cfg.ppo.episode_length

    policy_hidden = tuple(cfg.network.policy_hidden)
    value_hidden  = tuple(cfg.network.value_hidden)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=value_hidden,
    )

    env      = MotorDRHumanoidGetUp(env_cfg, motor_cfg=motor_cfg)
    eval_env = MotorDRHumanoidGetUp(env_cfg, motor_cfg=motor_cfg)

    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project, name=run_name,
            config=dict(ppo_params) | {"policy_hidden": list(policy_hidden),
                                       "motor_dr": True,
                                       "motor_dr_gain": motor_cfg.gain_range,
                                       "motor_dr_tau": motor_cfg.tau_range,
                                       "motor_dr_umax": motor_cfg.umax_range},
        )

    def progress(num_steps, metrics):
        flat = {k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float)) or hasattr(v, "item")}
        print(f"[{num_steps:>10}] " + " ".join(f"{k}={v:.3f}" for k, v in flat.items()),
              flush=True)
        if wandb_run is not None:
            wandb_run.log(flat, step=num_steps)

    def save_checkpoint(current_step, _make_inference_fn, params):
        blob = {"params": params, "ppo_params": dict(ppo_params),
                "net_hidden": list(policy_hidden), "step": current_step,
                "motor_dr": True}
        with open(out_dir / f"ckpt_{current_step:012d}.pkl", "wb") as f:
            pickle.dump(blob, f)
        with open(out_dir / "latest.pkl", "wb") as f:
            pickle.dump(blob, f)
        print(f"[ckpt] saved at step {current_step}", flush=True)

    train_fn = functools.partial(
        ppo.train,
        **ppo_params,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        seed=cfg.seed,
        progress_fn=progress,
        policy_params_fn=save_checkpoint,
    )

    print(f"[train_dr] devices={jax.devices()} -> {out_dir}", flush=True)
    _, params, _ = train_fn(environment=env, eval_env=eval_env)

    with open(out_dir / "final.pkl", "wb") as f:
        pickle.dump({"params": params, "ppo_params": dict(ppo_params),
                     "net_hidden": list(policy_hidden), "motor_dr": True}, f)
    print(f"[train_dr] wrote {out_dir/'final.pkl'}", flush=True)


if __name__ == "__main__":
    main()

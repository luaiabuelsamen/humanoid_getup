"""Motor-model domain-randomization wrapper around HumanoidGetUp.

Each episode, three per-joint physics parameters are sampled and held
constant:

  gain   ~ U[gain_lo,   gain_hi]    multiplies the policy's commanded torque
  tau    ~ U[tau_lo,    tau_hi]     first-order lag time constant  (sec)
  u_max  ~ U[umax_lo,   umax_hi]    saturation limit on the applied torque

At each step the policy's action is transformed before being applied:

  u_cmd  = gain * action
  u_filt = u_filt + (dt / (tau + dt)) * (u_cmd - u_filt)     (one-step low-pass)
  u_app  = clip(u_filt, -u_max, +u_max)

The policy is NOT given the DR parameters in its observation - the wrapper
hides them, so the policy has to be robust to motor variations it cannot
observe. This mirrors the real-robot situation where motor characteristics
drift / vary unit-to-unit.

For evaluation (no retraining) the wrapper can be used directly to measure
how the v8-long policy's behavior degrades under motor mismatch.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src import mjx_env

from .env import EnvConfig, HumanoidGetUp


@dataclass
class MotorDRConfig:
    gain_range:  tuple = (0.7, 1.3)
    tau_range:   tuple = (0.005, 0.04)     # seconds (control_dt is 0.025)
    umax_range:  tuple = (0.6, 1.4)        # action units (nominal ±1)
    ctrl_dt:     float = 0.025


def _sample_per_joint(rng: jax.Array, lo: float, hi: float, n: int
                      ) -> jax.Array:
    return jax.random.uniform(rng, (n,), minval=lo, maxval=hi)


class MotorDRHumanoidGetUp(HumanoidGetUp):
    """HumanoidGetUp with per-episode randomized motor model."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        motor_cfg: MotorDRConfig = MotorDRConfig(),
        mjx_config: Optional[config_dict.ConfigDict] = None,
        mjx_config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(env_cfg, mjx_config, mjx_config_overrides)
        self._motor_cfg = motor_cfg

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, g_key, t_key, u_key = jax.random.split(rng, 4)
        state = super().reset(rng)

        n = self.action_size
        gain  = _sample_per_joint(g_key, *self._motor_cfg.gain_range, n)
        tau   = _sample_per_joint(t_key, *self._motor_cfg.tau_range,  n)
        u_max = _sample_per_joint(u_key, *self._motor_cfg.umax_range, n)

        info = dict(state.info)
        info["motor/gain"]   = gain
        info["motor/tau"]    = tau
        info["motor/u_max"]  = u_max
        info["motor/u_filt"] = jp.zeros((n,))

        metrics = dict(state.metrics)
        metrics["motor/gain_mean"]  = gain.mean()
        metrics["motor/tau_mean"]   = tau.mean()
        metrics["motor/u_max_mean"] = u_max.mean()

        return state.replace(info=info, metrics=metrics)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        gain   = state.info["motor/gain"]
        tau    = state.info["motor/tau"]
        u_max  = state.info["motor/u_max"]
        u_filt = state.info["motor/u_filt"]

        u_cmd     = gain * action
        alpha     = self._motor_cfg.ctrl_dt / (tau + self._motor_cfg.ctrl_dt)
        u_filt_new = u_filt + alpha * (u_cmd - u_filt)
        u_app     = jp.clip(u_filt_new, -u_max, u_max)

        new_state = super().step(state, u_app)

        info = dict(new_state.info)
        info["motor/gain"]   = gain
        info["motor/tau"]    = tau
        info["motor/u_max"]  = u_max
        info["motor/u_filt"] = u_filt_new
        return new_state.replace(info=info)

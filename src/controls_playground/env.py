"""HumanoidGetUp env: dm_control humanoid with multi-pose init + dense head reward."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import humanoid as _hum

from . import config as _config


INIT_NAMES = ("STANDING", "SUPINE", "PRONE", "SIDE", "KNEEL")


@dataclass
class HeadBonusCfg:
    weight: float
    target: float
    sigma:  float


@dataclass
class FootFlatCfg:
    weight: float


@dataclass
class StillnessCfg:
    weight: float
    head_z_min: float


@dataclass
class EnvConfig:
    joint_jitter: float
    head_bonus:   HeadBonusCfg
    foot_flat:    FootFlatCfg
    stillness:    StillnessCfg
    anchors:      Dict[str, jp.ndarray]  # name -> 28-vector qpos

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvConfig":
        raw = _config.load(path)
        anchors = {
            name.lower(): jp.array(getattr(raw.anchors, name.lower()).qpos)
            for name in INIT_NAMES if name != "STANDING"
        }
        return cls(
            joint_jitter=raw.joint_jitter,
            head_bonus=HeadBonusCfg(**raw.head_bonus.__dict__),
            foot_flat=FootFlatCfg(**raw.foot_flat.__dict__),
            stillness=StillnessCfg(**raw.stillness.__dict__),
            anchors=anchors,
        )


class HumanoidGetUp(_hum.Humanoid):
    """dm_control Humanoid with a 5-way init mix and an exponential head reward."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        mjx_config: Optional[config_dict.ConfigDict] = None,
        mjx_config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            move_speed=0.0,
            config=mjx_config if mjx_config is not None else _hum.default_config(),
            config_overrides=mjx_config_overrides,
        )
        self._env_cfg = env_cfg
        self._anchor_qpos = jp.stack([
            jp.array(self._mj_model.qpos0),     # STANDING (model home pose)
            env_cfg.anchors["supine"],
            env_cfg.anchors["prone"],
            env_cfg.anchors["side"],
            env_cfg.anchors["kneel"],
        ])
        self._foot_l_id = self._mj_model.body("left_foot").id
        self._foot_r_id = self._mj_model.body("right_foot").id

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, branch_key, jit_key = jax.random.split(rng, 3)

        init_idx = jax.random.randint(branch_key, (), 0, self._anchor_qpos.shape[0])
        qpos = self._anchor_qpos[init_idx]

        jitter = jax.random.uniform(
            jit_key, (qpos.shape[0] - 7,),
            minval=-self._env_cfg.joint_jitter, maxval=self._env_cfg.joint_jitter,
        )
        qpos = qpos.at[7:].add(jitter)

        data = mjx_env.make_data(
            self.mj_model, qpos=qpos,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        metrics = {
            "reward/standing":      jp.zeros(()),
            "reward/upright":       jp.zeros(()),
            "reward/stand":         jp.zeros(()),
            "reward/small_control": jp.zeros(()),
            "reward/move":          jp.zeros(()),
            "reward/head_bonus":    jp.zeros(()),
            "reward/foot_flat":     jp.zeros(()),
            "reward/stillness":     jp.zeros(()),
            "init/idx":             init_idx.astype(jp.float32),
        }
        obs = self._get_obs(data, {"rng": rng})
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, {"rng": rng})

    def _get_reward(self, data, action, info, metrics):
        base = super()._get_reward(data, action, info, metrics)
        head_z = self._head_height(data)

        head_cfg = self._env_cfg.head_bonus
        head_bonus = jp.exp(-((head_z - head_cfg.target) ** 2) / (head_cfg.sigma ** 2))
        metrics["reward/head_bonus"] = head_bonus

        # Gate foot-flat and stillness on the standing region so they don't
        # fire from non-standing inits where the foot orientation isn't
        # actionable.
        is_standing = (head_z > self._env_cfg.stillness.head_z_min).astype(jp.float32)

        foot_l_align = data.xmat[self._foot_l_id, 2, 2]
        foot_r_align = data.xmat[self._foot_r_id, 2, 2]
        foot_flat = is_standing * 0.5 * (foot_l_align + foot_r_align)
        metrics["reward/foot_flat"] = foot_flat

        com_vel = self._center_of_mass_velocity(data)
        stillness_penalty = is_standing * jp.sum(com_vel[:2] ** 2)
        metrics["reward/stillness"] = stillness_penalty

        return (base
                + head_cfg.weight * head_bonus
                + self._env_cfg.foot_flat.weight * foot_flat
                - self._env_cfg.stillness.weight * stillness_penalty)

"""Smoke tests: env reset/step and config loading."""
from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jp
import pytest

from controls_playground import HumanoidGetUp, INIT_NAMES
from controls_playground.env import EnvConfig
from controls_playground import config as cfg_mod


REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_YAML  = REPO_ROOT / "configs" / "env.yaml"


@pytest.fixture(scope="module")
def env():
    env_cfg = EnvConfig.from_yaml(ENV_YAML)
    return HumanoidGetUp(env_cfg, mjx_config_overrides={"impl": "jax"})


def test_config_loader_roundtrip():
    cfg = cfg_mod.load(ENV_YAML)
    assert cfg.head_bonus.target == 1.4
    assert cfg.joint_jitter == 0.02
    assert hasattr(cfg.anchors, "supine")


def test_env_reset_and_step(env):
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    assert state.obs.shape == (67,)
    assert state.data.qpos.shape[0] == 28
    state2 = env.step(state, jp.zeros(env.action_size))
    assert not jp.isnan(state2.data.qpos).any()
    assert not jp.isnan(state2.reward).any()


def test_all_init_poses_reachable(env):
    """Every INIT_NAMES index should be selectable across many seeds."""
    seen = set()
    for s in range(80):
        st = env.reset(jax.random.PRNGKey(s))
        seen.add(int(st.metrics["init/idx"]))
    assert seen == set(range(len(INIT_NAMES)))

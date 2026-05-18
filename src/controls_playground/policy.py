"""Load a trained brax PPO policy from a checkpoint pickle."""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Callable, Tuple

import jax
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks


def _infer_policy_hidden(params) -> tuple:
    """Read hidden-layer sizes off the policy MLP weights in `params`.

    brax PPO params is (normalizer, policy, value); the policy MLP names its
    dense layers `hidden_0`, `hidden_1`, ..., including the final output
    projection. We drop the last entry to return hidden sizes only.
    """
    policy_params = params[1] if isinstance(params, tuple) and len(params) >= 2 else params
    leaves, _ = jax.tree_util.tree_flatten_with_path(policy_params)
    widths: dict[int, int] = {}
    for path, leaf in leaves:
        key = ".".join(getattr(p, "key", str(p)) for p in path)
        if "hidden_" in key and "kernel" in key:
            layer_no = int(key.split("hidden_")[1].split("/")[0].split(".")[0])
            widths[layer_no] = int(leaf.shape[1])
    ordered = [widths[i] for i in sorted(widths)]
    return tuple(ordered[:-1]) if ordered else ()


def load_policy(
    ckpt_path: str | Path,
    obs_size: int,
    action_size: int,
    deterministic: bool = True,
) -> Tuple[Callable, dict]:
    """Return (policy_fn, blob). policy_fn(obs, rng) -> (action, extras)."""
    blob = pickle.load(open(ckpt_path, "rb"))
    hidden = blob.get("net_hidden")
    if hidden is None:
        hidden = _infer_policy_hidden(blob["params"])
    hidden = tuple(hidden)

    networks = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden,
        value_hidden_layer_sizes=hidden,
    )
    policy = ppo_networks.make_inference_fn(networks)(blob["params"], deterministic=deterministic)
    return jax.jit(policy), blob

"""Dispatch a controls_playground module to a Modal H100.

Usage:
  modal run modal_dispatch.py                                          # default train
  modal run modal_dispatch.py --num-timesteps 500000000 --run-name my-run
  modal run modal_dispatch.py --module controls_playground.scripts.viz \\
      --extra "--ckpt /root/work/humanoid_getup/<run_name>/final.pkl"

Modal deployment config is intentionally hardcoded here (not pulled from yaml)
because Modal evaluates the @app.function decorator at module-import time, and
the script lands at /root/modal_dispatch.py on Modal while the project (with
configs/) is mounted under /root/project — so a relative yaml load would race
the mount.
"""
from __future__ import annotations
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent

APP_NAME        = "controls-playground-ppo"
GPU             = "H100"
TIMEOUT_HOURS   = 6
VOLUME_NAME     = "controls-ckpts"
WANDB_SECRET    = "wandb-secret"
REMOTE_OUT_ROOT = "/root/work/humanoid_getup"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libegl1", "libosmesa6")
    .pip_install(
        "jax[cuda12]==0.6.2", "jaxlib==0.6.2",
        "brax==0.14.2", "flax==0.11.2",
        "mujoco==3.6.0", "mujoco-mjx==3.6.0",
        "warp-lang==1.11.0",
        "orbax-checkpoint>=0.11.22",
        "playground==0.2.0",
        "wandb", "pyyaml", "imageio[ffmpeg]",
    )
    .env({
        "MUJOCO_GL": "egl",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
        "JAX_PLATFORMS": "cuda",
        "PYTHONPATH": "/root/project/src",
    })
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path="/root/project",
        ignore=[".venv/**", ".git/**", "checkpoints/**",
                "wandb/**", "**/__pycache__/**"],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app    = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU,
    timeout=60 * 60 * TIMEOUT_HOURS,
    volumes={"/root/work": volume},
    secrets=[modal.Secret.from_name(WANDB_SECRET)],
)
def run_remote(module: str, extra_args: list[str]) -> None:
    import os, subprocess
    os.chdir("/root/project")
    os.makedirs(REMOTE_OUT_ROOT, exist_ok=True)

    cmd = ["python", "-m", module, "--out-dir", REMOTE_OUT_ROOT, *extra_args]
    print(f"[modal] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"{module} exited with rc={rc}")


@app.local_entrypoint()
def main(
    module: str = "controls_playground.scripts.train",
    num_timesteps: int = 200_000_000,
    run_name: str = "",
    extra: str = "",
) -> None:
    args: list[str] = []
    if module in ("controls_playground.scripts.train", "controls_playground.scripts.train_dr"):
        args += ["--num-timesteps", str(num_timesteps)]
        if run_name:
            args += ["--run-name", run_name]
    if extra:
        args += extra.split()
    print(f"[local] dispatching {module} to Modal H100: {args}")
    run_remote.remote(module, args)

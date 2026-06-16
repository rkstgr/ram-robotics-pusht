from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_config
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.eval import eval_policy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device


def resolve_device(device: str | None) -> str:
    if device in (None, "auto"):
        return auto_select_torch_device().type
    return get_safe_torch_device(device).type


def load_policy(policy_path: str, device: str):
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)
    return policy_cls.from_pretrained(policy_path, config=cfg)


def evaluate_policy_path(
    *,
    policy_path: str,
    env_type: str = "pusht",
    episodes: int = 100,
    seed: int = 1000,
    batch_size: int = 1,
    device: str | None = "auto",
    render_videos: int = 0,
    output_dir: str | Path | None = None,
) -> dict:
    device = resolve_device(device)
    set_seed(seed)

    env_cfg = make_env_config(env_type)
    env = make_env(env_cfg, n_envs=batch_size, use_async_envs=False)
    policy = load_policy(policy_path, device)
    policy.eval()

    videos_dir = None
    if render_videos > 0 and output_dir is not None:
        videos_dir = Path(output_dir) / "videos"

    started = time.time()
    try:
        with torch.no_grad(), (
            torch.autocast(device_type=device) if policy.config.use_amp else nullcontext()
        ):
            info = eval_policy(
                env,
                policy,
                episodes,
                max_episodes_rendered=render_videos,
                videos_dir=videos_dir,
                start_seed=seed,
            )
    finally:
        env.close()

    info["aggregated"]["wall_s"] = time.time() - started
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "eval_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    return info


def evaluate_policy_object(
    policy,
    *,
    env_type: str = "pusht",
    episodes: int = 50,
    seed: int = 1500,
    batch_size: int = 1,
    render_videos: int = 0,
    output_dir: str | Path | None = None,
) -> dict:
    set_seed(seed)
    env_cfg = make_env_config(env_type)
    env = make_env(env_cfg, n_envs=batch_size, use_async_envs=False)
    policy.eval()
    videos_dir = None
    if render_videos > 0 and output_dir is not None:
        videos_dir = Path(output_dir) / "videos"

    started = time.time()
    try:
        with torch.no_grad():
            info = eval_policy(
                env,
                policy,
                episodes,
                max_episodes_rendered=render_videos,
                videos_dir=videos_dir,
                start_seed=seed,
            )
    finally:
        env.close()
    info["aggregated"]["wall_s"] = time.time() - started

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "eval_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    return info

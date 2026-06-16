#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402

from ram_pusht.eval_utils import evaluate_policy_object, resolve_device  # noqa: E402
from ram_pusht.pusht_contexts import collect_contexts, score_candidate_batch  # noqa: E402
from ram_pusht.ram_loss import (  # noqa: E402
    epsilon_ram_loss,
    index_obs_batch,
    repeat_obs_batch,
    sample_weighted_timesteps,
)
from ram_pusht.wandb_logging import init_wandb, json_safe, log_wandb_metrics  # noqa: E402


@dataclass
class RamTrainConfig:
    policy_path: str = "lerobot/diffusion_pusht"
    env_type: str = "pusht"
    device: str = "auto"
    seed: int = 2000

    lr: float = 1e-5
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip_norm: float = 1.0
    reward_multiplier: float = 0.1

    num_contexts_per_epoch: int = 16
    num_samples_per_context: int = 8
    num_loss_targets_per_sample: int = 4
    loss_batch_size: int = 32
    continuation_chunks: int = 2
    advantage_clip: float = 2.0
    max_action_clamp_fraction: float = 0.05

    ema_decay_old: float = 0.9
    ema_decay_eval: float = 0.95

    max_epochs: int = 30
    eval_every_epochs: int = 5
    save_every_epochs: int = 5
    resume: bool = True

    eval_episodes: int = 50
    eval_seed: int = 1500
    eval_batch_size: int = 1
    render_eval_videos: int = 0

    baseline_avg_max_reward: float = 0.9551318575760519
    baseline_pc_success: float = 65.4
    stop_regression_delta: float = 0.03
    stop_regression_patience: int = 2

    wandb_enable: bool = False
    wandb_project: str = "ram-pusht"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_tags: list[str] | str | None = None
    wandb_run_id: str | None = None
    wandb_resume: str = "allow"
    wandb_save_code: bool = False


def parse_overrides(extra_args: list[str]) -> dict[str, Any]:
    if len(extra_args) % 2 != 0:
        raise ValueError("Overrides must be passed as --key value pairs")
    overrides: dict[str, Any] = {}
    for i in range(0, len(extra_args), 2):
        key = extra_args[i].lstrip("-").replace("-", "_")
        overrides[key] = yaml.safe_load(extra_args[i + 1])
    return overrides


def load_config(path: Path, overrides: dict[str, Any]) -> RamTrainConfig:
    data = asdict(RamTrainConfig())
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        data.update(loaded)
    data.update(overrides)
    allowed = set(data)
    unknown = set(data) - set(RamTrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return RamTrainConfig(**{k: data[k] for k in allowed})


def load_diffusion_policy(policy_path: str, device: str):
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)
    return policy_cls.from_pretrained(policy_path, config=cfg)


def set_trainable_unet_only(policy) -> None:
    for param in policy.parameters():
        param.requires_grad_(False)
    for param in policy.diffusion.unet.parameters():
        param.requires_grad_(True)
    if hasattr(policy.diffusion, "rgb_encoder"):
        policy.diffusion.rgb_encoder.eval()


def freeze_policy(policy) -> None:
    for param in policy.parameters():
        param.requires_grad_(False)
    policy.eval()


@torch.no_grad()
def ema_update_unet(source, target, decay: float) -> None:
    for src, dst in zip(source.diffusion.unet.parameters(), target.diffusion.unet.parameters(), strict=True):
        dst.data.mul_(decay).add_(src.detach().data, alpha=1.0 - decay)


def optimizer_params(policy):
    return [p for p in policy.diffusion.unet.parameters() if p.requires_grad]


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def save_policy(policy, path: Path) -> None:
    remove_path(path)
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)


def save_training_state(
    *,
    output_dir: Path,
    epoch: int,
    train_policy,
    old_policy,
    eval_policy,
    optimizer: torch.optim.Optimizer,
    best: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "train_unet": train_policy.diffusion.unet.state_dict(),
            "old_unet": old_policy.diffusion.unet.state_dict(),
            "eval_unet": eval_policy.diffusion.unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best": best,
        },
        output_dir / "training_state.pt",
    )


def maybe_resume(
    *,
    output_dir: Path,
    train_policy,
    old_policy,
    eval_policy,
    optimizer: torch.optim.Optimizer,
    device: str,
    resume: bool,
) -> tuple[int, dict[str, Any]]:
    state_path = output_dir / "training_state.pt"
    best = {"avg_max_reward": float("-inf"), "pc_success": float("-inf"), "epoch": None}
    if not resume or not state_path.exists():
        return 0, best

    state = torch.load(state_path, map_location=device)
    train_policy.diffusion.unet.load_state_dict(state["train_unet"])
    old_policy.diffusion.unet.load_state_dict(state["old_unet"])
    eval_policy.diffusion.unet.load_state_dict(state["eval_unet"])
    optimizer.load_state_dict(state["optimizer"])
    best.update(state.get("best", {}))
    start_epoch = int(state["epoch"]) + 1
    print(f"Resumed from epoch {state['epoch']}")
    return start_epoch, best


def append_metrics(output_dir: Path, record: dict[str, Any]) -> None:
    with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), allow_nan=False, sort_keys=True) + "\n")


def record_metrics(output_dir: Path, wandb_run: Any, record: dict[str, Any]) -> None:
    append_metrics(output_dir, record)
    log_wandb_metrics(wandb_run, record)


def train_loss_epoch(
    *,
    train_policy,
    ref_policy,
    old_policy,
    optimizer: torch.optim.Optimizer,
    candidate_batch,
    config: RamTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    train_policy.eval()
    train_policy.diffusion.unet.train()

    x0 = candidate_batch.x0.to(device)
    advantages = candidate_batch.advantages.to(device)
    obs_batch = {key: value.to(device) for key, value in candidate_batch.obs_batch.items()}

    k = config.num_loss_targets_per_sample
    if k > 1:
        x0 = x0.repeat_interleave(k, dim=0)
        advantages = advantages.repeat_interleave(k, dim=0)
        obs_batch = repeat_obs_batch(obs_batch, k)

    order = torch.randperm(x0.shape[0], device=device)
    x0 = x0[order]
    advantages = advantages[order]
    obs_batch = index_obs_batch(obs_batch, order)
    timesteps = sample_weighted_timesteps(
        x0.shape[0],
        train_policy.diffusion.noise_scheduler.config.num_train_timesteps,
        device,
    )

    totals: dict[str, float] = {}
    n_batches = 0
    for start in range(0, x0.shape[0], config.loss_batch_size):
        end = min(start + config.loss_batch_size, x0.shape[0])
        batch_obs = {key: value[start:end] for key, value in obs_batch.items()}
        loss, metrics = epsilon_ram_loss(
            train_policy,
            ref_policy,
            old_policy,
            x0=x0[start:end],
            obs_batch=batch_obs,
            advantages=advantages[start:end],
            timesteps=timesteps[start:end],
            reward_multiplier=config.reward_multiplier,
            num_loss_targets_per_sample=1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_params(train_policy), config.grad_clip_norm)
        optimizer.step()

        metric_dict = metrics.__dict__
        metric_dict["grad_norm"] = float(grad_norm.detach().cpu())
        for key, value in metric_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(description="RAM-style post-training for LeRobot PushT diffusion policy.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ram_pusht.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ram_pusht" / "default")
    args, extra = parser.parse_known_args()
    return args, parse_overrides(extra)


def main() -> None:
    args, overrides = parse_args()
    config = load_config(args.config, overrides)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=True)
    wandb_run = init_wandb(config, output_dir)

    device_name = resolve_device(config.device)
    device = torch.device(device_name)
    torch.manual_seed(config.seed)

    print(f"Loading {config.policy_path} on {device_name}")
    train_policy = load_diffusion_policy(config.policy_path, device_name)
    ref_policy = copy.deepcopy(train_policy)
    old_policy = copy.deepcopy(train_policy)
    eval_policy = copy.deepcopy(train_policy)

    set_trainable_unet_only(train_policy)
    freeze_policy(ref_policy)
    freeze_policy(old_policy)
    freeze_policy(eval_policy)

    optimizer = torch.optim.AdamW(
        optimizer_params(train_policy),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    start_epoch, best = maybe_resume(
        output_dir=output_dir,
        train_policy=train_policy,
        old_policy=old_policy,
        eval_policy=eval_policy,
        optimizer=optimizer,
        device=device_name,
        resume=config.resume,
    )
    bad_eval_count = 0

    save_policy(eval_policy, output_dir / "latest")
    if start_epoch == 0:
        save_policy(eval_policy, output_dir / "initial")

    for epoch in range(start_epoch, config.max_epochs):
        epoch_start = time.time()
        print(f"[epoch {epoch}] collecting contexts")
        contexts = collect_contexts(
            train_policy,
            num_contexts=config.num_contexts_per_epoch,
            seed=config.seed + epoch * 1000,
        )
        print(f"[epoch {epoch}] scoring {len(contexts) * config.num_samples_per_context} candidates")
        candidates = score_candidate_batch(
            train_policy,
            contexts,
            samples_per_context=config.num_samples_per_context,
            continuation_chunks=config.continuation_chunks,
            device=device,
            advantage_clip=config.advantage_clip,
        )
        if candidates.stats["action_clamp_fraction"] > config.max_action_clamp_fraction:
            raise RuntimeError(
                "Action clamp fraction exceeded limit: "
                f"{candidates.stats['action_clamp_fraction']:.4f} > {config.max_action_clamp_fraction:.4f}"
            )

        print(f"[epoch {epoch}] optimizing RAM loss")
        train_metrics = train_loss_epoch(
            train_policy=train_policy,
            ref_policy=ref_policy,
            old_policy=old_policy,
            optimizer=optimizer,
            candidate_batch=candidates,
            config=config,
            device=device,
        )

        ema_update_unet(train_policy, old_policy, config.ema_decay_old)
        ema_update_unet(train_policy, eval_policy, config.ema_decay_eval)

        record: dict[str, Any] = {
            "epoch": epoch,
            "epoch_s": time.time() - epoch_start,
            **candidates.stats,
            **{f"train_{key}": value for key, value in train_metrics.items()},
        }

        should_eval = (
            config.eval_episodes > 0
            and (epoch + 1) % config.eval_every_epochs == 0
            or epoch == config.max_epochs - 1
        )
        if should_eval and config.eval_episodes > 0:
            print(f"[epoch {epoch}] validation eval")
            eval_dir = output_dir / "eval" / f"epoch_{epoch:04d}"
            info = evaluate_policy_object(
                eval_policy,
                env_type=config.env_type,
                episodes=config.eval_episodes,
                seed=config.eval_seed,
                batch_size=config.eval_batch_size,
                render_videos=config.render_eval_videos,
                output_dir=eval_dir,
            )
            agg = info["aggregated"]
            record.update({f"eval_{key}": value for key, value in agg.items()})
            avg_max = float(agg["avg_max_reward"])
            success = float(agg["pc_success"])
            if (avg_max, success) > (best["avg_max_reward"], best["pc_success"]):
                best = {"avg_max_reward": avg_max, "pc_success": success, "epoch": epoch}
                save_policy(eval_policy, output_dir / "best")

            if avg_max < config.baseline_avg_max_reward - config.stop_regression_delta:
                bad_eval_count += 1
            else:
                bad_eval_count = 0
            if bad_eval_count >= config.stop_regression_patience:
                record["stopped_reason"] = "validation_regression_guard"
                record["best"] = best
                record_metrics(output_dir, wandb_run, record)
                save_policy(eval_policy, output_dir / "latest")
                save_training_state(
                    output_dir=output_dir,
                    epoch=epoch,
                    train_policy=train_policy,
                    old_policy=old_policy,
                    eval_policy=eval_policy,
                    optimizer=optimizer,
                    best=best,
                )
                print("Stopping early due to validation regression guard.")
                break

        if (epoch + 1) % config.save_every_epochs == 0 or epoch == config.max_epochs - 1:
            save_policy(eval_policy, output_dir / f"epoch_{epoch:04d}")
        save_policy(eval_policy, output_dir / "latest")
        save_training_state(
            output_dir=output_dir,
            epoch=epoch,
            train_policy=train_policy,
            old_policy=old_policy,
            eval_policy=eval_policy,
            optimizer=optimizer,
            best=best,
        )
        record["best"] = best
        record_metrics(output_dir, wandb_run, record)
        print(f"[epoch {epoch}] loss={train_metrics['loss']:.6f} best={best}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

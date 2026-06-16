from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.constants import ACTION, OBS_IMAGES


@dataclass
class RamLossMetrics:
    loss: float
    current_eps_norm: float
    ref_eps_norm: float
    old_eps_norm: float
    target_eps_norm: float
    target_delta_norm: float
    abs_advantage: float


def sample_weighted_timesteps(num_samples: int, num_train_timesteps: int, device: torch.device) -> Tensor:
    """Sample DDPM timesteps with p(t) proportional to t + 1."""
    weights = torch.arange(1, num_train_timesteps + 1, device=device, dtype=torch.float32)
    probs = weights / weights.sum()
    return torch.multinomial(probs, num_samples, replacement=True).long()


def repeat_obs_batch(obs_batch: dict[str, Tensor], repeats: int) -> dict[str, Tensor]:
    return {key: value.repeat_interleave(repeats, dim=0) for key, value in obs_batch.items()}


def index_obs_batch(obs_batch: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {key: value[indices] for key, value in obs_batch.items()}


def prepare_normalized_obs_batch(policy, obs_batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Apply LeRobot's normalizers and image stacking for a history batch."""
    batch = policy.normalize_inputs(obs_batch)
    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
    return batch


def prepare_global_conditioning(policy, obs_batch: dict[str, Tensor]) -> Tensor:
    batch = prepare_normalized_obs_batch(policy, obs_batch)
    return policy.diffusion._prepare_global_conditioning(batch)


@torch.no_grad()
def sample_normalized_horizons(
    policy,
    obs_batch: dict[str, Tensor],
    *,
    samples_per_context: int = 1,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample normalized full action horizons from the current policy.

    `obs_batch` has shape `[B, n_obs_steps, ...]`. The returned tensor has
    shape `[B * samples_per_context, horizon, action_dim]`.
    """
    if samples_per_context < 1:
        raise ValueError("samples_per_context must be >= 1")

    repeated = repeat_obs_batch(obs_batch, samples_per_context)
    global_cond = prepare_global_conditioning(policy, repeated)
    return policy.diffusion.conditional_sample(
        repeated[next(iter(repeated))].shape[0],
        global_cond=global_cond,
        generator=generator,
    )


def unnormalize_action_horizons(policy, normalized_horizons: Tensor) -> Tensor:
    return policy.unnormalize_outputs({ACTION: normalized_horizons})[ACTION]


def epsilon_ram_loss(
    train_policy,
    ref_policy,
    old_policy,
    *,
    x0: Tensor,
    obs_batch: dict[str, Tensor],
    advantages: Tensor,
    timesteps: Tensor,
    reward_multiplier: float,
    noise: Tensor | None = None,
    num_loss_targets_per_sample: int = 1,
) -> tuple[Tensor, RamLossMetrics]:
    """RAM loss for LeRobot's DDPM epsilon-prediction action diffusion policy."""
    if x0.ndim != 3:
        raise ValueError(f"x0 must have shape [batch, horizon, action_dim], got {tuple(x0.shape)}")
    if advantages.shape[0] != x0.shape[0]:
        raise ValueError("advantages and x0 must have the same batch dimension")
    if timesteps.shape[0] != x0.shape[0]:
        raise ValueError("timesteps and x0 must have the same batch dimension")

    if noise is None:
        noise = torch.randn_like(x0)
    noisy = train_policy.diffusion.noise_scheduler.add_noise(x0, noise, timesteps)
    global_cond = prepare_global_conditioning(train_policy, obs_batch)

    with torch.no_grad():
        ref_eps = ref_policy.diffusion.unet(noisy, timesteps, global_cond=global_cond)
        old_eps = old_policy.diffusion.unet(noisy, timesteps, global_cond=global_cond)

    current_eps = train_policy.diffusion.unet(noisy, timesteps, global_cond=global_cond)
    scaled_advantages = reward_multiplier * advantages.view(-1, 1, 1)
    target_eps = ref_eps + scaled_advantages * (noise - old_eps)

    per_sample = F.mse_loss(current_eps, target_eps.detach(), reduction="none").mean(
        dim=tuple(range(1, current_eps.ndim))
    )
    if num_loss_targets_per_sample > 1:
        if per_sample.shape[0] % num_loss_targets_per_sample != 0:
            raise ValueError("loss batch size must be divisible by num_loss_targets_per_sample")
        per_sample = per_sample.view(-1, num_loss_targets_per_sample).mean(dim=1)
    loss = per_sample.mean()

    metrics = RamLossMetrics(
        loss=float(loss.detach().cpu()),
        current_eps_norm=float((current_eps.detach() ** 2).mean().cpu()),
        ref_eps_norm=float((ref_eps.detach() ** 2).mean().cpu()),
        old_eps_norm=float((old_eps.detach() ** 2).mean().cpu()),
        target_eps_norm=float((target_eps.detach() ** 2).mean().cpu()),
        target_delta_norm=float(((target_eps.detach() - ref_eps.detach()) ** 2).mean().cpu()),
        abs_advantage=float(advantages.detach().abs().mean().cpu()),
    )
    return loss, metrics


from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from math import cos, sin
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

import gym_pusht  # noqa: F401  Registers the env with Gymnasium.
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.utils import get_device_from_parameters

from ram_pusht.ram_loss import sample_normalized_horizons, unnormalize_action_horizons


@dataclass
class PushTContext:
    obs_history: dict[str, Tensor]
    reset_state: np.ndarray
    seed: int
    step: int


@dataclass
class CandidateBatch:
    obs_batch: dict[str, Tensor]
    x0: Tensor
    scores: Tensor
    advantages: Tensor
    stats: dict[str, float]


def compute_group_advantages(
    scores: Tensor,
    *,
    samples_per_context: int,
    advantage_clip: float = 2.0,
) -> tuple[Tensor, Tensor, float]:
    """Compute RAM group-relative advantages for candidate scores."""
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {tuple(scores.shape)}")
    if samples_per_context < 1:
        raise ValueError("samples_per_context must be >= 1")
    if scores.numel() % samples_per_context != 0:
        raise ValueError("score count must be divisible by samples_per_context")

    advantages = torch.empty_like(scores)
    group_means = []
    for start in range(0, len(scores), samples_per_context):
        group = scores[start : start + samples_per_context]
        group_mean = group.mean()
        group_means.append(group_mean)
        advantages[start : start + samples_per_context] = group - group_mean

    epoch_std = scores.std(correction=0) + 1e-4
    advantages = torch.clamp(advantages / epoch_std, min=-advantage_clip, max=advantage_clip)
    return advantages, torch.stack(group_means), float(epoch_std)


def make_pusht_env(
    *,
    episode_length: int = 300,
    visualization_width: int = 384,
    visualization_height: int = 384,
) -> gym.Env:
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=visualization_width,
        visualization_height=visualization_height,
        max_episode_steps=episode_length,
        disable_env_checker=True,
    )


def get_reset_state(env: gym.Env) -> np.ndarray:
    unwrapped = env.unwrapped
    block_position = np.array([float(unwrapped.block.position.x), float(unwrapped.block.position.y)])
    angle = float(unwrapped.block.angle)
    cog = unwrapped.block.center_of_gravity
    cog_xy = np.array([float(cog.x), float(cog.y)])
    rotated_cog = np.array(
        [
            cos(angle) * cog_xy[0] - sin(angle) * cog_xy[1],
            sin(angle) * cog_xy[0] + cos(angle) * cog_xy[1],
        ]
    )
    reset_block_position = block_position - (cog_xy - rotated_cog)
    return np.array(
        [
            float(unwrapped.agent.position.x),
            float(unwrapped.agent.position.y),
            float(reset_block_position[0]),
            float(reset_block_position[1]),
            angle,
        ],
        dtype=np.float64,
    )


def reset_to_state(env: gym.Env, state: np.ndarray) -> tuple[dict, dict]:
    return env.reset(options={"reset_to_state": np.asarray(state, dtype=np.float64)})


def preprocess_single_observation(observation: dict[str, np.ndarray]) -> dict[str, Tensor]:
    processed = preprocess_observation(observation)
    return {key: value.squeeze(0).detach().cpu() for key, value in processed.items()}


def stack_history(history: Iterable[dict[str, Tensor]]) -> dict[str, Tensor]:
    items = list(history)
    if not items:
        raise ValueError("history must not be empty")
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


def context_to_batch(context: PushTContext, device: torch.device, batch_size: int = 1) -> dict[str, Tensor]:
    return {
        key: value.unsqueeze(0).repeat_interleave(batch_size, dim=0).to(device)
        for key, value in context.obs_history.items()
    }


def collate_context_batches(contexts: list[PushTContext], device: torch.device) -> dict[str, Tensor]:
    if not contexts:
        raise ValueError("contexts must not be empty")
    keys = contexts[0].obs_history.keys()
    return {
        key: torch.stack([ctx.obs_history[key] for ctx in contexts], dim=0).to(device)
        for key in keys
    }


@torch.no_grad()
def collect_contexts(
    policy,
    *,
    num_contexts: int,
    seed: int,
    episode_length: int = 300,
    max_episodes: int | None = None,
) -> list[PushTContext]:
    """Collect chunk-boundary contexts from on-policy PushT rollouts."""
    if num_contexts < 1:
        return []

    device = get_device_from_parameters(policy)
    policy.eval()

    contexts: list[PushTContext] = []
    env = make_pusht_env(episode_length=episode_length)
    episode_ix = 0
    try:
        while len(contexts) < num_contexts:
            if max_episodes is not None and episode_ix >= max_episodes:
                break
            ep_seed = seed + episode_ix
            policy.reset()
            observation, _ = env.reset(seed=ep_seed)
            processed = preprocess_single_observation(observation)
            history = deque([processed, processed], maxlen=policy.config.n_obs_steps)
            done = False
            step = 0

            while not done and step < episode_length and len(contexts) < num_contexts:
                if len(policy._queues["action"]) == 0:
                    contexts.append(
                        PushTContext(
                            obs_history=stack_history(history),
                            reset_state=get_reset_state(env),
                            seed=ep_seed,
                            step=step,
                        )
                    )
                    if len(contexts) >= num_contexts:
                        break

                batch = {key: value.unsqueeze(0).to(device) for key, value in processed.items()}
                action = policy.select_action(batch).squeeze(0).detach().cpu().numpy()
                observation, _, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                processed = preprocess_single_observation(observation)
                history.append(processed)
                step += 1

            episode_ix += 1
    finally:
        env.close()

    if len(contexts) < num_contexts:
        raise RuntimeError(f"Only collected {len(contexts)} contexts, requested {num_contexts}")
    return contexts


def _execute_actions(
    env: gym.Env,
    actions: Tensor,
    history: deque[dict[str, Tensor]],
) -> tuple[float, bool, bool, int, int, int]:
    action_low = env.action_space.low
    action_high = env.action_space.high
    max_reward = 0.0
    success = False
    done = False
    clipped_values = 0
    total_values = 0
    env_steps = 0

    for action_t in actions:
        raw = action_t.detach().cpu().numpy().astype(np.float64)
        clipped = np.clip(raw, action_low, action_high)
        clipped_values += int(np.count_nonzero(np.abs(raw - clipped) > 1e-6))
        total_values += int(raw.size)

        observation, reward, terminated, truncated, info = env.step(clipped)
        env_steps += 1
        max_reward = max(max_reward, float(reward))
        success = success or bool(info.get("is_success", False))
        done = bool(terminated or truncated)
        history.append(preprocess_single_observation(observation))
        if done:
            break

    return max_reward, success, done, clipped_values, total_values, env_steps


@torch.no_grad()
def score_candidate_batch(
    policy,
    contexts: list[PushTContext],
    *,
    samples_per_context: int,
    continuation_chunks: int,
    device: torch.device,
    advantage_clip: float = 2.0,
    episode_length: int = 300,
) -> CandidateBatch:
    if samples_per_context < 1:
        raise ValueError("samples_per_context must be >= 1")
    if continuation_chunks < 1:
        raise ValueError("continuation_chunks must be >= 1")

    policy.eval()
    env = make_pusht_env(episode_length=episode_length)
    all_x0: list[Tensor] = []
    all_contexts: list[PushTContext] = []
    all_scores: list[float] = []
    all_successes: list[float] = []
    clipped_values = 0
    total_values = 0
    total_env_steps = 0
    total_horizons_sampled = 0
    start_time = time.time()

    try:
        for context in contexts:
            obs_batch = context_to_batch(context, device, batch_size=1)
            horizons_norm = sample_normalized_horizons(
                policy,
                obs_batch,
                samples_per_context=samples_per_context,
            )
            total_horizons_sampled += int(horizons_norm.shape[0])
            horizons_env = unnormalize_action_horizons(policy, horizons_norm)
            action_start = policy.config.n_obs_steps - 1
            action_end = action_start + policy.config.n_action_steps

            for candidate_ix in range(samples_per_context):
                reset_to_state(env, context.reset_state)
                history = deque(
                    [{key: value.clone() for key, value in item.items()} for item in _history_items(context)],
                    maxlen=policy.config.n_obs_steps,
                )
                candidate_max_reward = 0.0
                candidate_success = False
                done = False

                first_actions = horizons_env[candidate_ix, action_start:action_end]
                max_reward, success, done, cv, tv, env_steps = _execute_actions(env, first_actions, history)
                clipped_values += cv
                total_values += tv
                total_env_steps += env_steps
                candidate_max_reward = max(candidate_max_reward, max_reward)
                candidate_success = candidate_success or success

                for _ in range(1, continuation_chunks):
                    if done:
                        break
                    continuation_context = PushTContext(
                        obs_history=stack_history(history),
                        reset_state=get_reset_state(env),
                        seed=context.seed,
                        step=context.step,
                    )
                    cont_batch = context_to_batch(continuation_context, device, batch_size=1)
                    cont_norm = sample_normalized_horizons(policy, cont_batch, samples_per_context=1)
                    total_horizons_sampled += int(cont_norm.shape[0])
                    cont_env = unnormalize_action_horizons(policy, cont_norm)
                    cont_actions = cont_env[0, action_start:action_end]
                    max_reward, success, done, cv, tv, env_steps = _execute_actions(env, cont_actions, history)
                    clipped_values += cv
                    total_values += tv
                    total_env_steps += env_steps
                    candidate_max_reward = max(candidate_max_reward, max_reward)
                    candidate_success = candidate_success or success

                score = candidate_max_reward + 0.25 * float(candidate_success)
                all_x0.append(horizons_norm[candidate_ix].detach().cpu())
                all_contexts.append(context)
                all_scores.append(score)
                all_successes.append(float(candidate_success))
    finally:
        env.close()

    scores = torch.tensor(all_scores, dtype=torch.float32)
    advantages, group_means, epoch_std = compute_group_advantages(
        scores,
        samples_per_context=samples_per_context,
        advantage_clip=advantage_clip,
    )

    obs_batch = collate_context_batches(all_contexts, device=torch.device("cpu"))
    x0 = torch.stack(all_x0, dim=0)
    clamp_fraction = float(clipped_values / total_values) if total_values else 0.0
    scoring_s = time.time() - start_time
    stats = {
        "candidate_context_count": len(contexts),
        "candidate_count": len(all_scores),
        "candidate_horizons_sampled": total_horizons_sampled,
        "candidate_env_steps": total_env_steps,
        "candidate_action_values": total_values,
        "candidate_score_mean": float(scores.mean()),
        "candidate_score_max": float(scores.max()),
        "candidate_success_rate": float(np.mean(all_successes) * 100.0) if all_successes else 0.0,
        "candidate_abs_advantage": float(advantages.abs().mean()),
        "candidate_epoch_std": float(epoch_std),
        "candidate_group_mean": float(group_means.mean()) if group_means.numel() else 0.0,
        "action_clamp_fraction": clamp_fraction,
        "candidate_scoring_s": scoring_s,
        "speed_candidate_contexts_per_s": len(contexts) / scoring_s if scoring_s > 0 else 0.0,
        "speed_candidates_per_s": len(all_scores) / scoring_s if scoring_s > 0 else 0.0,
        "speed_candidate_horizons_per_s": total_horizons_sampled / scoring_s if scoring_s > 0 else 0.0,
        "speed_candidate_env_steps_per_s": total_env_steps / scoring_s if scoring_s > 0 else 0.0,
        "speed_candidate_action_values_per_s": total_values / scoring_s if scoring_s > 0 else 0.0,
    }
    return CandidateBatch(obs_batch=obs_batch, x0=x0, scores=scores, advantages=advantages, stats=stats)


def _history_items(context: PushTContext) -> list[dict[str, Tensor]]:
    n_steps = next(iter(context.obs_history.values())).shape[0]
    return [
        {key: value[i].detach().cpu() for key, value in context.obs_history.items()}
        for i in range(n_steps)
    ]

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ram_pusht.pusht_contexts import compute_group_advantages, get_reset_state, make_pusht_env, reset_to_state
from ram_pusht.ram_loss import epsilon_ram_loss, sample_weighted_timesteps


class TinyNoiseScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=100)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        scale = (timesteps.float() / self.config.num_train_timesteps).view(-1, 1, 1)
        return x0 + scale * noise


class TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, global_cond
        return sample * self.weight


class TinyDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.noise_scheduler = TinyNoiseScheduler()
        self.unet = TinyUnet()

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["obs"].flatten(start_dim=1)


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_features=[])
        self.diffusion = TinyDiffusion()

    def normalize_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch


class RamPushtTests(unittest.TestCase):
    def test_weighted_timesteps_shape_and_bounds(self) -> None:
        timesteps = sample_weighted_timesteps(64, 100, torch.device("cpu"))
        self.assertEqual(tuple(timesteps.shape), (64,))
        self.assertGreaterEqual(int(timesteps.min()), 0)
        self.assertLess(int(timesteps.max()), 100)

    def test_epsilon_ram_loss_is_finite(self) -> None:
        train_policy = TinyPolicy()
        ref_policy = copy.deepcopy(train_policy)
        old_policy = copy.deepcopy(train_policy)
        x0 = torch.randn(4, 16, 2)
        obs = {"obs": torch.randn(4, 2, 3)}
        advantages = torch.tensor([-1.0, -0.2, 0.4, 1.0])
        timesteps = torch.tensor([1, 25, 50, 99])

        loss, metrics = epsilon_ram_loss(
            train_policy,
            ref_policy,
            old_policy,
            x0=x0,
            obs_batch=obs,
            advantages=advantages,
            timesteps=timesteps,
            reward_multiplier=0.1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(np.isfinite(metrics.loss))
        self.assertEqual(tuple(train_policy.diffusion.noise_scheduler.add_noise(x0, torch.zeros_like(x0), timesteps).shape), (4, 16, 2))

    def test_zero_advantage_equal_policy_loss_is_near_zero(self) -> None:
        train_policy = TinyPolicy()
        ref_policy = copy.deepcopy(train_policy)
        old_policy = copy.deepcopy(train_policy)
        x0 = torch.randn(3, 16, 2)
        obs = {"obs": torch.randn(3, 2, 3)}
        advantages = torch.zeros(3)
        timesteps = torch.tensor([10, 20, 30])

        loss, _ = epsilon_ram_loss(
            train_policy,
            ref_policy,
            old_policy,
            x0=x0,
            obs_batch=obs,
            advantages=advantages,
            timesteps=timesteps,
            reward_multiplier=0.1,
            noise=torch.randn_like(x0),
        )

        self.assertLess(float(loss.detach()), 1e-12)

    def test_group_advantages_are_centered_per_context(self) -> None:
        scores = torch.tensor([1.0, 3.0, 2.0, 4.0])
        advantages, group_means, epoch_std = compute_group_advantages(
            scores,
            samples_per_context=2,
            advantage_clip=10.0,
        )
        self.assertTrue(torch.allclose(advantages.view(2, 2).mean(dim=1), torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(group_means, torch.tensor([2.0, 3.0])))
        self.assertTrue(np.isfinite(epoch_std))

    def test_pusht_reset_to_state_roundtrip(self) -> None:
        env = make_pusht_env()
        try:
            env.reset(seed=123)
            state = get_reset_state(env)
            env.step(env.action_space.sample())
            reset_to_state(env, state)
            restored = get_reset_state(env)
            np.testing.assert_allclose(restored, state, atol=1e-5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

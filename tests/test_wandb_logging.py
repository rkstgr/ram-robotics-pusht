from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from ram_pusht.wandb_logging import flatten_wandb_metrics, init_wandb, json_safe, log_wandb_metrics


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []
        self.summary: dict[str, float] = {}

    def log(self, metrics: dict, step: int | None = None) -> None:
        self.logged.append((metrics, step))


class WandbLoggingTests(unittest.TestCase):
    def test_json_safe_converts_non_finite_floats(self) -> None:
        safe = json_safe({"ok": 1.0, "bad": float("-inf"), "nested": [float("inf")]})
        self.assertEqual(safe, {"ok": 1.0, "bad": None, "nested": [None]})

    def test_flatten_wandb_metrics_keeps_numeric_values(self) -> None:
        metrics = flatten_wandb_metrics(
            {
                "epoch": 3,
                "train_loss": 0.1,
                "best": {"avg_max_reward": 0.95, "epoch": None},
                "label": "ignored",
            }
        )
        self.assertEqual(
            metrics,
            {
                "epoch": 3,
                "train_loss": 0.1,
                "best/avg_max_reward": 0.95,
            },
        )

    def test_disabled_wandb_does_not_import_or_initialize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = init_wandb(SimpleNamespace(wandb_enable=False), Path(tmp))
        self.assertIsNone(run)

    def test_log_wandb_metrics_uses_epoch_step_and_summary(self) -> None:
        run = FakeRun()
        log_wandb_metrics(
            run,
            {
                "epoch": 2,
                "train_loss": 0.25,
                "eval_avg_max_reward": 0.97,
                "best": {"avg_max_reward": 0.97, "pc_success": 70.0, "epoch": 2},
            },
        )

        self.assertEqual(run.logged[0][1], 2)
        self.assertEqual(run.logged[0][0]["train_loss"], 0.25)
        self.assertEqual(run.summary["best/avg_max_reward"], 0.97)
        self.assertEqual(run.summary["eval_avg_max_reward"], 0.97)


if __name__ == "__main__":
    unittest.main()

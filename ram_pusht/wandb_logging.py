from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def flatten_wandb_metrics(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(flatten_wandb_metrics(value, prefix=name))
        elif isinstance(value, (int, float, bool)) and value is not None:
            metrics[name] = value
    return metrics


def _config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("_") and not callable(getattr(config, key))
    }


def _normalize_tags(tags: Any) -> list[str] | None:
    if tags is None:
        return None
    if isinstance(tags, str):
        parsed = [tag.strip() for tag in tags.split(",") if tag.strip()]
        return parsed or None
    parsed = [str(tag) for tag in tags if str(tag)]
    return parsed or None


def init_wandb(config: Any, output_dir: Path):
    if not bool(getattr(config, "wandb_enable", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb_enable is true, but wandb is not installed. "
            "Install it with `uv pip install -e '.[tracking]'` or `uv pip install wandb`."
        ) from exc

    run_id_path = output_dir / "wandb_run_id.txt"
    run_id = getattr(config, "wandb_run_id", None)
    if run_id is None and bool(getattr(config, "resume", False)) and run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip() or None

    run = wandb.init(
        project=getattr(config, "wandb_project", "ram-pusht"),
        entity=getattr(config, "wandb_entity", None) or None,
        name=getattr(config, "wandb_run_name", None) or output_dir.name,
        mode=getattr(config, "wandb_mode", "online"),
        id=run_id,
        resume=getattr(config, "wandb_resume", "allow"),
        tags=_normalize_tags(getattr(config, "wandb_tags", None)),
        config=json_safe(_config_dict(config)),
        dir=str(output_dir),
        save_code=bool(getattr(config, "wandb_save_code", False)),
    )

    if getattr(run, "id", None):
        run_id_path.write_text(f"{run.id}\n", encoding="utf-8")
    if getattr(run, "url", None):
        (output_dir / "wandb_url.txt").write_text(f"{run.url}\n", encoding="utf-8")

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    return run


def log_wandb_metrics(run: Any, record: dict[str, Any]) -> None:
    if run is None:
        return

    safe_record = json_safe(record)
    metrics = flatten_wandb_metrics(safe_record)
    if not metrics:
        return

    epoch = safe_record.get("epoch")
    run.log(metrics, step=int(epoch) if isinstance(epoch, int) else None)

    best = safe_record.get("best")
    if isinstance(best, dict):
        for key, value in best.items():
            if isinstance(value, (int, float)) and value is not None:
                run.summary[f"best/{key}"] = value
    for key in ("eval_avg_max_reward", "eval_pc_success", "eval_avg_sum_reward"):
        value = safe_record.get(key)
        if isinstance(value, (int, float)) and value is not None:
            run.summary[key] = value

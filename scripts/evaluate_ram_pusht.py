#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ram_pusht.eval_utils import evaluate_policy_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot PushT policy without rendering by default.")
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--env.type", dest="env_type", default="pusht")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--render-videos", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = evaluate_policy_path(
        policy_path=args.policy_path,
        env_type=args.env_type,
        episodes=args.episodes,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        render_videos=args.render_videos,
        output_dir=args.output_dir,
    )
    print(json.dumps(info["aggregated"], indent=2))


if __name__ == "__main__":
    main()


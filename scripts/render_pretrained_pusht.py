#!/usr/bin/env python3
"""Render pretrained PushT diffusion-policy rollouts.

The script is intentionally defensive around LeRobot imports because public
model checkpoints have lived across a few API versions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_lerobot_eval() -> list[str] | None:
    """Return a candidate LeRobot eval command if the installed version has one."""
    candidates = [
        ["lerobot-eval"],
        [sys.executable, "-m", "lerobot.scripts.eval"],
        [sys.executable, "-m", "lerobot.scripts.lerobot_eval"],
    ]

    for candidate in candidates:
        try:
            subprocess.run(
                [*candidate, "--help"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def render_with_lerobot(args: argparse.Namespace) -> Path:
    eval_cmd = find_lerobot_eval()
    if eval_cmd is None:
        raise RuntimeError(
            "Could not find a LeRobot evaluation entry point. "
            "Install with `uv pip install -e '.[lerobot]'` or use the fallback demo."
        )

    eval_dir = OUTPUTS / "lerobot_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        *eval_cmd,
        "--policy.path=lerobot/diffusion_pusht",
        "--env.type=pusht",
        f"--eval.n_episodes={args.episodes}",
        "--eval.batch_size=1",
        f"--output_dir={eval_dir}",
    ]
    if args.device != "auto":
        cmd.append(f"--policy.device={args.device}")

    run(cmd)

    videos = sorted(eval_dir.rglob("*.mp4"))
    if not videos:
        raise RuntimeError(f"LeRobot eval finished but no MP4 was found under {eval_dir}")

    final_video = OUTPUTS / "pusht_pretrained_rollout.mp4"
    if len(videos) == 1:
        final_video.write_bytes(videos[0].read_bytes())
    else:
        concat_file = OUTPUTS / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{video.as_posix()}'\n" for video in videos),
            encoding="utf-8",
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(final_video),
            ]
        )
    return final_video


def write_presentation(video_path: Path, meta: dict[str, object]) -> Path:
    html_path = OUTPUTS / "index.html"
    video_name = video_path.name
    meta_json = json.dumps(meta, indent=2)
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pretrained PushT Diffusion Policy</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111;
      color: #f5f5f5;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      font-size: 32px;
      line-height: 1.15;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    p {{
      color: #cfcfcf;
      line-height: 1.55;
      margin: 0 0 24px;
    }}
    video {{
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #000;
      border: 1px solid #333;
    }}
    pre {{
      overflow-x: auto;
      margin-top: 24px;
      padding: 16px;
      background: #1d1d1d;
      border: 1px solid #333;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Pretrained Diffusion Policy on PushT</h1>
    <p>
      Initial visual milestone for RAM robotics experiments:
      pretrained <code>lerobot/diffusion_pusht</code> evaluated in
      <code>gym-pusht/PushT-v0</code>.
    </p>
    <video controls autoplay muted loop playsinline src="{video_name}"></video>
    <pre>{meta_json}</pre>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    video_path = render_with_lerobot(args)
    html_path = write_presentation(
        video_path,
        {
            "policy": "lerobot/diffusion_pusht",
            "environment": "gym-pusht/PushT-v0",
            "episodes": args.episodes,
            "video": str(video_path),
        },
    )
    print(f"Wrote {video_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()

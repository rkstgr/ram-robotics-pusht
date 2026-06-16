# RAM Robotics PushT

First milestone for testing RAM-style post-training on a pretrained robotic action diffusion policy.

## Milestone 1

Create an initial visual presentation of the pretrained `lerobot/diffusion_pusht` policy running in `gym-pusht/PushT-v0`.

Target artifacts:

- `outputs/pusht_pretrained_rollout.mp4`
- `outputs/index.html`
- `outputs/presentation_screenshot.png`

## Setup

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[lerobot]"
```

## Generate The Visual

```bash
python scripts/render_pretrained_pusht.py --episodes 3 --device auto
```

Preview the result:

```bash
python -m http.server 8765 --directory outputs
```

Then open `http://127.0.0.1:8765/index.html`.

## Verified Local Run

This repo has been smoke-tested on Apple Silicon with:

- Python `3.10.19`
- `lerobot==0.3.2`
- `gym-pusht==0.1.6`
- `pymunk==6.11.1`
- Torch MPS available

The first one-episode render produced:

- `avg_max_reward`: `1.0`
- `pc_success`: `100.0`
- video: `23.1s`, `384x384`, `10 FPS`

## Notes

The first implementation target is deliberately small:

- pretrained action-diffusion policy
- dense environment reward
- replayable visual rollout

## RAM Post-Training

This repo now includes a local RAM-style trainer for `lerobot/diffusion_pusht` on
`gym-pusht/PushT-v0`. It adapts RAM from SD3 flow-matching velocity prediction to
LeRobot's DDPM epsilon-prediction action diffusion model.

The pretrained policy diffuses a normalized action horizon, not pixels:

- clean endpoint `x0`: full action chunk with shape `[16, 2]`
- execution slice: indices `1:9`, matching `n_obs_steps=2` and `n_action_steps=8`
- diffusion objective: epsilon prediction over the action horizon
- RAM target:
  `target_eps = ref_eps + reward_multiplier * advantage * (eps - old_eps)`

Reward comes directly from PushT coverage:

- `coverage = intersection_area(block, goal) / goal_area`
- `reward = clip(coverage / 0.95, 0, 1)`
- success is `coverage > 0.95`

The published pretrained LeRobot model-card metrics are:

- `avg_sum_reward`: `104.83847404039778`
- `avg_max_reward`: `0.9551318575760519`
- `pc_success`: `65.4`

Train with the default config:

```bash
.venv/bin/python scripts/train_ram_pusht.py \
  --config configs/ram_pusht.yaml \
  --output-dir outputs/ram_pusht/default
```

Run a small smoke train:

```bash
.venv/bin/python scripts/train_ram_pusht.py \
  --output-dir outputs/ram_pusht/smoke \
  --max-epochs 1 \
  --num-contexts-per-epoch 1 \
  --num-samples-per-context 2 \
  --num-loss-targets-per-sample 1 \
  --loss-batch-size 2 \
  --continuation-chunks 1 \
  --eval-episodes 0 \
  --resume false
```

Evaluate any saved checkpoint without videos:

```bash
.venv/bin/python scripts/evaluate_ram_pusht.py \
  --policy.path outputs/ram_pusht/default/latest \
  --env.type pusht \
  --episodes 100 \
  --seed 1000 \
  --batch-size 1 \
  --render-videos 0
```

Generated RAM run artifacts include:

- `latest/config.json`
- `latest/model.safetensors`
- `training_state.pt`
- `metrics.jsonl`
- `eval/*.json`

## RAM Verification

Implemented checks:

- RAM loss returns finite values on a synthetic diffusion policy.
- With zero advantage and identical train/reference policies, RAM loss is near zero.
- DDPM noising preserves `[B, 16, 2]` action-horizon shape.
- Group-relative advantages are centered per context and finite.
- PushT `reset_to_state` round-trips the reset-compatible state.
- A one-epoch trainer smoke writes a reloadable LeRobot checkpoint.
- The saved smoke checkpoint reloads through the no-video eval CLI.

Local smoke results on this laptop:

- tiny trainer smoke: `epoch_s=17.02s`, `action_clamp_fraction=0.0`
- strict-JSON smoke: `epoch_s=9.71s`, valid `metrics.jsonl`
- one-episode no-video eval from saved checkpoint: `eval_s=55.53s`,
  `avg_max_reward=1.0`, `pc_success=100.0`

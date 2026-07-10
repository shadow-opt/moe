# WIN MoE-CTS Thin-Wall Training

This training chain keeps the existing `win` and `win_flat_slow` tasks unchanged. The new policy still has 48 actor observations and 266 privileged observations. Its six visible command values are:

```text
[vx, vy, yaw, height_mode, climb_mode, wall_height_m * 3]
```

The internal seven-value command buffer retains `heading` at index 3:

```text
[vx, vy, yaw, heading, height_mode, climb_mode, wall_height_m]
```

## Checkpoint Transfer

The first stage must start from the 85k flat policy. On a remote trainer, copy it without changing its filename:

```bash
scp logs/win_flat_slow_moe_cts/Jul05_17-22-50_flat_slow/model_85000.pt \
  USER@TRAINER:/path/to/moe/logs/win_flat_slow_moe_cts/Jul05_17-22-50_flat_slow/
```

`--warmstart_path` strictly loads model parameters only. It deliberately starts new optimizers and iteration zero. `--resume` restores the optimizer and iteration for an interrupted run; do not combine them.

## Stage 1: Acquisition

```bash
python legged_gym/scripts/train.py \
  --task win_close_climb_acquire \
  --headless \
  --warmstart_path logs/win_flat_slow_moe_cts/Jul05_17-22-50_flat_slow/model_85000.pt
```

Acquisition uses 75% wall and 25% flat environments. Wall robots start 0.40-0.45 m from the front face with climb mode already enabled. The default run is 4096 environments, 24 steps per rollout, 5000 iterations, and saves at the 5000-iteration interval plus the final checkpoint.

Resume an interrupted acquisition run:

```bash
python legged_gym/scripts/train.py \
  --task win_close_climb_acquire --headless --resume \
  --load_run RUN_DIRECTORY --checkpoint CHECKPOINT_NUMBER
```

## Stage 2: Transition

Choose an acquisition checkpoint based on evaluation, then warm-start transition training:

```bash
python legged_gym/scripts/train.py \
  --task win_close_climb_transition \
  --headless \
  --warmstart_path logs/win_close_climb_acquire/RUN_DIRECTORY/model_CHECKPOINT.pt
```

`win_close_climb` is an alias of this transition configuration. Transition uses 60% wall and 40% flat environments. Wall robots approach normally from 0.70-1.00 m and switch climb mode once at a randomized 0.40-0.45 m trigger line.

Resume works the same way, using task `win_close_climb_transition` and that experiment's run directory.

## Smoke Runs

Run these before a full job:

```bash
python legged_gym/scripts/train.py --task win_close_climb_acquire --headless \
  --num_envs 64 --max_iterations 1 \
  --warmstart_path logs/win_flat_slow_moe_cts/Jul05_17-22-50_flat_slow/model_85000.pt

python legged_gym/scripts/train.py --task win_close_climb_transition --headless \
  --num_envs 64 --max_iterations 1 \
  --warmstart_path logs/win_close_climb_acquire/RUN_DIRECTORY/model_CHECKPOINT.pt
```

## Evaluation

The default batch evaluation is a fixed 30 cm wall and has no hard pass rate:

```bash
python legged_gym/scripts/evaluate_close_climb.py \
  --task win_close_climb_eval --headless --num_envs 256 \
  --resume --load_run RUN_DIRECTORY --checkpoint CHECKPOINT_NUMBER
```

Run height slices by adding `--eval_wall_height 0.23`, `0.30`, or `0.35`. The report contains success, fall, stuck, timeout, out-of-track, success time, peak hip force, and integrated hip-force violations. FL/FR hip forces are allowed up to 30 N; RL/RR hip forces are penalized above 5 N with a 5x reward multiplier. Flat speed and height tracking use the same checkpoint with `--eval_mode flat`.

## Export

Checkpoint export rebuilds the exact task model and strictly loads its state dict:

```bash
python legged_gym/scripts/export.py \
  --input logs/win_close_climb_transition/RUN_DIRECTORY/model_CHECKPOINT.pt \
  --task win_close_climb_transition \
  --output_dir logs/win_close_climb_transition/exported \
  --export both
```

This writes TorchScript and ONNX policies. The TorchScript policy maintains its CTS history internally; the checkpoint-to-ONNX path exports the stacked-history form used by the ONNX deployment code.

## MuJoCo Acceptance

Place the exported TorchScript policy at the path configured in `deploy/deploy_mujoco/configs/win_close_climb.yaml`, then run:

```bash
python deploy/deploy_mujoco/deploy_win_pt.py \
  --config win_close_climb.yaml --control keyboard
```

Use H for height mode and C for climb mode. With an Xbox controller, use A for height and B for climb. Check ordinary walking, height switching, the climb rising edge, the one-second post-wall tail, and the transition back to ordinary walking. MuJoCo validation is interactive; real-robot deployment is outside this change.

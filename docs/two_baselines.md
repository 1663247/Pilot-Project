# Two-baseline pilot

The active experiment has exactly two baselines, both starting from
`Qwen/Qwen2.5-0.5B-Instruct`: Vanilla RL and World Model RL (VAGEN text adaptation).
The user selected this model for the first short comparison. Qwen3.5 backend
migration is outside this comparison. No self-play/SFT pipeline is included.

## Method

Both use RAGEN PPO/GAE, identical Sokoban states, train seed 10000, evaluation seed
123, batch size 8 (4 groups × 2), maximum 3 turns and 256 generated tokens per turn.
The first queue runs 10 updates per baseline sequentially on one L4.

Vanilla uses the existing task reward and common invalid-action penalty.
World Model additionally requests `<observation>` and `<prediction>` full grids
inside `<think>`, before a single `<answer>` action. During training:

`reward = task_reward + 0.1 * exact_current_grid + 0.1 * exact_next_grid`

The standard invalid-action penalty is applied as before. Credit requires valid
one-action output; next-state credit additionally requires an effective action.
Missing/duplicate tags, partial grids, invalid actions and mismatched grids receive
no associated credit. No credit is given merely for writing a format tag.
Evaluation always uses task reward without the extra world-model bonus. Both task
and optimization reward are logged separately; shaped reward is not a fair measure
of task improvement across the two methods.

[VAGEN](https://github.com/mll-lab-nu/VAGEN) supervises state estimation and transition
modeling with visual observations and a world-model judge. This pilot adapts that
idea to text grids with exact deterministic scoring; it is not an exact paper
reproduction and does not use a supervised world-model warm start.

## Outputs

`results/two_baselines/<run-id>/manifest.json` tracks queued/running/complete/failed.
Each baseline has `resolved.yaml`, `command.json`, `train.log`, `metrics.jsonl`,
`train_cases/<step>.jsonl`, `eval_cases/<step>.jsonl` and checkpoints.
`curves.png` and `curves.svg` are generated after each completed baseline.

Initial and final evaluation use the same four seeds. Per-step loss/reward, task
success, action validity, length, and exact state prediction metrics are available.
A single short run is a pipeline/pilot result, not statistical evidence of improvement.

## Launch

Use the already configured Python 3.12 training environment:

```bash
python scripts/colab/run_two_baselines.py --run-id sokoban-two-baselines-v2 --steps 10
```

The notebook provides launch/status/plot/case cells. Existing result directories
are never overwritten. Preflight verifies both configs and gold-standard world-model
scores through actual environment interaction before GPU training begins.

The first training attempt exposed a legacy case-export assumption (`prompts` is absent in RAGEN multi-turn batches). Case export now uses the original `messages_list`, retaining environment observations and assistant outputs. The matched comparison was restarted from the base checkpoint under run ID `sokoban-two-baselines-v2`; failed-attempt artifacts are retained separately.

## Verified result

`sokoban-two-baselines-v2`: both baselines completed 10 updates with exit code 0,
saved `global_step_10`, and generated `curves.png` / `curves.svg` on Colab.
Final fixed-case evaluation success is 0.0 for both; improvement is not established.
Results have not yet been backed up locally or turned into a case-study report.

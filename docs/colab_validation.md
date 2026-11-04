# Colab validation status

Validated on a connected NVIDIA L4 (23034 MiB, driver 580.82.07).
Notebook kernel: Python 3.13.16. Model/training processes use isolated Python 3.12 environments.

## Completed checks

- Qwen/Qwen3.5-0.8B, official multimodal architecture with text-only inputs:
  torch 2.8.0, Transformers 5.2.0, BF16, SDPA.
- Greedy generation answered `4` to `what is 2 plus 2?`.
- Forward loss: 1.8488622903823853; backward gradient norm: 146.10178171837083.
- Peak allocated GPU memory: 4.957214832305908 GiB.
- Model probe process exited 0. This check has no optimizer step and is not RL evidence.
- Separate reference training environment installed successfully, including `pip check`.
- Reference: torch 2.8.0+cu128, Transformers 4.57.1, vLLM 0.11.0,
  FlashAttention 2.8.1, Python 3.12.
- Real Sokoban and FrozenLake reset/step checks exited 0.

## Dependency corrections

- vLLM 0.11.0 requires setuptools >=77,<80, conflicting with the original setup script's <70 downgrade.
  The Colab reference recipe uses setuptools 79.0.1, which still provides pkg_resources.
- datasets 4.1.1 requires PyArrow >=21; the recipe uses 21.0.0.
- FlashAttention wheel selection must match PyTorch's actual C++ ABI. The Colab installer queries it.
- Fixed verl source commit: d62da4950573d7a4b7ef2362337952e7ab59e78d.
- The reference dependency set passed `uv pip compile` on the remote Python 3.12 runtime.

## PPO reference result

- One-step Sokoban PPO reference passed with Qwen/Qwen2.5-0.5B-Instruct (exit 0).
  Actor and critic both saved `global_step_1` model/optimizer checkpoints.
  Policy loss: 0.06623384; value loss: 1.85792848; actor gradient norm: 148.18161.
  Mean training reward: -0.32500002; task success rate: 0. This establishes execution, not task learning.
  See [recorded metrics](colab_ppo_reference.json).
## Completed two-baseline pilot

Both Qwen2.5-0.5B/Sokoban treatments completed ten PPO updates with exit code 0
and final checkpoints under `sokoban-two-baselines-v2`. Both final evaluation
success rates were 0/4. See [the handoff guide](handoff.md) for current commands
and [the experiment record](two_baselines.md) for details.

## Outstanding

- Qwen3.5 inference/backward support does not establish compatibility with the fixed older verl/vLLM engine.
  Do not upgrade Transformers inside the reference environment or claim the Qwen3.5 PPO path is validated.
- WebShop data/index setup and training remain unvalidated. The historical SPA pipeline has been removed from this project.
- Neither the model probe nor a one-step PPO test establishes convergence or improved task performance.

Remote evidence paths for the current runtime:

- `/content/qwen35-preflight.json`
- `/content/ragen-jobs/model-check.log`
- `/content/reference-install.log` and `/content/reference-install.exit`
- `/content/reference-lock.txt`
- `/content/ragen-jobs/reference-env-check.log`
- `/content/ragen-jobs/ppo-reference-smoke-3.log`

These paths disappear when the Colab runtime is recycled.

The first two PPO launch attempts exposed source-archive omissions (verl model modules and Hydra data configs). The packager was corrected and regression-checked before the successful third launch. No algorithm changes were needed to complete the reference run.

# Pilot Project: handoff and run instructions

## 1. What is ready

The active experiment compares **Vanilla RL** with **World Model RL (VAGEN text
adaptation)**. Both use Qwen/Qwen2.5-0.5B-Instruct and RAGEN PPO on Sokoban.
Both completed 10 updates on one NVIDIA L4, exited with code 0, and saved
`global_step_10` checkpoints. The final four-case evaluation had zero successes
for both methods. This validates training execution, not improvement or convergence.

| Environment | Preset | Verified status |
| --- | --- | --- |
| Sokoban | `_2_sokoban` | Both pilot baselines trained successfully |
| FrozenLake | `_3_frozen_lake` | reset/step passed; training not yet validated |
| WebShop | `_6_webshop` | Preset retained; data/index setup and training pending |

The two-baseline runner currently targets **Sokoban only**. Do not change its tags
to WebShop: the world-model reward scorer compares text grids and does not support
shopping pages. FrozenLake needs its own reviewed configuration and validation.

The historical SPA directory has been removed. Both active baselines use direct RL
without a self-play/SFT stage. Qwen3.5-0.8B passed a separate generation/backward probe only; use
Qwen2.5 for the verified RL workflow.

The local folder is `pilot-project`. Internal imports still use `ragen`, and
Colab commands still use `/content/RAGEN`. These names are intentional.

## 2. Files to use

- `notebooks/training_pipeline.ipynb`: main notebook; setup, launch, status, plots, cases, export.
- `config/pilot_common.yaml`: shared model, seeds, batch size and rollout budget.
- `config/pilot_vanilla.yaml` and `config/pilot_world_model.yaml`: two active treatments.
- `scripts/colab/requirements-reference.txt`: pinned training dependencies.
- `scripts/colab/setup_reference.sh`: isolated Python 3.12 installation.
- `scripts/colab/run_two_baselines.py`: preflight and sequential training queue.
- `scripts/colab/plot_pilot.py`: plots actual metrics from both runs.
- `docs/project_requirements.md`: submission checklist and bonus evidence.

Shared `base`, `envs`, PPO and evaluation configs are retained because the
three environment presets depend on them. Other numbered environment presets
were deleted; the remaining presets are in the root `config/` directory. Historical environment
implementations remain outside the registered environment set. Old experiment
reports, sweep launchers, promotional assets and paper PDFs have been removed.
The trainer still uses diagnostic modules, so their implementation remains.

## 3. Prepare a new Colab runtime

**On the local machine**, open a terminal at the repository root:

```bash
git submodule update --init verl
python3 scripts/package_colab.py
```

Transfer this modified repository to the next person, including `notebooks/` and the
new files; a fresh clone of upstream alone does not contain these changes.
The generated `outputs/ragen-colab-source.tar.gz` contains source and documentation,
not trained checkpoints or downloaded datasets.

**In Colab**, select a GPU runtime. L4 is the tested device. Upload the source
archive to `/content` through the file panel. Open `notebooks/training_pipeline.ipynb`
in Colab, or use the VS Code Colab extension to select the remote GPU kernel.
A local VS Code file and the remote filesystem are separate: local edits are not
automatically synchronized.

Run notebook cells in this order:

1. Hardware and helper cells. Confirm the GPU is visible.
2. Set `RUN_EXTRACT=True` and execute once. It creates `/content/RAGEN` and refuses
   to overwrite an existing repo.
3. Set `RUN_REFERENCE_SETUP=True` and execute once. Then execute
   `status('reference-setup')` until it reports `exit=0`.
4. Set `RUN_REFERENCE_CHECK=True`, execute, and check `status('reference-check')`.
5. Proceed to the training section only after installation and checks succeed.

The notebook kernel may be Python 3.13. Training runs through
`/content/ragen-reference/bin/python` (Python 3.12). The verified stack is Torch
2.8.0+cu128, vLLM 0.11.0, Transformers 4.57.1, FlashAttention 2.8.1. The installer
checks dependencies and selects the FlashAttention wheel using Torch's actual ABI.
Do not mix the old upstream installation scripts with this environment.

For an existing runtime, skip extraction and installation. Run the hardware/helper
cells again after reconnecting, then inspect the existing run.

## 4. Start or inspect training

In the notebook training cell:

```python
RUN_ID = 'sokoban-two-baselines-v3'  # New ID for a new experiment
STEPS = 10                        # Updates per baseline, not total across both
RUN_COMPARISON = True
```

Execute once. Vanilla runs first, then World Model. Both start independently from
the base model. The runner stops if either process fails. Reusing an existing run
ID is refused so previous results cannot be overwritten. Set the launch switch
back to `False` after starting.

Equivalent command **inside the Colab environment**, from `/content/RAGEN`:

```bash
/content/ragen-reference/bin/python scripts/colab/run_two_baselines.py \
  --run-id sokoban-two-baselines-v3 --steps 10
```

Use the notebook's background launcher for long jobs. The command above runs in
the foreground. To validate configurations/reward scoring without launching PPO:

```bash
/content/ragen-reference/bin/python scripts/colab/run_two_baselines.py --preflight-only
```

To inspect the completed pilot, set `RUN_ID='sokoban-two-baselines-v2'` and
`RUN_COMPARISON=False`. Run the status cell. Each baseline should show `complete`,
exit code 0, and metric steps 0 through 10 (step 0 is initial evaluation).
A launch cell returning or the word `running` is not evidence of completion.

For a longer pilot, choose a new run ID and increase STEPS (runner accepts 1–100).
This starts fresh; it does not resume the previous checkpoint. Keep both methods'
shared settings equal. Change rollout budget/seeds in `pilot_common.yaml`, then
synchronize reviewed source changes to Colab before launching. Back up previous
results first. Do not extract a fresh archive over a live training directory.

## 5. Find curves, checkpoints and cases

Completed pilot location **on Colab**:

```text
/content/RAGEN/results/two_baselines/sokoban-two-baselines-v2/
  manifest.json
  curves.png
  curves.svg
  vanilla_rl/
    command.json
    resolved.yaml
    train.log
    metrics.jsonl
    exit.txt
    train_cases/<step>.jsonl
    eval_cases/<step>.jsonl
    checkpoints/global_step_10/
  world_model_rl/
    ...same layout...
```

The status and plot cells show progress and curves. To regenerate the plots:

```bash
/content/ragen-reference/bin/python scripts/colab/plot_pilot.py \
  /content/RAGEN/results/two_baselines/sokoban-two-baselines-v2
```

`actor/pg_loss` is policy loss; `critic/vf_loss` is value loss.
`train/CoordSokoban/task_reward` is task reward per turn;
`critic/rewards/mean` includes reward shaping and penalties.
Compare **unshaped evaluation success and task reward** to assess improvement,
not the World Model optimization reward, which includes extra bonuses.

Use the notebook case cell to inspect initial/final evaluation output.
`train_cases` also stores structured `messages` with intermediate observations,
generated reasoning and actions. Evaluation output may omit intermediate
observations. For the report, select concrete successes/failures, check state
predictions against the grid, and describe what the text actually shows.
Generated reasoning is not proof of internal model mechanisms.

## 6. Export before ending the runtime

Run the notebook export cell with `RUN_EXPORT=True`. By default it archives
logs, configs, curves and cases into `/content/<RUN_ID>-report.tar.gz`.
Set `INCLUDE_CHECKPOINTS=True` for a larger archive containing model checkpoints.
Download the archive through the Colab file panel, or copy it to your persistent
storage. Verify the downloaded archive before releasing the GPU:

```bash
tar -tzf /path/to/sokoban-two-baselines-v2-report.tar.gz
```

Creating an archive inside `/content` alone is not a backup. The completed pilot
artifacts were verified remotely; local backup has not yet been confirmed.
`docs/colab_ppo_reference.json` records the earlier one-step smoke test, not this
10-step comparison. `docs/colab_model_probe.json` is the separate Qwen3.5 probe.

## 7. Troubleshooting

| Symptom | Next action |
| --- | --- |
| No `/content` or no GPU | Select the remote Colab GPU kernel, not local Python. |
| Import/version failure | Read setup log; rerun the pinned installer only after identifying the conflict. Do not upgrade everything. |
| Missing verl modules or Hydra config | Initialize verl locally, rebuild the source archive, and check its contents. |
| Existing result directory | Inspect it or choose a new run ID. Do not delete it to bypass the guard. |
| Failed training | Read that baseline's `train.log` and `exit.txt`; preserve failed-run evidence. |
| Kernel reconnecting | Reconnect to the same runtime and rerun helpers/status; do not launch another training queue. |
| Final success is zero | Execution can still be correct. Inspect actions and reward sparsity; ten updates are not a learning guarantee. |
| World Model bonus stays zero | Inspect exact grid predictions, tags, truncation and effective actions; formatting alone earns no bonus. |

The earlier case-export `prompts` error was fixed by exporting `messages_list`.
The v2 queue completed successfully after that repair. Use the cleaned source,
not cells copied from earlier failed debug attempts.

## 8. Remaining handoff work

Training execution is verified. Curves and raw cases exist remotely, but the
submission still needs artifact backup, curve interpretation, selected visualized
case studies, report and presentation. Improvement and extra-environment bonuses
are unproven. See the requirements checklist for the exact remaining deliverables.

## GitHub repository

Repository: `git@github.com:1663247/Pilot-Project.git`.

```bash
git clone git@github.com:1663247/Pilot-Project.git pilot-project
cd pilot-project
git submodule update --init verl
```

For future WebShop setup, also run `git submodule update --init external/webshop-minimal`.
Both dependencies are pinned submodules, not copied source. Lean and Spatial
submodule registrations are excluded from this pilot. Use the Colab instructions
above after cloning; the repository does not contain trained checkpoints.

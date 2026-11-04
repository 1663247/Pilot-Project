# Pilot Project — LLM Agent RL

**Start here: [handoff and run instructions](docs/handoff.md).**

This project uses RAGEN with two active baselines: **Vanilla RL** and
**World Model RL (VAGEN text adaptation)**. Both completed a ten-step Sokoban PPO
run with Qwen2.5-0.5B-Instruct on one NVIDIA L4. Both saved checkpoints and metrics;
final success was 0/4 for each. Training works; improved task performance is not
established.

Only **Sokoban, FrozenLake, and WebShop** environment presets are retained.
Shared training/evaluation configurations remain; the historical SPA directory has been removed. FrozenLake reset/step was
verified; its training and WebShop setup remain future work.

| File | Purpose |
| --- | --- |
| [Handoff guide](docs/handoff.md) | Install, run, monitor, export, troubleshoot |
| [Colab notebook](notebooks/training_pipeline.ipynb) | Executable setup and training workflow |
| [Two-baseline experiment](docs/two_baselines.md) | Method, matched settings, recorded outcome |
| [Requirements](docs/project_requirements.md) | Course deliverables and bonus status |
| [Validation record](docs/colab_validation.md) | Dependency matrix and earlier smoke checks |

Historical paper PDFs, promotional assets, experiment reports and sweep scripts
have been removed. Framework diagnostic modules remain because the trainer still uses them. Dependency source under `verl/` is unchanged.
See [references](docs/references.md) for attribution and citations.

The local project folder is named `pilot-project`. The internal Python package
remains `ragen`, and the validated Colab path remains `/content/RAGEN`; renaming
those would require separate code and runtime migration.

## Local preparation

From this repository root:

```bash
git submodule update --init verl
python3 scripts/package_colab.py
```

Upload `outputs/ragen-colab-source.tar.gz` to Colab, then follow the notebook.
Do not use **Run All** on old debug notebooks. The cleaned notebook defaults
installation and training switches to `False`.

## Attribution

Built on [RAGEN](https://github.com/RAGEN-AI/RAGEN), with a text-grid adaptation of
[VAGEN](https://github.com/mll-lab-nu/VAGEN). This adaptation is not an exact VAGEN
paper reproduction. See [references](docs/references.md) for RAGEN citations, and
[LICENSE](LICENSE) for repository licensing.

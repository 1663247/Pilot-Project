# Course requirements and evidence

## Required deliverables (priority)

| Requirement | Completion evidence | Current verified status |
| --- | --- | --- |
| 1. Framework runs successfully | Real environment rollout, finite PPO losses/gradients, successful exit and checkpoint | One-step Qwen2.5/Sokoban PPO smoke passed; see `colab_validation.md` and `colab_ppo_reference.json`. Both 10-step baselines completed with exit code 0 under `sokoban-two-baselines-v2`; both saved `global_step_10`. |
| 2. Loss and reward training curves | Per-step JSONL, policy/value loss and task/optimization reward plots | Both runs logged steps 0–10; `curves.png` and `curves.svg` generated on Colab. Local export and report interpretation pending. |
| 3. Visualized case study, especially thinking | Fixed-seed before/after observations, generated reasoning text, actions, outcomes; success/failure analysis | Transcript export implemented; selected cases and analysis still pending. Generated reasoning text is not direct evidence of the model's internal mechanism. |

Active comparison: **Vanilla RL** versus **World Model RL (VAGEN text adaptation)**,
both Qwen2.5-0.5B-Instruct, Sokoban, PPO, matching seeds and budgets.
These are the only active baselines; the historical SPA directory has been removed.

## Bonus priorities

1. **Improve baseline (5–20 pts):** VAGEN adaptation is the proposed improvement.
   Compare unshaped evaluation task reward and success rate with Vanilla. Additional
   training reward alone does not demonstrate improvement. Ten updates and four
   evaluation cases are a pilot; larger evaluation and repeated seeds are needed
   for a robust performance claim. No improvement is claimed yet.
2. **Interesting RL findings (2–5 pts):** inspect prediction accuracy, invalid actions,
   sparse rewards, failure modes, and agreement between generated reasoning and
   actual transitions. Include negative findings if supported by recorded cases.
3. **More environments/baselines (2–5 pts):** both baselines completed 10 PPO updates. FrozenLake reset/step passed but training remains pending;
   WebShop dependencies/data and training remain pending.
4. **New environment (5–20 pts):** not currently planned. Sokoban, FrozenLake and
   WebShop already exist in the repository and do not count as newly added environments.

## Submission checklist

- [ ] Project report (10 pts): setup, reproducible commands/configs, real curves,
      comparison, case studies, limitations, and evidence for any bonus claim.
- [ ] Presentation (10 pts): prioritize pilot findings and case studies.
- [ ] Final-project proposal: 1–2 slides describing the intended agent and data resources.
- [ ] People Page / Group information (needs actual group details and page access).

The supplied assignment states April 28 for report/presentation and Monday of
Week 5 for the joint presentation. The year was not supplied; confirm the course schedule.

Artifacts: `results/two_baselines/<run-id>/` on the Colab runtime. Save results outside
ephemeral `/content` before releasing the runtime. See `two_baselines.md` for the
method and output layout. Failed attempts are retained and are not counted as completed runs.

## Verified short-run outcome

Both final evaluations report success rate 0.0 (four fixed evaluation cases).
This confirms training execution, not a demonstrated performance improvement.
Colab artifacts remain under `/content/RAGEN/results/two_baselines/sokoban-two-baselines-v2/`; local backup and case-study analysis remain pending.

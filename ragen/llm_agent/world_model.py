"""Deterministic, text-grid adaptation of VAGEN state/transition supervision.

This is direct RL reward shaping, not self-play SFT and not an exact VAGEN
reproduction (which uses visual inputs and an LLM judge).
"""
import re

GRID_CHARS = frozenset('#_O√XPSG')


def grid(text):
    rows = [line.strip() for line in str(text).splitlines()
            if line.strip() and set(line.strip()) <= GRID_CHARS]
    if not rows or len({len(row) for row in rows}) != 1:
        return None
    return tuple(rows)


def score_world_model(response, before, after, valid_action, effective_action):
    """Exact full-grid scores; never give prediction credit to a no-op action."""
    matches = {tag: list(re.finditer(fr'<{tag}>(.*?)</{tag}>', response, re.S))
               for tag in ('observation', 'prediction', 'answer')}
    well_formed = all(len(m) == 1 for m in matches.values())
    if well_formed:
        o, p, a = [matches[t][0] for t in ('observation', 'prediction', 'answer')]
        think = list(re.finditer(r'<think>(.*?)</think>', response, re.S))
        well_formed = (len(think) == 1 and think[0].start() <= o.start()
                       and o.end() <= p.start() and p.end() <= think[0].end() <= a.start()
                       and a[1].strip().lower() in {'up', 'down', 'left', 'right'})
    obs_score = pred_score = 0.0
    if well_formed and valid_action:
        before_grid, after_grid = grid(before), grid(after)
        obs_score = float(before_grid is not None and grid(o[1]) == before_grid)
        pred_score = float(effective_action and after_grid is not None and grid(p[1]) == after_grid)
    return {'wm_format_valid': float(well_formed),
            'wm_observation_exact': obs_score, 'wm_prediction_exact': pred_score}


WORLD_MODEL_INSTRUCTION = '''
Before acting, describe the current state and predict the next state for ONE action.
Inside <think>, output <observation> with the COMPLETE current symbol grid, then
reason about ONE action, then output <prediction> with the COMPLETE grid expected
AFTER that action. Put each grid row on its own line; copy all wall/floor/entity
symbols exactly. Close </think>, then output <answer>Up</answer> (or Down/Left/Right).
Use each observation, prediction, and answer block exactly once, in that order.
Correct current-state grids and correct next-state grids receive small training
bonuses. Prediction credit requires an action that changes the environment.
'''

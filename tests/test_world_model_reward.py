"""Reward tests are dependency-free, to exercise scoring before GPU launch."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('world_model', Path(__file__).resolve().parents[1]/'ragen/llm_agent/world_model.py')
wm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wm)

BEFORE = '#####\n#P_X#\n#_O_#\n#####'
AFTER = '#####\n#_PX#\n#_O_#\n#####'

def response(before=BEFORE, after=AFTER, action='Right'):
    return f'<think><observation>{before}</observation>move right<prediction>{after}</prediction></think><answer>{action}</answer>'

class WorldModelRewardTests(unittest.TestCase):
    def test_exact_state_and_transition(self):
        scores=wm.score_world_model(response(), BEFORE, AFTER, True, True)
        self.assertEqual(scores['wm_observation_exact'], 1)
        self.assertEqual(scores['wm_prediction_exact'], 1)

    def test_copying_current_state_is_not_a_prediction(self):
        scores=wm.score_world_model(response(after=BEFORE), BEFORE, AFTER, True, True)
        self.assertEqual(scores['wm_observation_exact'], 1)
        self.assertEqual(scores['wm_prediction_exact'], 0)

    def test_no_op_or_invalid_action_cannot_collect_prediction_credit(self):
        self.assertEqual(wm.score_world_model(response(), BEFORE, AFTER, True, False)['wm_prediction_exact'],0)
        scores=wm.score_world_model(response(), BEFORE, AFTER, False, True)
        self.assertEqual(scores['wm_observation_exact']+scores['wm_prediction_exact'],0)

    def test_ambiguous_or_malformed_outputs_are_rejected(self):
        for r in (response()+f'<prediction>{AFTER}</prediction>',response(action='Right || Left'),response().replace('</think>', ''),response(before='')):
            scores=wm.score_world_model(r, BEFORE, AFTER, True, True)
            self.assertEqual(scores['wm_prediction_exact'] if '<observation></observation>' not in r else scores['wm_observation_exact'],0)

    def test_full_grid_not_partial_rows(self):
        self.assertEqual(wm.score_world_model(response(after='#_PX#'), BEFORE, AFTER, True, True)['wm_prediction_exact'],0)

if __name__=='__main__': unittest.main()

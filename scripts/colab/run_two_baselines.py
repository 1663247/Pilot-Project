"""Bounded, sequential Vanilla/World Model PPO pilot on one GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASELINES = {'vanilla_rl': 'pilot_vanilla', 'world_model_rl': 'pilot_world_model'}


def preflight():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from train import add_dependency_and_validate_config
    from ragen.llm_agent.es_manager import EnvStateManager
    from ragen.llm_agent.ctx_manager import ContextManager
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    for name, config_name in BASELINES.items():
        with initialize_config_dir(config_dir=str(ROOT/'config'), version_base=None):
            cfg=compose(config_name=config_name)
        cfg=add_dependency_and_validate_config(cfg)
        manager=EnvStateManager(cfg, mode='train')
        manager.reset(seed=10000)
        context=ContextManager(cfg,tokenizer)
        assert ('<observation>' in context.prefix_lookup[0]) == cfg.world_model.enabled
        if cfg.world_model.enabled:
            import copy
            env=copy.deepcopy(manager.envs[0]['env'])
            before=manager.rollout_cache[0]['history'][-1]['state']
            for action, label in env.ACTION_LOOKUP.items():
                env.reset(seed=10000)
                after, _, _, info=env.step(action)
                if info.get('action_is_effective'): break
            else: raise AssertionError('No effective test move')
            raw=f'<think><observation>{before}</observation>move<prediction>{after}</prediction></think><answer>{label}</answer>'
            item=dict(env_id=0,actions=[label],llm_response=raw,llm_raw_response=raw)
            manager.step([item])
            scores=manager.rollout_cache[0]['history'][0]['info']
            assert scores['wm_observation_exact']==scores['wm_prediction_exact']==1, scores
            assert abs(scores['wm_bonus']-0.2)<1e-6, scores
            manager.mode='val'
            manager.reset(seed=10000)
            manager.step([item])
            assert manager.rollout_cache[0]['history'][0]['info']['wm_bonus']==0
        for entry in manager.envs: entry['env'].close()
        print('Preflight passed:',name,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id')
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    preflight()
    if args.preflight_only: return
    assert args.run_id and args.run_id.replace('-','').replace('_','').isalnum(), 'Safe run ID required'
    assert 1<=args.steps<=100, 'Use a bounded pilot budget'
    out=ROOT/'results'/'two_baselines'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(model='Qwen/Qwen2.5-0.5B-Instruct',environment='Sokoban',algorithm='PPO/GAE',
                  steps_per_baseline=args.steps,train_seed=10000,val_seed=123,
                  batch_size=8,max_turn=3,response_tokens=256,
                  adaptation='Text grids + deterministic exact-state reward; not original VAGEN reproduction',
                  runs={name:dict(status='queued') for name in BASELINES})
    def save():
        tmp=out/'manifest.tmp'; tmp.write_text(json.dumps(manifest,indent=2)+'\n'); tmp.replace(out/'manifest.json')
    save()
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',RAY_ADDRESS='local',
             TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    for name, config_name in BASELINES.items():
        run=out/name; run.mkdir()
        command=[sys.executable,str(ROOT/'train.py'),'--config-name',config_name,
                 f'trainer.total_training_steps={args.steps}',
                 f'trainer.experiment_name={args.run_id}-{name}',
                 f'trainer.default_local_dir={run}/checkpoints',
                 f'trainer.metrics_jsonl={run}/metrics.jsonl',
                 f'trainer.rollout_data_dir={run}/train_cases',
                 f'trainer.validation_data_dir={run}/eval_cases',
                 'trainer.resume_mode=disable']
        (run/'command.json').write_text(json.dumps(command,indent=2))
        with (run/'resolved.yaml').open('w') as f:
            subprocess.run(command+['--cfg','job','--resolve'],cwd=ROOT,env=env,stdout=f,check=True)
        manifest['runs'][name].update(status='running',started=time.time(),directory=str(run));save()
        print('STARTING',name,run,flush=True)
        with (run/'train.log').open('w') as f:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        (run/'exit.txt').write_text(str(result.returncode)+'\n')
        manifest['runs'][name].update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,finished=time.time());save()
        subprocess.run([sys.executable,str(ROOT/'scripts/colab/plot_pilot.py'),str(out)],check=False)
        if result.returncode: raise SystemExit(result.returncode)
    print('BOTH BASELINES COMPLETE',out,flush=True)

if __name__=='__main__': main()

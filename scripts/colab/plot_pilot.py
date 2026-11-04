"""Plot actual scalar logs. Task rewards are separate from shaped RL rewards."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1])
series={}
for name in ('vanilla_rl','world_model_rl'):
    file=root/name/'metrics.jsonl'
    if file.exists(): series[name]=[json.loads(line) for line in file.read_text().splitlines() if line.strip()]
plots=[('actor/pg_loss','Policy loss'),('critic/vf_loss','Value loss'),
       ('train/CoordSokoban/task_reward','Task reward / turn (comparable)'),
       ('critic/rewards/mean','Optimization reward (WM includes bonus)'),
       ('val-env/CoordSokoban/success','Evaluation success'),
       ('train/CoordSokoban/wm_prediction_exact','Exact next-state prediction')]
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for ax,(key,title) in zip(axes.flat,plots):
    plotted=False
    for name,records in series.items():
        points=[(r['step'],r['metrics'][key]) for r in records if r['metrics'].get(key) is not None]
        if points:
            ax.plot(*zip(*points),marker='o',label=name);plotted=True
    ax.set(title=title,xlabel='PPO step');ax.grid(alpha=.2)
    if plotted: ax.legend()
    else: ax.text(.5,.5,'No observations yet',ha='center',transform=ax.transAxes)
fig.suptitle('Short pilot: single seed; no statistical improvement claim')
fig.savefig(root/'curves.png',dpi=160)
fig.savefig(root/'curves.svg')
print(root/'curves.png')

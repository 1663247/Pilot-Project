"""Optional durable scalar metrics; does not change policy optimization."""
import json
import math
from numbers import Real
from pathlib import Path


def append_metrics(path, data, step):
    if not path:
        return
    values = {str(k): (float(v) if math.isfinite(float(v)) else None)
              for k, v in data.items() if isinstance(v, Real)}
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('a') as f:
        f.write(json.dumps({'step': int(step), 'metrics': values}, allow_nan=False)+'\n')


def case_transcripts(messages_list):
    """Keep the original turn order and observations for multi-turn case studies."""
    chats=[chat.tolist() if hasattr(chat,'tolist') else list(chat) for chat in messages_list]
    inputs=['\n'.join(str(m.get('content','')) for m in chat[:2]) for chat in chats]
    outputs=['\n\n'.join(f"[{m.get('role','unknown')}]\n{m.get('content','')}" for m in chat) for chat in chats]
    return inputs,outputs,chats

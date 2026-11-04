"""Real Qwen3.5 forward/backward/generation check; this is not an RL run."""
import json
import math
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    p.add_argument('--output', default='/content/qwen35-preflight.json')
    args = p.parse_args()
    assert torch.cuda.is_available(), 'CUDA GPU required'
    assert torch.cuda.is_bf16_supported(), 'Use L4/A100 or newer for this BF16 preflight'
    torch.manual_seed(123)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    # Text-only inputs; retain official checkpoint architecture and weight names.
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Answer with one word: what is 2 plus 2?'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt').to('cuda')
    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print('Generated:', response, flush=True)
    model.train()
    model.config.use_cache = False
    # Verify loss/backprop without allocating optimizer states for all parameters.
    result = model(**inputs, labels=inputs.input_ids, use_cache=False)
    assert torch.isfinite(result.loss), 'Nonfinite loss'
    result.loss.backward()
    grad_sq = sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.grad is not None)
    grad_norm = math.sqrt(grad_sq)
    assert math.isfinite(grad_norm) and grad_norm > 0, 'Invalid gradients'
    report = dict(status='passed', scope='HF generation + backward only; not RL',
                  torch=torch.__version__, transformers=transformers.__version__,
                  gpu=torch.cuda.get_device_name(), model=args.model, response=response,
                  loss=float(result.loss.detach()), grad_norm=grad_norm,
                  peak_gb=torch.cuda.max_memory_allocated()/2**30)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()

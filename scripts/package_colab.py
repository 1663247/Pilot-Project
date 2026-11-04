"""Package allowlisted source files, including initialized verl, for Colab upload."""
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.ipynb', '.py', '.sh', '.yaml', '.yml', '.json', '.toml', '.txt', '.md', '.jinja', '.j2', '.html', '.css', '.js'}
EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', 'outputs', 'results', 'checkpoints', 'sftdata', 'sftckpt', 'model_saving', 'logs', 'log', 'wandb'}

def main():
    assert (ROOT/'verl/verl/__init__.py').is_file(), 'Initialize verl first: git submodule update --init verl'
    out = ROOT/'outputs/ragen-colab-source.tar.gz'
    out.parent.mkdir(exist_ok=True)
    files=[]
    for base in ('ragen', 'notebooks', 'config', 'scripts', 'verl', 'docs'):
        for p in (ROOT/base).rglob('*'):
            rel=p.relative_to(ROOT)
            if (p.is_file() and not p.is_symlink() and (p.suffix in ALLOWED or p.name == 'version')
                and not any(part in EXCLUDED or part.startswith('.') for part in rel.parts)
                and p.name not in {'credentials.json', 'secrets.json', 'token.json'}):
                files.append(p)
    files += [ROOT/'setup.py', ROOT/'train.py', ROOT/'README.md', ROOT/'LICENSE']
    with tarfile.open(out, 'w:gz') as t:
        for p in sorted(set(files)):
            t.add(p, arcname=str(Path('RAGEN')/p.relative_to(ROOT)), recursive=False)
    print(out, out.stat().st_size, 'bytes;', len(set(files)), 'source files')

if __name__ == '__main__':
    main()

"""Fetch exact upstream model revisions and apply the experiment's recorded adaptations."""
import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for entry in json.loads((ROOT / 'tools/upstream.json').read_text()):
    target = ROOT / entry['path']
    if target.exists():
        raise SystemExit(f'{target} exists; inspect/remove it yourself before rerunning (nothing overwritten).')
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', entry['url'], str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', entry['commit']], check=True)
    if 'patch' in entry:
        subprocess.run(['git', '-C', str(target), 'apply', str(ROOT / entry['patch'])], check=True)
print('Upstream source ready. Dataset images and model weights are separate requirements.')

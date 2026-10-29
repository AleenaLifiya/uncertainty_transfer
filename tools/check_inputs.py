"""Report missing inputs without importing models or modifying files."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
required = {
    'brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData',
    'brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData',
    'IXI_tiny/image', 'IXI_tiny/label',
    'IR3C_more_comparisons/predictions/saint_msr_cache_index.csv',
    'IR3C_experiments_V2/stage2_finetune/IXI/checkpoints/SAINT_rfn/SAINT_rfn_step000654.pt',
    'runs/brats_ir3c_20260902/checkpoints/arssr_slowlr/arssr_slowlr_step000000.pt',
}
required.update(x['path'] for x in json.loads((ROOT/'tools/upstream.json').read_text()))
for rel in sorted(required):
    print(('OK      ' if (ROOT/rel).exists() else 'MISSING ')+rel)
print('\nFrozen selected-checkpoint paths (outputs must be trained or restored):')
paths=set()
def visit(obj):
    if isinstance(obj, dict):
        for v in obj.values(): visit(v)
    elif isinstance(obj, list):
        for v in obj: visit(v)
    elif isinstance(obj, str) and obj.endswith(('.pt','.pth','.ckpt')) and '${UT_ROOT}' in obj:
        paths.add(obj.replace('${UT_ROOT}/',''))
for f in (ROOT/'path_templates').rglob('*.json.in'):
    visit(json.loads(f.read_text()))
for rel in sorted(paths):
    print(('OK      ' if (ROOT/rel).exists() else 'MISSING ')+rel)
missing=sum(not (ROOT/p).exists() for p in required | paths)
print(f'\n{missing} missing inputs/checkpoints. See docs/REPRODUCING.md; saved-metric analysis remains available.')
raise SystemExit(1 if missing else 0)

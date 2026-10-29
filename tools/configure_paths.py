"""Materialize machine-local paths from versioned templates. Run after moving the clone."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for template in (root / 'path_templates').rglob('*.in'):
    target = root / str(template.relative_to(root / 'path_templates'))[:-3]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.read_text().replace('${UT_ROOT}', str(root)))
print('Configured paths for', root)

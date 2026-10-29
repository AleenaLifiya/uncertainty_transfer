"""
comparisons/
============
IR3C --- adding I3Net (TMI 2024) and ArSSR (J-BHI 2022) to the IR3 through-plane
slice-reconstruction comparison on BraTS-GLI 2023 T2-FLAIR.

Design rule (ir3_comparisons.md §17): METHOD LOGIC lives in the upstream repositories
(`I3Net/`, `ArSSR/`) and is imported unchanged. Everything in this package is
DATASET/INTERFACE ADAPTATION plus the shared protocol that every method must obey.

Nothing here modifies `image_residual/`, `ir2_*`, `X-Diffusion/` or either upstream repo.
"""
from . import protocol  # noqa: F401

# Attribution and license boundaries

This is an extraction of the experiment workspace, not a new implementation of SAINT, ArSSR, I3Net, or X-Diffusion. Source hashes before portability edits are recorded in `docs/source_manifest.json`.

- SAINT: https://github.com/cpeng93/SAINT — model fetched at the exact revision in `tools/upstream.json`; the experiment's geometry/interface adaptations are applied from `tools/patches/SAINT.patch`.
- ArSSR: https://github.com/iwuqing/ArSSR — fetched at the experiment's pinned revision.
- I3Net: https://github.com/eeeric-code/I3Net — fetched for compatibility with the original BraTS evaluation, which also evaluates legacy controls.
- DCSRN: https://github.com/YunzeMan/DCSRN — fetched for shared legacy adapters, with the recorded patch. Not a final headline-paper method.
- X-Diffusion: https://github.com/EmmanuelleB985/X-Diffusion — selected locally used modules are included, with the original `X-Diffusion/LICENSE`. The workspace was based on revision `866b974608771f6db1609e465cba2bd882589ad8`; local experiment additions and modifications are retained.

Upstream model repositories are fetched separately so their own notices and terms accompany the source. Consult those terms before reuse. No license for external datasets or trained weights is conveyed by this repository. No new blanket license has been selected for the authors' research code.

No image datasets, model weights, virtual environments, credentials, or original workstation job-control scripts are bundled. Per-case numerical results use dataset subject identifiers. The manuscript itself, editing notes, presentations, and medical-image figures are not part of this code release.

"""Thumbnails for Fig. 1 v5 (2026-10-29): same patient/plane/mask as Fig. 3 (planes4_clustered.npz, clustered x4).
input = sparse clustered input; width maps = 90 % interval width (hi - lo), viridis 0-0.35 as in Fig. 3, acquired rows grey,
background black. Without and with geometry: SAINT-P learned, ArSSR-P learned, linear post hoc (v6: both versions)."""

from project_paths import root_path
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

I = root_path('for_ISBI')
OUT = os.path.join(I, "manuscript_4page", "figures", "thumbs_v5")
z = dict(np.load(os.path.join(I, "exp_uniform_arssr", "results", "qual", "planes4_clustered.npz")))
fg, acq = z["fg"].astype(bool), z["acq"].astype(bool)
WMAX = 0.35


def up(a, k=4):
    return np.kron(a, np.ones((k, k) + (() if a.ndim == 2 else (1,))))


g = np.clip(z["sparse"], 0, 1)
plt.imsave(os.path.join(OUT, "input.png"), up(np.stack([g] * 3, -1)))
for name, key in (("saint", "SAINT_P__learned"), ("arssr", "ArSSR_P__learned"), ("posthoc", "linear__ph_nogeo"),
                  ("saint_geo", "SAINT_P__learned_geo"), ("arssr_geo", "ArSSR_P__learned_geo"), ("posthoc_geo", "linear__ph_geo")):
    w = np.clip((z[key + "_hi"] - z[key + "_lo"]) / WMAX, 0, 1)
    rgb = plt.get_cmap("viridis")(w)[..., :3]
    rgb[~fg] = 0
    rgb[acq & fg] = (0.42, 0.43, 0.46)
    plt.imsave(os.path.join(OUT, f"{name}.png"), up(rgb))
print("ok", g.shape)

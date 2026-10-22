"""
Category image signatures
=========================

A curated set that separates two models may also differ from the pool
it was drawn from in plain image statistics.  Measuring the spatial
frequency and chromaticity of each category is how that is checked.

This example writes a small synthetic image set so it runs anywhere;
point :func:`~repscreen.signatures.analysis.dataset_signatures` at
your own dataset root to use real images.
"""

# %%
# A small image set
# -----------------
#
# Three categories of eight images each.  The categories differ in
# how much high spatial frequency they carry and in their hue, so
# their signatures should come out visibly different.

import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from repscreen.plotting.signatures import plot_category_signature
from repscreen.signatures.analysis import dataset_signatures
from repscreen.signatures.spectra import downsample, log_magnitude

SIZE = 64
root = Path(tempfile.mkdtemp()) / "dataset"
rng = np.random.default_rng(0)

grid_y, grid_x = np.mgrid[0:SIZE, 0:SIZE]
recipes = {
    "0000_coarse": (4.0, (0.9, 0.4, 0.3)),
    "0001_medium": (12.0, (0.3, 0.8, 0.4)),
    "0002_fine": (24.0, (0.3, 0.4, 0.9)),
}
for name, (frequency, tint) in recipes.items():
    (root / name).mkdir(parents=True)
    for index in range(8):
        phase = rng.uniform(0, 2 * np.pi)
        grating = 0.5 + 0.4 * np.sin(
            2 * np.pi * frequency * grid_x / SIZE + phase
        )
        coloured = grating[:, :, None] * np.array(tint)
        image = (255 * np.clip(coloured, 0, 1)).astype(np.uint8)
        cv2.imwrite(
            str(root / name / f"{name}_{index:03d}.png"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

print(f"wrote {len(recipes)} categories under {root.name}/")

# %%
# Computing the signatures
# ------------------------
#
# Every exemplar is framed to the same square so the spectra can be
# averaged, and its pixels are sampled in CIELab and pooled across
# the category.

signatures = dataset_signatures(
    root, target_size=SIZE, n_samples=2000, seed=0
)
for name, signature in signatures.items():
    print(
        f"{name:14s} {signature.n_images} images, "
        f"{len(signature.chroma_a)} sampled pixels"
    )

# %%
# One category's signature
# ------------------------
#
# The left panel is the mean Fourier spectrum, with low spatial
# frequencies at the centre.  The right panel scatters the sampled
# pixels in the a*-b* plane, each drawn in its own colour.

figure = plot_category_signature(
    signatures["0000_coarse"], n_plot=4000, seed=0
)
plt.show()

# %%
# Comparing the categories
# ------------------------
#
# Laying the spectra side by side shows the grating frequency: the
# energy of the coarse category sits close to the centre and moves
# outwards as the gratings get finer.

figure, axes = plt.subplots(1, len(signatures), figsize=(10, 3.5))
for panel, (name, signature) in enumerate(signatures.items()):
    axes[panel].imshow(
        downsample(log_magnitude(signature.spectrum)), cmap="grey"
    )
    axes[panel].set_title(name)
    axes[panel].axis("off")
figure.suptitle("Mean Fourier spectrum per category")
figure.tight_layout()
plt.show()

# %%
# Chromaticity side by side
# -------------------------
#
# Pooling each category's sampled pixels in the a*-b* plane separates
# them by hue, which is the second axis a selection can be checked
# against.

figure, axes = plt.subplots(figsize=(6, 6))
for name, signature in signatures.items():
    axes.scatter(
        signature.chroma_a[::20],
        signature.chroma_b[::20],
        s=6,
        alpha=0.5,
        label=name,
    )
axes.axhline(0, color="k", linewidth=0.8, alpha=0.5)
axes.axvline(0, color="k", linewidth=0.8, alpha=0.5)
axes.set_xlabel("a* (green to red)")
axes.set_ylabel("b* (blue to yellow)")
axes.set_title("Pooled chromaticity per category")
axes.set_aspect("equal")
axes.legend()
figure.tight_layout()
plt.show()

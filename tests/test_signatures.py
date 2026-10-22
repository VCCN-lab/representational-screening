"""Tests for the repscreen.signatures modules."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from repscreen.plotting.signatures import plot_category_signature
from repscreen.signatures.analysis import (
    CategorySignature,
    category_signature,
    dataset_signatures,
    load_signatures,
    save_signatures,
)
from repscreen.signatures.chromaticity import lab_samples, lab_to_rgb
from repscreen.signatures.images import (
    load_category_images,
    resize_and_center_crop,
)
from repscreen.signatures.spectra import (
    downsample,
    fourier_spectrum,
    log_magnitude,
)

from .conftest import N_CATEGORIES, N_PER_CATEGORY

TARGET = 16


def test_resize_and_center_crop_returns_a_square():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    result = resize_and_center_crop(image, target_size=10)
    assert result.shape == (10, 10, 3)


def test_resize_and_center_crop_scales_on_the_smaller_side():
    image = np.zeros((40, 20, 3), dtype=np.uint8)
    result = resize_and_center_crop(image, target_size=10)
    assert result.shape == (10, 10, 3)


def test_resize_and_center_crop_leaves_a_matching_square_alone():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
    result = resize_and_center_crop(image, target_size=12)
    assert np.array_equal(result, image)


def test_resize_and_center_crop_takes_the_middle():
    image = np.zeros((10, 30), dtype=np.uint8)
    image[:, 10:20] = 255
    result = resize_and_center_crop(image, target_size=10)
    assert np.all(result == 255)


def test_resize_and_center_crop_rejects_a_non_positive_size():
    with pytest.raises(ValueError, match="target_size"):
        resize_and_center_crop(np.zeros((4, 4, 3)), target_size=0)


def test_load_category_images_reads_every_image(
    image_tree, category_names
):
    images, paths = load_category_images(
        image_tree / category_names[0], target_size=TARGET
    )
    assert len(images) == N_PER_CATEGORY
    assert len(paths) == N_PER_CATEGORY
    assert images[0].shape == (TARGET, TARGET, 3)


def test_load_category_images_limits_the_count(
    image_tree, category_names
):
    images, _ = load_category_images(
        image_tree / category_names[0],
        n_exemplars=2,
        target_size=TARGET,
    )
    assert len(images) == 2


def test_load_category_images_sorts_the_paths(
    image_tree, category_names
):
    _, paths = load_category_images(
        image_tree / category_names[0], target_size=TARGET
    )
    assert paths == sorted(paths)


def test_load_category_images_rejects_a_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError, match="absent"):
        load_category_images(tmp_path / "absent")


def test_load_category_images_warns_on_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.warns(UserWarning, match="no images"):
        images, paths = load_category_images(empty)
    assert images == []
    assert paths == []


def test_fourier_spectrum_keeps_the_image_shape():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (TARGET, TARGET, 3), dtype=np.uint8)
    assert fourier_spectrum(image).shape == (TARGET, TARGET)


def test_fourier_spectrum_accepts_a_grayscale_image():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (TARGET, TARGET), dtype=np.uint8)
    assert fourier_spectrum(image).shape == (TARGET, TARGET)


def test_fourier_spectrum_is_complex():
    image = np.zeros((TARGET, TARGET), dtype=np.uint8)
    assert np.iscomplexobj(fourier_spectrum(image))


def test_fourier_spectrum_of_a_constant_image_has_one_component():
    image = np.full((TARGET, TARGET), 128, dtype=np.uint8)
    magnitude = np.abs(fourier_spectrum(image))
    assert np.count_nonzero(magnitude > 1e-6) == 1


def test_fourier_spectrum_matches_the_transform_order():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (TARGET, TARGET), dtype=np.uint8)
    expected = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image))
    )
    assert np.allclose(fourier_spectrum(image), expected)


def test_log_magnitude_is_finite_for_a_zero_entry():
    spectrum = np.zeros((4, 4), dtype=complex)
    assert np.all(np.isfinite(log_magnitude(spectrum)))


def test_log_magnitude_matches_the_logarithm():
    rng = np.random.default_rng(4)
    spectrum = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    assert np.allclose(
        log_magnitude(spectrum), np.log(np.abs(spectrum))
    )


def test_downsample_halves_each_side_by_default():
    values = np.arange(64.0).reshape(8, 8)
    assert downsample(values).shape == (4, 4)


def test_downsample_honours_the_factor():
    values = np.arange(64.0).reshape(8, 8)
    assert downsample(values, factor=4).shape == (2, 2)


def test_downsample_rejects_a_factor_below_one():
    with pytest.raises(ValueError, match="factor"):
        downsample(np.zeros((8, 8)), factor=0)


def test_lab_samples_returns_three_channels():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    lightness, chroma_a, chroma_b = lab_samples(image, n_samples=10)
    assert lightness.shape == chroma_a.shape == chroma_b.shape
    assert len(lightness) == 10


def test_lab_samples_returns_every_pixel_when_asked_for_more():
    rng = np.random.default_rng(6)
    image = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    lightness, _, _ = lab_samples(image, n_samples=1000)
    assert len(lightness) == 16


def test_lab_samples_lightness_is_in_range():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    lightness, _, _ = lab_samples(image, n_samples=20)
    assert np.all(lightness >= 0.0)
    assert np.all(lightness <= 100.0)


def test_lab_samples_of_grey_has_no_chroma():
    # A neutral sRGB grey carries no hue, but converting it against
    # the D65 white point leaves a residual of a few thousandths of
    # an a*/b* unit, so the tolerance is set above that.
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    _, chroma_a, chroma_b = lab_samples(image, n_samples=10)
    assert np.allclose(chroma_a, 0.0, atol=0.01)
    assert np.allclose(chroma_b, 0.0, atol=0.01)


def test_lab_samples_is_reproducible():
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    left = lab_samples(image, n_samples=10, seed=1)
    right = lab_samples(image, n_samples=10, seed=1)
    assert np.allclose(left[0], right[0])


def test_lab_samples_rejects_a_non_rgb_image():
    with pytest.raises(ValueError, match="three colour channels"):
        lab_samples(np.zeros((8, 8)), n_samples=4)


def test_lab_to_rgb_is_bounded():
    rng = np.random.default_rng(9)
    lightness = rng.uniform(0, 100, 20)
    chroma_a = rng.uniform(-100, 100, 20)
    chroma_b = rng.uniform(-100, 100, 20)
    colours = lab_to_rgb(lightness, chroma_a, chroma_b)
    assert colours.shape == (20, 3)
    assert np.all(colours >= 0.0)
    assert np.all(colours <= 1.0)


def test_lab_to_rgb_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        lab_to_rgb(np.zeros(3), np.zeros(3), np.zeros(4))


def test_category_signature_averages_the_spectra(
    image_tree, category_names
):
    signature = category_signature(
        image_tree / category_names[0],
        target_size=TARGET,
        n_samples=10,
        seed=0,
    )
    assert isinstance(signature, CategorySignature)
    assert signature.category == category_names[0]
    assert signature.spectrum.shape == (TARGET, TARGET)
    assert signature.n_images == N_PER_CATEGORY


def test_category_signature_pools_the_chromaticity(
    image_tree, category_names
):
    signature = category_signature(
        image_tree / category_names[0],
        target_size=TARGET,
        n_samples=10,
        seed=0,
    )
    expected = N_PER_CATEGORY * 10
    assert len(signature.lightness) == expected
    assert len(signature.chroma_a) == expected
    assert len(signature.chroma_b) == expected


def test_category_signature_spectrum_is_the_mean(
    image_tree, category_names
):
    directory = image_tree / category_names[0]
    signature = category_signature(
        directory, target_size=TARGET, n_samples=5, seed=0
    )
    images, _ = load_category_images(directory, target_size=TARGET)
    expected = np.mean(
        [fourier_spectrum(image) for image in images], axis=0
    )
    assert np.allclose(signature.spectrum, expected)


def test_dataset_signatures_covers_every_category(image_tree):
    signatures = dataset_signatures(
        image_tree, target_size=TARGET, n_samples=5, seed=0
    )
    assert len(signatures) == N_CATEGORIES


def test_dataset_signatures_limits_the_category_count(image_tree):
    signatures = dataset_signatures(
        image_tree,
        n_categories=2,
        target_size=TARGET,
        n_samples=5,
        seed=0,
    )
    assert len(signatures) == 2


def test_dataset_signatures_keys_are_category_names(
    image_tree, category_names
):
    signatures = dataset_signatures(
        image_tree, target_size=TARGET, n_samples=5, seed=0
    )
    assert sorted(signatures) == sorted(category_names)


def test_dataset_signatures_rejects_a_missing_root(tmp_path):
    with pytest.raises(NotADirectoryError, match="absent"):
        dataset_signatures(tmp_path / "absent")


def test_signatures_round_trip(image_tree, tmp_path):
    signatures = dataset_signatures(
        image_tree,
        n_categories=2,
        target_size=TARGET,
        n_samples=5,
        seed=0,
    )
    path = save_signatures(signatures, tmp_path / "signatures.pkl")
    loaded = load_signatures(path)
    assert sorted(loaded) == sorted(signatures)
    first = sorted(signatures)[0]
    assert np.allclose(
        loaded[first].spectrum, signatures[first].spectrum
    )


def test_load_signatures_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent"):
        load_signatures(tmp_path / "absent.pkl")


def test_plot_category_signature_has_two_panels(
    image_tree, category_names
):
    signature = category_signature(
        image_tree / category_names[0],
        target_size=TARGET,
        n_samples=20,
        seed=0,
    )
    figure = plot_category_signature(signature)
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2


def test_plot_category_signature_labels_the_chromaticity_axes(
    image_tree, category_names
):
    signature = category_signature(
        image_tree / category_names[0],
        target_size=TARGET,
        n_samples=20,
        seed=0,
    )
    figure = plot_category_signature(signature)
    assert "a*" in figure.axes[1].get_xlabel()
    assert "b*" in figure.axes[1].get_ylabel()

"""Tests for perceptual hash and region-based screenshot comparison."""

from PIL import Image

from src.device.adb import ADBController


def test_phash_identical():
    img = Image.new("RGB", (200, 200), "red")
    h1 = ADBController.compute_phash(img)
    h2 = ADBController.compute_phash(img)
    assert h1 == h2
    assert ADBController.phash_distance(h1, h2) == 0
    assert ADBController.phash_similarity(h1, h2) == 1.0


def test_phash_different():
    """Two structurally different images should have different hashes."""
    from PIL import ImageDraw

    img1 = Image.new("RGB", (200, 200), "white")
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([0, 0, 100, 200], fill="black")  # left half black

    img2 = Image.new("RGB", (200, 200), "white")
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([0, 0, 200, 100], fill="black")  # top half black

    h1 = ADBController.compute_phash(img1)
    h2 = ADBController.compute_phash(img2)
    assert ADBController.phash_distance(h1, h2) > 0


def test_phash_similar():
    """Slightly modified images should have small distance."""
    from PIL import ImageDraw

    img1 = Image.new("RGB", (200, 200), "white")
    img2 = img1.copy()
    draw = ImageDraw.Draw(img2)
    draw.rectangle([50, 50, 60, 60], fill="black")  # tiny change

    h1 = ADBController.compute_phash(img1)
    h2 = ADBController.compute_phash(img2)
    sim = ADBController.phash_similarity(h1, h2)
    assert sim > 0.8  # very similar


def test_phash_similarity_range():
    img = Image.new("RGB", (100, 100), "green")
    h = ADBController.compute_phash(img)
    sim = ADBController.phash_similarity(h, h)
    assert 0.0 <= sim <= 1.0

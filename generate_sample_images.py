"""
generate_sample_images.py
==========================
Generates 20 SYNTHETIC placeholder images (5 per category: good, blurry,
dark, glare) so the pipeline can be demonstrated end-to-end without a
phone on hand.

IMPORTANT: These are ridge-pattern simulations, not real fingerprints.
Before final submission, replace the contents of data/good, data/blurry,
data/dark, data/glare with 5 real phone photos each, per Part D of the
assignment. Keep this script only to show your reviewer how the harness
was bootstrapped, or delete it — it is not one of the graded deliverables.
"""

import numpy as np
import cv2
from pathlib import Path

OUT_ROOT = Path(__file__).parent / "data"
RNG = np.random.default_rng(42)


def synthesize_fingerprint(size=600, freq=0.25, curve=0.0025, seed=0):
    """Creates a plausible ridge-like pattern using a sinusoidal orientation
    field, similar in spirit to how synthetic fingerprint generators work."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.uniform(-40, 40), size / 2 + rng.uniform(-40, 40)
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    ridge = np.sin(2 * np.pi * freq * r / (1 + curve * r) + 3 * theta * 0.05)
    ridge = (ridge - ridge.min()) / (ridge.max() - ridge.min())
    img = (ridge * 200 + 30).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # add mild sensor noise
    noise = rng.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def apply_finger_mask(gray, coverage=0.55, seed=0):
    """Composites the ridge pattern onto a mid-grey background inside an
    ellipse (simulated finger silhouette) so ROI logic has something
    meaningful to detect, with the remainder as background."""
    rng = np.random.default_rng(seed)
    h, w = gray.shape
    canvas = np.full((h, w), 210, dtype=np.uint8)  # background
    background_noise = rng.normal(0, 4, (h, w)).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + background_noise, 0, 255).astype(np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    axes = (int(w * 0.5 * np.sqrt(coverage)), int(h * 0.42 * np.sqrt(coverage)))
    center = (w // 2 + rng.integers(-15, 15), h // 2 + rng.integers(-15, 15))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    out = canvas.copy()
    out[mask == 255] = gray[mask == 255]
    return out, mask


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_good(seed):
    ridge = synthesize_fingerprint(seed=seed)
    composed, _ = apply_finger_mask(ridge, coverage=0.6, seed=seed)
    return to_bgr(composed)


def make_blurry(seed):
    img = make_good(seed)
    return cv2.GaussianBlur(img, (25, 25), 12)


def make_dark(seed):
    img = make_good(seed).astype(np.float32)
    img = img * 0.22
    return np.clip(img, 0, 255).astype(np.uint8)


def make_glare(seed):
    img = make_good(seed)
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.circle(overlay, (w // 2 + 30, h // 2 - 20), int(w * 0.28), (255, 255, 255), -1)
    return cv2.addWeighted(overlay, 0.85, img, 0.15, 0)


def main():
    generators = {
        "good": make_good,
        "blurry": make_blurry,
        "dark": make_dark,
        "glare": make_glare,
    }
    for category, fn in generators.items():
        folder = OUT_ROOT / category
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            img = fn(seed=hash((category, i)) % (2**31))
            path = folder / f"{category}_{i+1:02d}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"wrote {path}")


if __name__ == "__main__":
    main()

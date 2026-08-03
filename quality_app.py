"""
quality_app.py
===============
Streamlit interface for the Fingerprint Quality Assessment pipeline
(Assignment 4, Part E).

Run with:
    streamlit run quality_app.py

Features:
- Drag-and-drop image upload (jpg/jpeg/png)
- Per-metric PASS/FAIL badges
- Large composite score, green if >= 60 else red
- Prominent guidance message
- Sidebar sliders for all 5 thresholds (live-adjustable, not hardcoded)
- Defensive handling of bad uploads (corrupt file, non-image file, huge file)
"""

import time

import cv2
import numpy as np
import streamlit as st

from quality_assessment import quality_gate, DEFAULT_THRESHOLDS

st.set_page_config(page_title="Fingerprint Quality Gate", page_icon="🔍", layout="centered")

st.title("🔍 Contactless Fingerprint Quality Assessment")
st.caption("YellowSense Technologies — FP-03 Quality Control prototype")

# ---------------------------------------------------------------------------
# Sidebar: adjustable thresholds
# ---------------------------------------------------------------------------
st.sidebar.header("Thresholds")
st.sidebar.caption("Adjust and re-run without touching code.")

blur_min = st.sidebar.slider("Blur min (Laplacian variance)", 0.0, 200.0,
                              DEFAULT_THRESHOLDS["blur_min"], step=1.0)
brightness_low = st.sidebar.slider("Brightness low cutoff", 0.0, 150.0,
                                    DEFAULT_THRESHOLDS["brightness_low"], step=1.0)
brightness_high = st.sidebar.slider("Brightness high cutoff", 150.0, 255.0,
                                     DEFAULT_THRESHOLDS["brightness_high"], step=1.0)
glare_max = st.sidebar.slider("Glare max fraction", 0.0, 0.5,
                               DEFAULT_THRESHOLDS["glare_max_fraction"], step=0.01)
roi_min = st.sidebar.slider("ROI min fraction", 0.0, 1.0,
                             DEFAULT_THRESHOLDS["roi_min_fraction"], step=0.01)
ridge_min = st.sidebar.slider("Ridge clarity min score", 0.0, 100.0,
                               DEFAULT_THRESHOLDS["ridge_min_score"], step=1.0)
composite_pass = st.sidebar.slider("Composite pass cutoff", 0.0, 100.0,
                                    DEFAULT_THRESHOLDS["composite_pass"], step=1.0)

thresholds = {
    "blur_min": blur_min,
    "brightness_low": brightness_low,
    "brightness_high": brightness_high,
    "glare_max_fraction": glare_max,
    "roi_min_fraction": roi_min,
    "ridge_min_score": ridge_min,
    "composite_pass": composite_pass,
}

if brightness_low >= brightness_high:
    st.sidebar.error("Brightness low cutoff must be less than the high cutoff — "
                      "adjust the sliders.")
    st.stop()

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a fingerprint capture", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to run the quality gate.")
    st.stop()

# Guard against oversized uploads before decoding (edge case: huge file crashes decode)
MAX_UPLOAD_MB = 25
if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File is too large ({uploaded.size / 1e6:.1f} MB). "
              f"Please upload an image under {MAX_UPLOAD_MB} MB.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is None:
    st.error("Could not decode this file as an image. Please upload a valid JPG or PNG.")
    st.stop()

st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded capture", width=320)

# ---------------------------------------------------------------------------
# Run the gate
# ---------------------------------------------------------------------------
try:
    t0 = time.perf_counter()
    result = quality_gate(image_bgr, thresholds=thresholds)
    elapsed = (time.perf_counter() - t0) * 1000
except Exception as e:
    st.error(f"Quality gate crashed on this image: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Composite score — big number, green/red
# ---------------------------------------------------------------------------
score = result["composite_score"]
color = "#1a7f37" if score >= composite_pass else "#c0392b"
st.markdown(
    f"<h1 style='text-align:center;color:{color};font-size:64px;'>{score:.1f}</h1>"
    f"<p style='text-align:center;color:{color};font-weight:600;'>"
    f"{'PASS' if result['passed'] else 'FAIL'}</p>",
    unsafe_allow_html=True,
)

st.markdown(f"### {result['guidance']}")
st.caption(f"Processed in {elapsed:.1f} ms")

# ---------------------------------------------------------------------------
# Per-metric badges
# ---------------------------------------------------------------------------
st.subheader("Metric breakdown")


def badge(label, ok, detail):
    icon = "✅" if ok else "❌"
    st.markdown(f"**{icon} {label}** — {detail}")


col1, col2 = st.columns(2)
with col1:
    badge("Blur", not result["blur"]["is_blurry"],
          f"score={result['blur']['blur_score']:.1f} (min {blur_min:.0f})")
    badge("Brightness", not (result["brightness"]["too_dark"] or result["brightness"]["too_bright"]),
          f"mean={result['brightness']['brightness']:.1f} "
          f"(range {brightness_low:.0f}-{brightness_high:.0f})")
    badge("Glare", not result["glare"]["has_glare"],
          f"overexposed={result['glare']['glare_fraction']*100:.1f}% (max {glare_max*100:.0f}%)")
with col2:
    badge("ROI completeness", result["roi"]["roi_complete"],
          f"finger fills {result['roi']['roi_fraction']*100:.1f}% of frame "
          f"(min {roi_min*100:.0f}%)")
    badge("Ridge clarity", result["ridge"]["ridges_clear"],
          f"score={result['ridge']['ridge_score']:.1f} (min {ridge_min:.0f})")

with st.expander("Raw JSON result"):
    st.json(result)

# Assignment 4 — Fingerprint Quality Assessment & Scoring Pipeline

## Setup
```bash
pip install opencv-python-headless numpy streamlit reportlab matplotlib
```

## Files
- `quality_assessment.py` — 5 metric functions + `quality_gate()` master function
- `quality_app.py` — Streamlit UI (`streamlit run quality_app.py`)
- `test_quality.py` — batch test over `data/{good,blurry,dark,glare}`, writes `results.json` / `results_table.csv`
- `generate_sample_images.py` — creates 20 SYNTHETIC placeholder images (replace with real phone photos before submitting — see Part D of the brief)
- `report_content.py` — builds `report.pdf` from the results
- `report.pdf` — written report answering the 5 required questions
- `composite_score_by_category.png` — summary chart

## Run order
```bash
python3 generate_sample_images.py     # or drop in your own 20 real photos into data/
python3 test_quality.py               # runs the gate on all images, prints table
python3 report_content.py             # rebuilds report.pdf from latest results
streamlit run quality_app.py          # interactive UI — take your 4 screenshots here
```

## Still required from you before submission
1. Replace `data/*` with 20 REAL phone photos (5 good / 5 blurry / 5 dark / 5 glare) — Part D.
2. Take 4 screenshots of the Streamlit app (one per defect type, each showing a ❌ badge) — Part E.
3. Re-run `test_quality.py` and `report_content.py` on the real photos so report.pdf reflects real data.

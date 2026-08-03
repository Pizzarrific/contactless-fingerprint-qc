# Assignment 4 — Fingerprint Quality Assessment & Scoring Pipeline

## Setup
```bash
pip install opencv-python-headless numpy streamlit reportlab matplotlib
```

## Files
- `quality_assessment.py` — 5 metric functions + `quality_gate()` master function
- `quality_app.py` — Streamlit UI (`streamlit run quality_app.py`)
- `test_quality.py` — batch test over `data/{good,blurry,dark,glare}`, writes `results.json` / `results_table.csv`
- `report_content.py` — builds `report.pdf` from the results
- `report.pdf` — written report answering the 5 required questions(Answers are present on page 2 followed by summary)


## Run order
```bash

python3 test_quality.py               # runs the gate on all images, prints table
python3 report_content.py             # rebuilds report.pdf from latest results
streamlit run quality_app.py          # interactive UI — take your 4 screenshots here
```



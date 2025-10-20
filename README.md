# Predictive UX Feedback Analyzer — v1.0 (MVP)

**What it does:** turns mixed-method UX feedback (scores + comments) into explainable churn risk, segment hotspots, and “next bets.”
- Works locally with CSV uploads or via published Google Sheets CSV links
- Privacy-first: no data leaves your session; files only saved locally on download
- Exports: **report.html** + **risk_scores.csv**

## Run locally (Windows)
1) Unzip this folder.
2) Double-click **run_app.bat**.
3) Browser opens at `http://localhost:8501`. Click **Try demo dataset** or upload your own CSV.

## Run locally (Mac/Linux)
```bash
chmod +x run_app.sh
./run_app.sh
```

## Deploy options
- **Streamlit Community Cloud**: point to repo, main file = `app.py`.
- **Render**: use `render.yaml`.
- **Docker**:
  ```bash
  docker build -t ux-analyzer .
  docker run -p 8501:8501 ux-analyzer
  ```

## Data schema (survey_responses.csv)
```
user_id,timestamp,nps,sus,feature_used,churned,comment
```

## Guardrails
- Yellow banner when **n < 200** (directional only)
- Class imbalance notice when churn <5% or >95%
- PII scrub for emails/URLs/phones in comments
- Deterministic seed for reproducibility

## How it works (1 paragraph)
We clean the data, convert comments into numerical signals (TF‑IDF), one‑hot encode segments, and scale numeric scores. A regularized, class-weighted logistic regression ranks churn risk per response and explains *why* via coefficients. We aggregate risk by segment (with avg NPS/SUS), surface top frustration keywords, and export both a visual report and a risk_scores CSV so teams can act.

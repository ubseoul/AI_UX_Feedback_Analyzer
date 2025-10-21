import os, io, pandas as pd, streamlit as st, numpy as np, requests, traceback
from src.model_utils import (
    REQUIRED_COLS, quality_checks, fit_and_score,
    save_chart_segment, save_chart_top_feats, render_html, export_risk_scores
)

st.set_page_config(page_title="Predictive UX Feedback Analyzer", layout="wide")
st.title("Predictive UX Feedback Analyzer")
st.caption("Quant + qual feedback → risk clusters and why. All processing is local to your session; we don’t store uploads.")

# ---------- Session state helpers ----------
if "survey_df" not in st.session_state:
    st.session_state.survey_df = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

def read_csv_input(uploaded, url):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if url and url.strip():
        r = requests.get(url.strip(), timeout=10)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    return None

# ---------- Inputs ----------
with st.expander("Data inputs", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Upload CSVs")
        survey_file = st.file_uploader("survey_responses.csv (required)", type=["csv"])
        usability_file = st.file_uploader("usability_test_notes.csv (optional)", type=["csv"])
        benchmarks_file = st.file_uploader("benchmarks.csv (optional)", type=["csv"])
    with c2:
        st.subheader("Or paste Google Sheets CSV URL")
        st.caption("Publish your Sheet as CSV; paste the URL here.")
        survey_url = st.text_input("Survey CSV URL")
        load_btns = st.columns(2)
        with load_btns[0]:
            if st.button("Load from inputs"):
                try:
                    df = read_csv_input(survey_file, survey_url)
                    if df is None:
                        st.warning("No file/URL provided or load failed.")
                    else:
                        st.session_state.survey_df = df
                        st.success("Loaded into session.")
                except Exception as e:
                    st.error(f"Could not load: {e}")
        with load_btns[1]:
            if st.button("▶ Try demo dataset"):
                st.session_state.survey_df = pd.DataFrame([
                    {"user_id":"u001","timestamp":"2025-09-10","nps":9,"sus":82,"feature_used":"stories","churned":0,"comment":"Love the simplicity, but onboarding was confusing."},
                    {"user_id":"u002","timestamp":"2025-09-12","nps":4,"sus":55,"feature_used":"editor","churned":1,"comment":"Editor crashes on large files. I switched tools."},
                    {"user_id":"u003","timestamp":"2025-09-13","nps":7,"sus":68,"feature_used":"export","churned":0,"comment":"Export is slow; team loves collaboration though."},
                    {"user_id":"u004","timestamp":"2025-09-15","nps":2,"sus":40,"feature_used":"stories","churned":1,"comment":"Couldn’t find settings; navigation feels messy."},
                    {"user_id":"u005","timestamp":"2025-09-17","nps":8,"sus":75,"feature_used":"editor","churned":0,"comment":"Great docs; dark mode saved my eyes."},
                ])
                st.success("Demo dataset loaded into session.")

st.markdown("---")

# ---------- Need data? ----------
if st.session_state.survey_df is None:
    st.info("Load data first (Upload + **Load from inputs** OR click **Try demo dataset**).")
    st.stop()

survey = st.session_state.survey_df

# ---------- Quality checks & preview ----------
st.subheader("Quality checks")
issues = quality_checks(survey)
if issues:
    for i in issues:
        st.warning("• " + i)
else:
    st.success("Inputs look reasonable. Proceed.")

with st.expander("Preview your data"):
    st.dataframe(survey.head(20), use_container_width=True)

# ---------- Run analysis (form keeps state stable) ----------
with st.form("run_form", clear_on_submit=False):
    submitted = st.form_submit_button("Run analysis & generate report")

if submitted:
    try:
        with st.spinner("Training and rendering..."):
            results = fit_and_score(survey, use_cv=True)
            seg = results["seg"]
            top_feats = results["top_feats"]
            top_keywords = results["top_keywords"]
            auc = results["auc"]
            report = results["report"]
            df_scored = results["df"]

            st.subheader("Overall Model")
            st.write(f"**AUC**: {auc if not np.isnan(auc) else 'n/a (tiny sample)'}")
            st.text_area("Classification report / notes", report, height=180)

            st.subheader("Risk by segment")
            st.dataframe(seg, use_container_width=True)

            st.subheader("Top risk signals (model coefficients)")
            st.dataframe(pd.DataFrame(top_feats, columns=["feature","weight"]), use_container_width=True)

            st.subheader("Top frustration keywords (TF-IDF)")
            st.dataframe(pd.DataFrame(top_keywords, columns=["token","importance"]), use_container_width=True)

            # Charts + report
            os.makedirs("report/assets", exist_ok=True)
            seg_chart_path = "report/assets/segment_risk.png"
            feats_chart_path = "report/assets/top_features.png"
            kw_chart_path = "report/assets/top_keywords.png"

            save_chart_segment(seg, seg_chart_path)
            save_chart_top_feats(top_feats, feats_chart_path, title="Top Risk Signals", xlabel="Coefficient (higher = more churn risk)")
            save_chart_top_feats(top_keywords, kw_chart_path, title="Top Frustration Keywords", xlabel="TF-IDF Δ (churn vs non-churn)")

            out_html = "report/report.html"
            render_html(df_scored, seg, auc, report, top_feats, top_keywords, seg_chart_path, feats_chart_path, kw_chart_path, out_html)

            risk_csv = "report/risk_scores.csv"
            export_risk_scores(df_scored, risk_csv)

            with open(out_html, "rb") as f:
                st.download_button("Download report.html", data=f, file_name="report.html", mime="text/html", use_container_width=True)
            with open(risk_csv, "rb") as f:
                st.download_button("Download risk_scores.csv", data=f, file_name="risk_scores.csv", mime="text/csv", use_container_width=True)

            st.success("Done! Files saved under ./report/ as well.")
            st.session_state.last_error = None

    except Exception as e:
        st.session_state.last_error = traceback.format_exc()
        st.error("Something went wrong during analysis. See details below.")

# ---------- Show last error (if any) ----------
if st.session_state.last_error:
    with st.expander("Error details (for debugging)"):
        st.code(st.session_state.last_error, language="text")

st.markdown("---")
st.caption("Privacy: we don’t store your uploads. Data is processed in-memory; files are only written locally for your own downloads.")

# src/model_utils.py
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# Required schema (display + checks)
# -----------------------------
REQUIRED_COLS = [
    "user_id",       # unique row/user id
    "timestamp",     # ISO date or datetime
    "nps",           # 0..10
    "sus",           # 0..100
    "feature_used",  # segment/group/category
    "churned",       # 0/1 (or inferred)
    "comment",       # free text
]

# ---- Exported names (put this here) ----
__all__ = [
    "REQUIRED_COLS",
    "quality_checks",
    "clean_df",
    "fit_and_score",
    "save_chart",
    "save_chart_segment",
    "save_chart_top_features",
    "save_chart_top_feats",   # alias for backward-compat with app.py
    "render_html",
]

# Optional explicit export list
__all__ = [
    "REQUIRED_COLS",
    "quality_checks",
    "clean_df",
    "fit_and_score",
    "save_chart",
    "save_chart_segment",
    "save_chart_top_features",
    "render_html",
]

# -----------------------------
# Light quality checks (messages for UI)
# -----------------------------
def quality_checks(df: pd.DataFrame) -> List[str]:
    msgs = []
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        msgs.append(f"• Missing required columns: {', '.join(missing)}")

    n = len(df)
    if n < 200:
        msgs.append(f"• Small sample (n={n}). Treat results as directional until n≥200.")
    if "comment" in df.columns and len(df) > 0:
        if df["comment"].astype(str).str.len().lt(5).mean() > 0.5:
            msgs.append("• Many very short comments; text signals may be weak.")
    if "churned" in df.columns and len(df) > 0:
        pos = pd.to_numeric(df["churned"], errors="coerce").fillna(0).astype(int).sum()
        if pos == 0 or pos == n:
            msgs.append("• churned has only one class; AUC cannot be computed.")
    return msgs

# -----------------------------
# Cleaning & coercion with safe defaults
# -----------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming dataframe:
    - create missing columns with sensible defaults
    - coerce types/ranges
    - avoid calling .fillna on plain strings
    """
    df = df.copy()

    # feature_used
    if "feature_used" not in df.columns:
        df["feature_used"] = pd.Series(["all"] * len(df), index=df.index)
    else:
        df["feature_used"] = (
            df["feature_used"].astype(str)
            .replace({"": "all", "nan": "all", "None": "all"})
            .fillna("all")
        )

    # comment
    if "comment" not in df.columns:
        df["comment"] = pd.Series([""] * len(df), index=df.index)
    else:
        df["comment"] = df["comment"].astype(str).fillna("")

    # nps (0..10)
    if "nps" not in df.columns:
        df["nps"] = 5
    df["nps"] = pd.to_numeric(df["nps"], errors="coerce").fillna(5).clip(0, 10).astype(int)

    # sus (0..100)
    if "sus" not in df.columns:
        df["sus"] = 70
    df["sus"] = pd.to_numeric(df["sus"], errors="coerce").fillna(70).clip(0, 100).astype(int)

    # churned (0/1) with inference if missing
    if "churned" in df.columns:
        raw = df["churned"].astype(str).str.strip().str.lower()
        df["churned"] = np.where(
            raw.isin(["1", "true", "yes", "y"]),
            1,
            np.where(raw.isin(["0", "false", "no", "n"]), 0, np.nan),
        )
        df["churned"] = pd.to_numeric(df["churned"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        if "sentiment" in df.columns:
            df["churned"] = df["sentiment"].astype(str).str.lower().isin(
                ["negative", "neg", "bad", "0"]
            ).astype(int)
        elif "rating" in df.columns:
            df["churned"] = (pd.to_numeric(df["rating"], errors="coerce") <= 2).astype(int)
        else:
            df["churned"] = 0

    # user_id
    if "user_id" not in df.columns:
        df["user_id"] = [f"id_{i:06d}" for i in range(len(df))]

    # timestamp
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.utcnow().date().isoformat()

    return df

# -----------------------------
# Plot saving helper
# -----------------------------
def save_chart(fig: plt.Figure, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

# NEW: explicit chart helpers to match app.py imports
def save_chart_segment(seg_series: pd.Series, path: str) -> str:
    """Create and save the 'risk by segment' horizontal bar chart."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(7, max(2.5, 0.35 * (len(seg_series) + 2))))
    seg_series.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("Avg predicted risk (0–1)")
    plt.ylabel("Segment (feature_used)")
    plt.title("Risk by segment")
    return save_chart(fig, path)

def save_chart_top_features(top_features: List[Tuple[str, float]], path: str) -> str:
    """Create and save the 'top risk signals' horizontal bar chart."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not top_features:
        top_features = [("No strong positive signals", 0.0)]
    names, weights = zip(*top_features)
    y_pos = np.arange(len(names))
    fig = plt.figure(figsize=(7, max(2.5, 0.35 * (len(names) + 2))))
    plt.barh(y_pos, weights)
    plt.yticks(y_pos, names)
    plt.xlabel("Model weight (higher → more risk)")
    plt.title("Top risk signals (positive coefficients)")
    plt.gca().invert_yaxis()
    return save_chart(fig, path)

def save_chart_top_feats(top_features, path: str) -> str:
    return save_chart_top_features(top_features, path)

# Backward-compat alias to match app.py import
def save_chart_top_feats(top_features, path: str) -> str:
    return save_chart_top_features(top_features, path)

# -----------------------------
# Core fit, score, features
# -----------------------------
def _build_pipeline():
    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2), "comment"),
            ("seg", OneHotEncoder(handle_unknown="ignore"), ["feature_used"]),
            ("num", "passthrough", ["nps", "sus"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=42,
    )
    return preprocess, model

def _get_feature_names(preprocess: ColumnTransformer) -> List[str]:
    try:
        return preprocess.get_feature_names_out().tolist()
    except Exception:
        return [f"f{i}" for i in range(1, 5001)] + ["seg_*", "nps", "sus"]

def _top_positive_features(coef: np.ndarray, feat_names: List[str], k: int = 15) -> List[Tuple[str, float]]:
    weights = coef.flatten()
    order = np.argsort(weights)[::-1]
    sel = [(feat_names[i] if i < len(feat_names) else f"f{i}", float(weights[i])) for i in order[:k]]
    return [(n, w) for (n, w) in sel if w > 0]

def fit_and_score(df: pd.DataFrame, use_cv: bool = True) -> Dict[str, Any]:
    df = clean_df(df)

    X_cols = ["comment", "feature_used", "nps", "sus"]
    y = df["churned"].astype(int).values

    preprocess, model = _build_pipeline()
    X_proc = preprocess.fit_transform(df[X_cols])

    # AUC via CV or in-sample fallback
    auc = None
    report_txt = "CV used — classification report not computed per fold."
    if len(np.unique(y)) >= 2:
        if use_cv and len(df) >= 40:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                aucs = cross_val_score(model, X_proc, y, cv=cv, scoring="roc_auc")
                auc = float(np.mean(aucs))
            except Exception:
                auc = None
        else:
            model.fit(X_proc, y)
            p = model.predict_proba(X_proc)[:, 1]
            try:
                auc = float(roc_auc_score(y, p))
            except Exception:
                auc = None
            try:
                y_pred = (p >= 0.5).astype(int)
                report_txt = classification_report(y, y_pred, digits=3)
            except Exception:
                pass

    if not hasattr(model, "classes_"):
        model.fit(X_proc, y)

    probs = model.predict_proba(X_proc)[:, 1]
    df_scored = df.copy()
    df_scored["predicted_risk"] = probs

    # Segment series for chart
    seg_series = (
        df_scored.groupby("feature_used", dropna=False)["predicted_risk"]
        .mean()
        .sort_values(ascending=True)
    )

    # Top features
    feat_names = _get_feature_names(preprocess)
    try:
        model.fit(X_proc, y)
        coefs = model.coef_
    except Exception:
        coefs = np.zeros((1, len(feat_names)))
    top_feats = _top_positive_features(coefs, feat_names, k=15)

    # Save charts via the new explicit helpers
    seg_chart_path = save_chart_segment(seg_series, "charts/segment_risk.png")
    top_feats_chart_path = save_chart_top_features(top_feats, "charts/top_features.png")

    return {
        "auc": auc,
        "report": report_txt,
        "df_scored": df_scored,
        "seg_chart_path": seg_chart_path,
        "top_feats_chart_path": top_feats_chart_path,
        "top_features": top_feats,
        "seg_series": seg_series,
    }

# -----------------------------
# Simple HTML report writer
# -----------------------------
def render_html(
    df_scored: pd.DataFrame,
    seg_series: pd.Series,
    auc: Any,
    report: str,
    top_features: List[Tuple[str, float]],
    seg_chart_path: str,
    top_feats_chart_path: str,
    out_path: str = "report/report.html",
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    seg_img = os.path.relpath(seg_chart_path, os.path.dirname(out_path)).replace("\\", "/")
    feat_img = os.path.relpath(top_feats_chart_path, os.path.dirname(out_path)).replace("\\", "/")

    # Top quotes for high-risk users
    sample_quotes = []
    high = df_scored.sort_values("predicted_risk", ascending=False).head(8)
    for _, r in high.iterrows():
        uid = str(r.get("user_id", "id"))
        seg = str(r.get("feature_used", "segment"))
        risk = float(r.get("predicted_risk", 0.0))
        text = str(r.get("comment", "")).strip()
        if len(text) > 140:
            text = text[:137] + "..."
        sample_quotes.append(f'{uid} ({seg}) — risk {risk:.2f}: “{text}”')

    tf_block = "\n".join([f"- {n}: {w:.3f}" for n, w in top_features[:12]])
    seg_tbl = "\n".join([f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" for k, v in seg_series.items()])

    auc_text = "N/A"
    if isinstance(auc, (int, float)) and not pd.isna(auc):
        auc_text = f"{auc:.3f}"

    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Predictive UX Feedback Report</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #111; }}
 h1,h2 {{ margin: 0 0 8px 0; }}
 .muted {{ color:#666; font-size: 14px; }}
 .card {{ border:1px solid #eee; border-radius:12px; padding:16px; margin:16px 0; }}
 .kpi {{ font-weight:600; font-size: 18px; }}
 img {{ max-width: 100%; height: auto; border:1px solid #eee; border-radius:8px; }}
 table {{ border-collapse: collapse; width:100%; }}
 th,td {{ border:1px solid #eee; padding:8px; text-align:left; }}
 pre {{ white-space: pre-wrap; }}
</style>
</head>
<body>
  <h1>Predictive UX Feedback Report</h1>
  <div class="muted">Generated {now_utc} • MVP • Directional Insights</div>

  <div class="card">
    <h2>Overall Model</h2>
    <div class="kpi">AUC: {auc_text}</div>
    <p><strong>Model:</strong> Logistic Regression on TF-IDF(text) + NPS/SUS + segment.</p>
    <pre class="muted">{report}</pre>
  </div>

  <div class="card">
    <h2>Dataset & Guardrails</h2>
    <p>Rows: {len(df_scored)} | Churn=1: {int(df_scored['churned'].sum())} | Segments: {df_scored['feature_used'].nunique()}</p>
    <ul class="muted">
      <li>Directional: great for prioritizing next steps; not precise forecasting on small samples.</li>
      <li>PII stripped; quotes are representative snippets.</li>
      <li>Repeatable: deterministic pipeline + archived data snapshot.</li>
    </ul>
  </div>

  <div class="card">
    <h2>Risk by segment</h2>
    <p>Avg predicted risk (0–1) per segment; higher = riskier.</p>
    <img src="{seg_img}" alt="Segment risk chart"/>
    <table><thead><tr><th>Segment</th><th>Avg risk</th></tr></thead><tbody>
      {seg_tbl}
    </tbody></table>
  </div>

  <div class="card">
    <h2>Top risk signals</h2>
    <p>Words/flags most associated with higher churn risk (model weights).</p>
    <img src="{feat_img}" alt="Top features chart"/>
    <pre>{tf_block}</pre>
  </div>

  <div class="card">
    <h2>Representative Quotes (High-Risk Users)</h2>
    <ul>
      {"".join([f"<li>{q}</li>" for q in sample_quotes])}
    </ul>
  </div>

  <div class="muted">© MVP — For portfolio/demo. Treat as directional until sample ≥ 500 rows.</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from string import Template
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

RANDOM_STATE = 42
REQUIRED_COLS = ["user_id","timestamp","nps","sus","feature_used","churned","comment"]

def scrub_text(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=str)
    s = s.fillna("").astype(str)
    s = s.str.replace(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", regex=True)
    s = s.str.replace(r"https?://\S+|www\.\S+", "[url]", regex=True)
    s = s.str.replace(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[phone]", regex=True)
    return s

def quality_checks(df: pd.DataFrame):
    issues = []
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {', '.join(missing_cols)}")
        return issues
    miss_pct = df[REQUIRED_COLS].isna().mean().to_dict()
    if any(v > 0.2 for v in miss_pct.values()):
        issues.append("High missingness (>20%) in some required columns.")
    try:
        churn_rate = float(df["churned"].fillna(0).astype(int).mean())
        if churn_rate < 0.05 or churn_rate > 0.95:
            issues.append(f"Imbalanced classes (churn rate {churn_rate:.2%}).")
    except Exception:
        issues.append("Could not compute churn rate.")
    if len(df) < 200:
        issues.append(f"Small sample (n={len(df)}). Treat results as directional until n≥200.")
    return issues

def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["feature_used"]),
        ("num", StandardScaler(), ["nps","sus"]),
        ("txt", TfidfVectorizer(max_features=1000, ngram_range=(1,2)), "comment"),
    ], verbose_feature_names_out=False)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([("union", pre), ("clf", clf)])
    return pipe

def clean_df(df: pd.DataFrame):
    df = df.copy()
    df["comment"] = scrub_text(df.get("comment"))
    df["feature_used"] = df.get("feature_used", "unknown").fillna("unknown")
    df["nps"] = pd.to_numeric(df["nps"], errors="coerce")
    df["sus"] = pd.to_numeric(df["sus"], errors="coerce")
    df["nps"] = df["nps"].fillna(df["nps"].median())
    df["sus"] = df["sus"].fillna(df["sus"].median())
    df["churned"] = df["churned"].fillna(0).astype(int)
    return df

def fit_and_score(df: pd.DataFrame, use_cv: bool = True):
    df = clean_df(df)
    X = df[["nps","sus","feature_used","comment"]]
    y = df["churned"]
    pipe = build_pipeline()

    if use_cv and y.nunique() > 1 and len(df) >= 50:
        cv = StratifiedKFold(n_splits=min(5, max(2, int(len(df) // 50))), shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        pipe.fit(X, y)
        auc = float(np.nanmean(cv_scores))
        auc_detail = cv_scores.tolist()
        report = "Cross-validation used — classification report not computed per fold."
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y if y.nunique()>1 else None
        )
        pipe.fit(Xtr, ytr)
        preds = pipe.predict_proba(Xte)[:,1]
        try:
            auc = float(roc_auc_score(yte, preds))
        except Exception:
            auc = float("nan")
        report = classification_report(yte, (preds>0.5).astype(int), zero_division=0)
        auc_detail = None

    df["_risk"] = pipe.predict_proba(X)[:,1]

    seg = df.groupby("feature_used").agg(
        n=("user_id","count"),
        avg_risk=("_risk","mean"),
        avg_nps=("nps","mean"),
        avg_sus=("sus","mean")
    ).reset_index().sort_values("avg_risk", ascending=False)

    clf = pipe.named_steps["clf"]
    feature_names = list(pipe.named_steps["union"].get_feature_names_out())
    coefs = clf.coef_.ravel()
    order = np.argsort(coefs)[::-1]
    top_idx = order[:10]
    top_feats = [(feature_names[i], float(coefs[i])) for i in top_idx]

    vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    tf = vec.fit_transform(df["comment"])
    vocab = np.array(vec.get_feature_names_out())
    mask = (df["churned"]==1).values
    mu_pos = tf[mask].mean(axis=0).A1 if mask.any() else np.zeros(tf.shape[1])
    mu_neg = tf[~mask].mean(axis=0).A1 if (~mask).any() else np.zeros(tf.shape[1])
    diff = mu_pos - mu_neg
    kw_idx = np.argsort(diff)[::-1][:10]
    top_keywords = [(vocab[i], float(diff[i])) for i in kw_idx]

    return {
        "df": df,
        "seg": seg,
        "auc": auc,
        "auc_detail": auc_detail,
        "report": report,
        "top_feats": top_feats,
        "top_keywords": top_keywords,
        "pipe": pipe,
    }

def save_chart_segment(seg: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    if len(seg)==0:
        plt.title("No segments available")
        plt.xlabel("Feature"); plt.ylabel("Avg predicted risk")
    else:
        plt.bar(seg["feature_used"], seg["avg_risk"])
        plt.title("Predicted Churn Risk by Feature")
        plt.xlabel("Feature"); plt.ylabel("Avg predicted risk")
        plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def save_chart_top_feats(top_list, path: str, title="Top Risk Signals", xlabel="Coefficient (higher = more churn risk)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = [t[0] for t in top_list]
    values = [t[1] for t in top_list]
    plt.figure()
    if len(labels)==0:
        plt.title("No signals available")
    else:
        plt.barh(range(len(labels)), values); plt.yticks(range(len(labels)), labels)
        plt.title(title); plt.xlabel(xlabel); plt.gca().invert_yaxis()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def render_html(df, seg, auc, report, top_feats, top_keywords, seg_chart_path, feats_chart_path, kw_chart_path, out_path):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    hi = df.sort_values("_risk", ascending=False).head(8)[["user_id","feature_used","_risk","comment"]]
    quotes_html = "".join([
        f"<li><strong>{r.user_id}</strong> ({r.feature_used}) — risk {r._risk:.2f}: “{r.comment}”</li>"
        for _, r in hi.iterrows()
    ])

    actions = []
    if len(seg):
        top_seg = seg.iloc[0]
        actions.append({
            "title": f"Improve {top_seg['feature_used']} onboarding & discoverability",
            "why": f"{int(top_seg['n'])} users; highest predicted churn risk segment ({top_seg['avg_risk']:.2f}).",
            "impact": "High (directional)"
        })
    actions.append({
        "title": "Address recurring text themes",
        "why": "Top keywords from comments highlight core frustrations to target.",
        "impact": "Medium–High (directional)"
    })
    actions.append({
        "title": "Close benchmark gaps (SUS/NPS)",
        "why": "Average NPS/SUS by feature shows where polish will have outsized impact.",
        "impact": "Medium (directional)"
    })
    actions_html = "".join([
        f"<li><strong>{a['title']}</strong> — {a['why']} <em>(Predicted impact: {a['impact']})</em></li>"
        for a in actions
    ])

    seg_rel   = os.path.relpath(seg_chart_path,  os.path.dirname(out_path)).replace('\\','/')
    feats_rel = os.path.relpath(feats_chart_path, os.path.dirname(out_path)).replace('\\','/')
    kw_rel    = os.path.relpath(kw_chart_path,   os.path.dirname(out_path)).replace('\\','/')
    auc_text = f"{auc:.3f}" if not np.isnan(auc) else "n/a (tiny sample)"

    banner = ""
    if len(df) < 200:
        banner = '<div style="background:#fff7cc;border:1px solid #eedc82;padding:8px;border-radius:8px;margin:10px 0;">Directional insights only — sample size below 200.</div>'

    tpl = Template(r"""<!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Predictive UX Feedback Report</title>
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;line-height:1.5}
  .wrap{max-width:1100px;margin:0 auto}
  h1{font-size:28px;margin-bottom:4px} h2{margin-top:28px}
  .badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;border:1px solid #ddd}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .card{border:1px solid #eee;border-radius:12px;padding:16px}
  table{width:100%;border-collapse:collapse}
  th,td{border:1px solid #eee;padding:8px;text-align:left}
  .small{color:#666;font-size:12px}
  img{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px}
  ol li{margin-bottom:8px}
</style>
</head><body><div class="wrap">
<h1>Predictive UX Feedback Report</h1>
<div class="small">Generated $now</div>
<p><span class="badge">MVP • Directional Insights</span></p>
$banner
<div class="grid">
  <div class="card">
    <h2>Overall Model</h2>
    <p><strong>AUC</strong>: $auc_text<br/>
    <span class="small">Classification report/notes:</span></p>
    <pre class="small">$report</pre>
    <p class="small">Model: Regularized logistic regression on TF-IDF(text) + SUS/NPS + feature segments.</p>
  </div>
  <div class="card">
    <h2>Dataset &amp; Guardrails</h2>
    <p><strong>Rows</strong>: $rows | <strong>Churn=1</strong>: $churn_ones | <strong>Segments</strong>: $segments</p>
    <ul class="small">
      <li>Privacy: emails/URLs/phones masked in comments; nothing stored unless you download.</li>
      <li>Directional until larger sample or stratified experiments are run.</li>
      <li>Deterministic seed for reproducibility ($seed).</li>
    </ul>
  </div>
</div>

<h2>Next 3 Bets</h2>
<ol>$actions_html</ol>

<h2>Risk by Segment</h2>
<p class="small">Avg predicted risk (0–1) per segment; higher = riskier. Includes avg NPS/SUS per feature.</p>
<img src="$seg_rel" alt="Segment risk chart"/>

<h2>Top Risk Signals (Model)</h2>
<p class="small">Words/flags most associated with higher churn risk (model weights).</p>
<img src="$feats_rel" alt="Top features chart"/>

<h2>Top Frustration Keywords (TF‑IDF)</h2>
<p class="small">Frequent, distinctive tokens among churn‑labeled comments.</p>
<img src="$kw_rel" alt="Top keywords chart"/>

<h2>Representative Quotes (High-Risk Users)</h2>
<ul>$quotes_html</ul>

<p class="small">© MVP — For portfolio/demo. Treat as directional until sample ≥ 500 rows.</p>
</div></body></html>""")

    html = tpl.safe_substitute(
        now=now,
        auc_text=auc_text,
        report=report,
        rows=len(df),
        churn_ones=int(df["churned"].sum()),
        segments=int(df["feature_used"].nunique()),
        actions_html=actions_html,
        seg_rel=seg_rel,
        feats_rel=feats_rel,
        kw_rel=kw_rel,
        quotes_html=quotes_html,
        banner=banner,
        seed=RANDOM_STATE
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

def export_risk_scores(df: pd.DataFrame, path: str):
    out = df[["user_id","feature_used","_risk","comment"]].copy()
    out = out.sort_values("_risk", ascending=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return path

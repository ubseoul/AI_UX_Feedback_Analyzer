# src/model_utils.py
import numpy as np
import pandas as pd
from datetime import datetime

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the incoming dataframe safe for modeling:
    - create missing columns with sensible defaults
    - coerce types/ranges
    - avoid calling .fillna on plain strings
    """
    df = df.copy()

    # ---------- feature_used ----------
    if "feature_used" not in df.columns:
        # create a Series of "all" with correct length
        df["feature_used"] = pd.Series(["all"] * len(df), index=df.index)
    else:
        df["feature_used"] = (
            df["feature_used"].astype(str)
            .replace({"": "all"})
            .fillna("all")
        )

    # ---------- comment ----------
    if "comment" not in df.columns:
        df["comment"] = pd.Series([""] * len(df), index=df.index)
    else:
        df["comment"] = df["comment"].astype(str).fillna("")

    # ---------- nps (0..10) ----------
    if "nps" not in df.columns:
        df["nps"] = 5
    df["nps"] = pd.to_numeric(df["nps"], errors="coerce").fillna(5).clip(0, 10).astype(int)

    # ---------- sus (0..100) ----------
    if "sus" not in df.columns:
        df["sus"] = 70
    df["sus"] = pd.to_numeric(df["sus"], errors="coerce").fillna(70).clip(0, 100).astype(int)

    # ---------- churned (0/1) ----------
    if "churned" in df.columns:
        # accept 0/1, yes/no, true/false
        raw = df["churned"].astype(str).str.strip().str.lower()
        df["churned"] = np.where(
            raw.isin(["1", "true", "yes", "y"]),
            1,
            np.where(raw.isin(["0", "false", "no", "n"]), 0, np.nan),
        )
        df["churned"] = pd.to_numeric(df["churned"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        # try to infer from common fields if present
        if "sentiment" in df.columns:
            df["churned"] = df["sentiment"].astype(str).str.lower().isin(
                ["negative", "neg", "bad", "0"]
            ).astype(int)
        elif "rating" in df.columns:
            df["churned"] = (pd.to_numeric(df["rating"], errors="coerce") <= 2).astype(int)
        else:
            df["churned"] = 0

    # ---------- user_id ----------
    if "user_id" not in df.columns:
        df["user_id"] = [f"id_{i:06d}" for i in range(len(df))]

    # ---------- timestamp ----------
    if "timestamp" not in df.columns:
        today = datetime.utcnow().date().isoformat()
        df["timestamp"] = pd.Series([today] * len(df), index=df.index)

    return df

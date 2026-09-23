"""Chronological forward-chaining evaluation (paper Section 6.3, Appendix E).

At each origin t, XGBoost is trained on articles with TA <= t and tested on
articles with TA in (t, t + W]; the origin advances by W, so each article is
predicted once by a model trained on strictly earlier articles. Origins satisfy
t <= TA_max - W. Warm-up before the first origin: 2 weeks (main corpora),
4 weeks (extended corpus). Decision threshold 0.5; XGBoost hyperparameters as
in Appendix B.4; scale_pos_weight and median imputation computed on the
training set. Pooled F1 is reported with a 95% percentile bootstrap interval.

Outputs forward_chaining_results.xlsx: main_table, stability_W, per-window
tables (win_*), warm-up x window grid (grid_*), subclass rates (sub_*).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_DIR = "DATA/private/"
CORPORA = [
    # display name, short tag (unique, used in sheet names), file, test sheet, warm-up (weeks)
    ("HydroNewsFr", "hydro",
     DATA_DIR + "requested_two_diagnostics_TA_k4.xlsx",
     "binary_50chrono_test", 2),
    ("ClimateNewsFr", "climat",
     DATA_DIR + "requested_two_diagnostics_TA_k4_climat.xlsx",   # contents k=6
     "binary_50chrono_test", 2),
    ("HydroNewsFr Extended (App. D)", "ext",
     DATA_DIR + "requested_two_diagnostics_TA_k4_all.xlsx",
     "binary_50chrono_preds", 4),
]
MAIN_WINDOWS = [7, 14]
STAB_WINDOWS = [5, 7, 14, 21]
GRID_WINDOWS = [7, 14, 21, 28]
EXT_P14_WARMUP = 8
LOW_N = 20  # configurations with fewer pooled positives are flagged

META = ["media_url", "y", "true_subclass", "n_outlier_votes", "n_TOA_votes",
        "n_TODlate_votes", "n_O_votes", "TA", "n_models_present",
        "y_score", "y_pred"]
XGB_KW = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8,
              objective="binary:logistic", eval_metric="logloss",
              random_state=42, verbosity=0)
N_BOOT = 2000


def load(path, test_sheet):
    tr = pd.read_excel(path, sheet_name="binary_50chrono_train")
    te = pd.read_excel(path, sheet_name=test_sheet).drop(
        columns=["y_score", "y_pred"], errors="ignore")
    df = pd.concat([tr, te], ignore_index=True)
    assert not df.media_url.duplicated().any(), "duplicate articles across sheets"
    df["TA"] = pd.to_datetime(df["TA"])
    df = df.sort_values("TA").reset_index(drop=True)
    feats = [c for c in df.columns if c not in META]
    return df, feats


def rolling(df, feats, warmup_weeks, window_days):
    """One forward-chaining run. Returns (pooled predictions, window table)."""
    end_limit = df.TA.max() - pd.Timedelta(days=window_days)
    origins = pd.date_range(df.TA.min() + pd.Timedelta(weeks=warmup_weeks),
                            end_limit, freq=f"{window_days}D")
    pooled, rows = [], []
    for t in origins:
        m_tr = df.TA <= t
        m_te = (df.TA > t) & (df.TA <= t + pd.Timedelta(days=window_days))
        ytr = df.y[m_tr]
        if m_te.sum() == 0 or ytr.nunique() < 2:
            rows.append((t.date(), int(m_tr.sum()), int(ytr.sum()),
                         int(m_te.sum()), int(df.y[m_te].sum()), "SKIPPED"))
            continue
        Xtr = df.loc[m_tr, feats]
        med = Xtr.median()                       # train-only imputation
        clf = XGBClassifier(
            scale_pos_weight=max((ytr == 0).sum(), 1) / max(ytr.sum(), 1),
            **XGB_KW)
        clf.fit(Xtr.fillna(med), ytr)
        p = clf.predict_proba(df.loc[m_te, feats].fillna(med))[:, 1]
        yw = df.y[m_te].values
        pooled.append(pd.DataFrame({"y": yw, "p": p, "origin": t,
                                    "sub": df.true_subclass[m_te].values}))
        yh = (p >= 0.5).astype(int)
        wf1 = f1_score(yw, yh, zero_division=0) if yw.sum() else np.nan
        rows.append((t.date(), int(m_tr.sum()), int(ytr.sum()),
                     int(m_te.sum()), int(yw.sum()),
                     f"{wf1:.3f}" if not np.isnan(wf1) else "no-pos"))
    table = pd.DataFrame(rows, columns=["origin", "n_train", "pos_train",
                                        "n_test", "pos_test", "win_F1"])
    P = pd.concat(pooled, ignore_index=True) if pooled else None
    return P, table


def pooled_f1_ci(P, seed=0):
    """Pooled F1 with percentile bootstrap interval."""
    yh = (P.p >= 0.5).astype(int)
    f1 = f1_score(P.y, yh)
    rng = np.random.default_rng(seed)
    yv, pv, n = P.y.values, P.p.values, len(P)
    f1s = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if yv[idx].sum() == 0:
            continue
        f1s.append(f1_score(yv[idx], (pv[idx] >= 0.5).astype(int),
                            zero_division=0))
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return f1, lo, hi


def cell(P, seed):
    f1, lo, hi = pooled_f1_ci(P, seed)
    b = P.y.mean()
    return dict(F1=f1, lo=lo, hi=hi,
                n=len(P), pos=int(P.y.sum()), neg=int((P.y == 0).sum()),
                Pr=precision_score(P.y, (P.p >= 0.5).astype(int),
                                   zero_division=0),
                Rc=recall_score(P.y, (P.p >= 0.5).astype(int)),
                baseline=2 * b / (1 + b))


def fmt(c):
    return f"{c['F1']:.3f} [{c['lo']:.2f}, {c['hi']:.2f}]"


xlsx = pd.ExcelWriter("RESULTS/forward_chaining_results.xlsx", engine="openpyxl")
main_rows, stab_rows = [], []
seed = 0

for name, tag, path, sheet, wu in CORPORA:
    df, feats = load(path, sheet)
    print("=" * 84)
    print(f"{name} | {df.TA.min().date()} -> {df.TA.max().date()} | "
          f"n={len(df)} pos={int(df.y.sum())} features={len(feats)} "
          f"warm-up={wu}w")

    # main table
    cells = {}
    for wd in MAIN_WINDOWS:
        seed += 1
        P, table = rolling(df, feats, wu, wd)
        cells[wd] = cell(P, seed)
        table.to_excel(xlsx, sheet_name=f"win_{tag}_{wd}d",
                       index=False)
        if wd == 7:
            rates = (P.groupby("sub")
                      .agg(n=("y", "size"),
                           pred_pos_rate=("p",
                                          lambda s: float((s >= 0.5).mean())))
                      .round(3))
            rates.to_excel(xlsx, sheet_name=f"sub_{tag}")
    c7, c14 = cells[7], cells[14]

    foot = ""
    if "Extended" in name:
        seed += 1
        P8, _ = rolling(df, feats, EXT_P14_WARMUP, 14)
        c8 = cell(P8, seed)
        foot = (f"window-fit rule pools {c14['pos']}/{c14['neg']} at W=14d; "
                f"8-week warm-up: F1 = {fmt(c8)}")

    main_rows.append(dict(Corpus=name, Pos=c7["pos"], Neg=c7["neg"],
                          F1_W7=fmt(c7), F1_W14=fmt(c14),
                          Baseline_F1=round(c7["baseline"], 3),
                          footnote=foot))

    # F1 across window lengths at fixed warm-up
    for wd in STAB_WINDOWS:
        seed += 1
        P, _ = rolling(df, feats, wu, wd)
        if P is None or P.y.sum() == 0:
            continue
        c = cell(P, seed)
        stab_rows.append(dict(Corpus=name, W=f"{wd}d", warmup=f"{wu}w",
                              F1=round(c["F1"], 3), lo=round(c["lo"], 3),
                              hi=round(c["hi"], 3), pos=c["pos"],
                              low_n=c["pos"] < LOW_N))

    # warm-up x window grid
    warmups = [2, 4, 6, 8, 10] if "Extended" in name else [1, 2, 3, 4]
    grid = []
    for gwu in warmups:
        for wd in GRID_WINDOWS:
            P, _ = rolling(df, feats, gwu, wd)
            if P is None or P.y.sum() == 0:
                continue
            yh = (P.p >= 0.5).astype(int)
            grid.append(dict(warmup=f"{gwu}w", window=f"{wd}d",
                             n=len(P), pos=int(P.y.sum()),
                             F1=round(f1_score(P.y, yh), 3),
                             Pr=round(precision_score(P.y, yh,
                                                      zero_division=0), 3),
                             Rc=round(recall_score(P.y, yh), 3),
                             low_n=int(P.y.sum()) < LOW_N,
                             single_window=P.origin.nunique() == 1))
    g = pd.DataFrame(grid)
    g.to_excel(xlsx, sheet_name=f"grid_{tag}", index=False)
    ok = g[~g.low_n]
    print(f"  grid: F1 median {g.F1.median():.3f}, "
          f"range {g.F1.min():.3f}-{g.F1.max():.3f} over {len(g)} configs "
          f"({(~g.low_n).sum()} with >= {LOW_N} positives: "
          f"range {ok.F1.min():.3f}-{ok.F1.max():.3f})" if len(ok)
          else f"  grid: all configs below {LOW_N} positives")

pd.DataFrame(main_rows).to_excel(xlsx, sheet_name="main_table", index=False)
pd.DataFrame(stab_rows).to_excel(xlsx, sheet_name="stability_W", index=False)
xlsx.close()

print()
print(pd.DataFrame(main_rows).drop(columns="footnote").to_string(index=False))
print()
print(pd.DataFrame(stab_rows).to_string(index=False))
print("wrote RESULTS/forward_chaining_results.xlsx")

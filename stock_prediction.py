"""
Technical Indicators Analysis in Stock Market using Machine Learning
Author: Vedant Deshmukh

Predicts next-day price direction (UP/DOWN) for high-liquidity stocks
using technical indicators. Frames trading as binary classification.

Stocks: Choose from 10 predefined high-liquidity stocks
Models: XGBoost, AdaBoost, CatBoost, RandomForest, LogisticRegression
Evaluation: AUC-ROC, Cumulative Returns vs Buy-and-Hold

Key design decisions:
- Chronological train/test split (not random) to prevent data leakage
- AUC-ROC metric chosen over accuracy (handles class imbalance)
- Confidence-threshold three-zone strategy: Buy >= 0.55, Short <= 0.45, Hold otherwise
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import yfinance as yf
import ta

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_END      = 2010
TEST_START     = 2011
DATA_START     = "2000-01-01"
DATA_END       = "2024-09-26"
BUY_THRESHOLD  = 0.55
SELL_THRESHOLD = 0.45

# ── Available stocks ──────────────────────────────────────────────────────────

AVAILABLE_STOCKS = {
    "1":  ("MSFT",  "Microsoft"),
    "2":  ("AAPL",  "Apple"),
    "3":  ("GOOGL", "Alphabet (Google)"),
    "4":  ("AMZN",  "Amazon"),
    "5":  ("TSLA",  "Tesla"),
    "6":  ("NVDA",  "NVIDIA"),
    "7":  ("META",  "Meta (Facebook)"),
    "8":  ("JPM",   "JPMorgan Chase"),
    "9":  ("V",     "Visa"),
    "10": ("NFLX",  "Netflix"),
}


# ── Stock selection ───────────────────────────────────────────────────────────

def select_stocks() -> list:
    print("\n" + "=" * 60)
    print("  STOCK SELECTION")
    print("=" * 60)
    print("  Available stocks:\n")
    for key, (ticker, name) in AVAILABLE_STOCKS.items():
        print(f"  [{key:>2}]  {ticker:<6}  {name}")

    print("\n  Options:")
    print("  - Enter numbers separated by commas  e.g. 1,2,6")
    print("  - Enter 'all' to run all 10 stocks")
    print("  - Press Enter to run default (MSFT, AAPL)\n")

    choice = input("  Your choice: ").strip().lower()

    if choice == "":
        selected = ["MSFT", "AAPL"]
        print(f"\n  Running default: {selected}")
    elif choice == "all":
        selected = [v[0] for v in AVAILABLE_STOCKS.values()]
        print(f"\n  Running all 10 stocks")
    else:
        keys = [k.strip() for k in choice.split(",")]
        selected = []
        for k in keys:
            if k in AVAILABLE_STOCKS:
                selected.append(AVAILABLE_STOCKS[k][0])
            else:
                print(f"  Warning: '{k}' is not valid — skipped")
        if not selected:
            print("  No valid selection — running default (MSFT, AAPL)")
            selected = ["MSFT", "AAPL"]

    print(f"\n  Selected: {selected}\n")
    return selected


# ── Stage 1: Load and split ───────────────────────────────────────────────────

def load_data(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=DATA_START, end=DATA_END, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    print(f"[Data] {ticker}: {raw.shape[0]} rows, "
          f"{raw.index[0].date()} → {raw.index[-1].date()}")
    return raw


def split_data(df: pd.DataFrame):
    train = df[df.index.year <= TRAIN_END].copy()
    test  = df[df.index.year >= TEST_START].copy()
    print(f"[Split] Train: {train.shape[0]} rows | Test: {test.shape[0]} rows")
    return train, test


# ── Stage 2: Target ───────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Close_Shift"] = df["Adj Close"].shift(1)
    df["Return"]      = ((df["Adj Close"] / df["Close_Shift"]) - 1) * 100
    df["target"]      = np.where(df["Return"].shift(-1) > 0, 1, 0)
    df.dropna(inplace=True)
    return df


# ── Stage 3: Feature engineering ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for window in [5, 10, 15, 20, 30, 50, 80, 100, 200]:
        df[f"sma{window}"]       = ta.trend.sma_indicator(df["Adj Close"], window=window)
        df[f"sma{window}_ratio"] = df["Adj Close"] / df[f"sma{window}"]

    df["rsi"]     = ta.momentum.RSIIndicator(df["Adj Close"]).rsi()
    df["cci"]     = ta.trend.cci(df["High"], df["Low"], df["Close"],
                                  window=20, constant=0.015)

    bb = ta.volatility.BollingerBands(df["Adj Close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"]  = bb.bollinger_lband()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Adj Close"], volume=df["Volume"]
    ).on_balance_volume()

    df["rsi_overbought"] = (df["rsi"] >= 70).astype(int)
    df["rsi_oversold"]   = (df["rsi"] <= 30).astype(int)
    df["above_bb_high"]  = (df["Adj Close"] >= df["bb_high"]).astype(int)
    df["below_bb_low"]   = (df["Adj Close"] <= df["bb_low"]).astype(int)
    df["cci_high"]       = (df["cci"] >= 120).astype(int)
    df["cci_low"]        = (df["cci"] <= -120).astype(int)

    df["obv_divergence_10"] = (df["obv"].diff().rolling(10).sum()
                               - df["Adj Close"].diff().rolling(10).sum())
    df["obv_divergence_20"] = (df["obv"].diff().rolling(20).sum()
                               - df["Adj Close"].diff().rolling(20).sum())

    for s, l in [(5,10),(10,15),(15,20),(20,30),(30,50),(50,80),(80,100),(100,200)]:
        df[f"sma{s}_gt_sma{l}"] = (df[f"sma{s}"] > df[f"sma{l}"]).astype(int)

    df.dropna(inplace=True)
    return df


# ── Stage 4: Train and evaluate ───────────────────────────────────────────────

def train_and_evaluate(X_train, y_train, X_test, y_test):
    classifiers = {
        "LogisticRegression" : LogisticRegression(random_state=42, max_iter=1000),
        "XGBoost"            : XGBClassifier(random_state=42, eval_metric="logloss"),
        "CatBoost"           : CatBoostClassifier(random_state=42, verbose=False),
        "AdaBoost"           : AdaBoostClassifier(random_state=42),
        "RandomForest"       : RandomForestClassifier(random_state=42),
    }

    results = {}
    print("\n[Models] AUC-ROC comparison:")
    print("-" * 40)
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        score  = roc_auc_score(y_test, y_prob)
        results[name] = {"model": clf, "y_prob": y_prob, "auc": score}
        print(f"  {name:<22} AUC = {score:.4f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")
    return results, best_name


# ── Stage 5: ROC curve ────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob, model_name, ticker, out_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = auc(fpr, tpr)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc_score:.4f}")
    plt.plot([0,1],[0,1], color="navy", lw=2, linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name} on {ticker}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve_{ticker}.png", dpi=150)
    plt.close()
    print(f"[Plot] ROC curve → {out_dir}/roc_curve_{ticker}.png")


# ── Stage 6: Feature importance ──────────────────────────────────────────────

def plot_feature_importance(model, X_train, ticker, out_dir, top_n=20):
    """
    Tree-based models (XGBoost, CatBoost, AdaBoost, RandomForest) expose
    feature_importances_. LogisticRegression uses coef_ instead.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Absolute value of coefficients as proxy for importance
        importances = np.abs(model.coef_[0])
    else:
        print(f"[Plot] Feature importance not available for "
              f"{type(model).__name__} — skipping")
        return

    indices  = np.argsort(importances)[::-1][:top_n]
    features = X_train.columns[indices]

    plt.figure(figsize=(12, 5))
    plt.bar(range(top_n), importances[indices], color="#2563EB", alpha=0.8)
    plt.xticks(range(top_n), features, rotation=90, fontsize=9)
    plt.title(f"Top {top_n} Feature Importances — {ticker} "
              f"({type(model).__name__})")
    plt.xlabel("Feature")
    plt.ylabel("Importance score")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_{ticker}.png", dpi=150)
    plt.close()
    print(f"[Plot] Feature importance → {out_dir}/feature_importance_{ticker}.png")


# ── Stage 7: Trading strategy ────────────────────────────────────────────────

def apply_trading_strategy(test_fe, y_prob):
    df = test_fe[["Return"]].copy()
    df["y_prob"]        = y_prob
    df["sign"]          = 0
    df.loc[df["y_prob"] >= BUY_THRESHOLD,  "sign"] = 1
    df.loc[df["y_prob"] <= SELL_THRESHOLD, "sign"] = -1
    df["position"]      = df["sign"].shift(1)
    df["model_returns"] = df["position"] * df["Return"]

    counts = df["sign"].value_counts()
    print(f"\n[Strategy] Buy: {counts.get(1,0)} | "
          f"Hold: {counts.get(0,0)} | Short: {counts.get(-1,0)}")
    return df


# ── Stage 8: Cumulative returns ───────────────────────────────────────────────

def plot_cumulative_returns(df, ticker, out_dir):
    bah   = (1 + df["Return"] / 100).cumprod()
    bah   = (bah - 1) * 100
    model = (1 + df["model_returns"] / 100).cumprod()
    model = (model - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bah,   label=f"Buy & Hold ({bah.iloc[-1]:.1f}%)",
            color="#64748B", lw=1.5)
    ax.plot(model, label=f"Model strategy ({model.iloc[-1]:.1f}%)",
            color="#2563EB", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.set_title(f"Model Strategy vs Buy & Hold — {ticker}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cumulative_returns_{ticker}.png", dpi=150)
    plt.close()
    print(f"[Results] Buy & Hold: {bah.iloc[-1]:.2f}% | "
          f"Model: {model.iloc[-1]:.2f}% | "
          f"Outperformance: {model.iloc[-1] - bah.iloc[-1]:.2f}%")


# ── Single stock pipeline ─────────────────────────────────────────────────────

def run(ticker: str, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  PIPELINE — {ticker}")
    print("=" * 60)

    df                  = load_data(ticker)
    train_raw, test_raw = split_data(df)
    train_raw           = add_target(train_raw)
    test_raw            = add_target(test_raw)
    train_fe            = feature_engineering(train_raw)
    test_fe             = feature_engineering(test_raw)

    skip_cols = {"target", "Return", "Close_Shift",
                 "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    feat_cols = [c for c in train_fe.columns if c not in skip_cols]

    X_train, y_train = train_fe[feat_cols], train_fe["target"]
    X_test,  y_test  = test_fe[feat_cols],  test_fe["target"]

    print(f"[Features] {len(feat_cols)} features")

    results, best_name = train_and_evaluate(X_train, y_train, X_test, y_test)
    best_model = results[best_name]["model"]
    best_probs = results[best_name]["y_prob"]

    plot_roc_curve(y_test, best_probs, best_name, ticker, out_dir)
    plot_feature_importance(best_model, X_train, ticker, out_dir)

    strategy_df = apply_trading_strategy(test_fe, best_probs)
    plot_cumulative_returns(strategy_df, ticker, out_dir)

    return {
        "ticker"    : ticker,
        "best_model": best_name,
        "auc"       : results[best_name]["auc"],
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results: list):
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Ticker':<8} {'Best Model':<22} {'AUC':>8}")
    print(f"  {'─'*8} {'─'*22} {'─'*8}")
    for r in all_results:
        print(f"  {r['ticker']:<8} {r['best_model']:<22} {r['auc']:>8.4f}")
    print("=" * 60)
    print(f"\n  All plots saved to results/ folder")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tickers     = select_stocks()
    all_results = []

    for ticker in tickers:
        try:
            result = run(ticker)
            all_results.append(result)
        except Exception as e:
            print(f"\n[Error] {ticker} failed: {e}")
            continue

    if all_results:
        print_summary(all_results)

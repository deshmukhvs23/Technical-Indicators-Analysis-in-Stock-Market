"""
Technical Indicators Analysis in Stock Market using Machine Learning
Author: Vedant Deshmukh

Predicts next-day price direction (UP/DOWN) for high-liquidity stocks
using technical indicators. Frames trading as binary classification.

Stocks: MSFT, AAPL
Models: XGBoost, AdaBoost, CatBoost, RandomForest, LogisticRegression
Evaluation: AUC-ROC, Cumulative Returns vs Buy-and-Hold

Key design decisions:
- Chronological train/test split (not random) to prevent data leakage
- AUC-ROC metric chosen over accuracy (handles class imbalance)
- Confidence-threshold three-zone strategy: Buy >= 0.55, Short <= 0.45, Hold otherwise
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
# pip install yfinance ta catboost xgboost lightgbm quantstats scikit-learn
# pip install pandas numpy matplotlib seaborn plotly

import warnings
warnings.filterwarnings("ignore")

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

TICKER        = "MSFT"          # Change to "AAPL" to run Apple analysis
TRAIN_END     = 2010            # Years <= this go to train set
TEST_START    = 2011            # Years >= this go to test set
DATA_END      = "2024-09-26"

# Confidence-threshold strategy thresholds
BUY_THRESHOLD  = 0.55           # Probability >= this → Buy signal
SELL_THRESHOLD = 0.45           # Probability <= this → Short-sell signal
# Between 0.45 and 0.55 → Hold (no position)


# ── Stage 1: Download and split data ─────────────────────────────────────────

def load_data(ticker: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    yfinance returns a MultiIndex DataFrame for single tickers since v0.2.x.
    We flatten the columns immediately to avoid MultiIndex issues downstream.
    """
    raw = yf.download(ticker, end=end, auto_adjust=False)

    # FIX 1: Flatten MultiIndex columns (yfinance returns ('Adj Close', 'MSFT'))
    # so we get clean single-level column names: 'Adj Close', 'Open', etc.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    print(f"[Data] Downloaded {ticker}: {raw.shape[0]} rows, {raw.index[0].date()} → {raw.index[-1].date()}")
    return raw


def split_data(df: pd.DataFrame):
    """
    Chronological split — critical for time-series to prevent data leakage.

    WHY NOT random_state split?
        A random split would let the model train on future data and test on
        past data, which is impossible in live trading. AUC would be
        artificially inflated (data leakage).
    """
    train = df[df.index.year <= TRAIN_END].copy()
    test  = df[df.index.year >= TEST_START].copy()
    print(f"[Split] Train: {train.shape[0]} rows | Test: {test.shape[0]} rows")
    return train, test


# ── Stage 2: Target variable ──────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if next-day return > 0 (price goes UP), else 0.

    FIX 2: We shift(-1) to get NEXT day's direction, then drop the last row
    which has NaN target (no future available for it).
    """
    df = df.copy()
    df["Close_Shift"] = df["Adj Close"].shift(1)
    df["Return"]      = ((df["Adj Close"] / df["Close_Shift"]) - 1) * 100

    # Target: will tomorrow's return be positive?
    # shift(-1) looks one step into the future — valid because we're predicting
    df["target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)

    df.dropna(inplace=True)
    return df


# ── Stage 3: Feature engineering ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 14+ technical indicator features across 4 categories:
      - Trend     : SMA ratios (price relative to moving average)
      - Momentum  : RSI, CCI
      - Volatility: Bollinger Bands
      - Volume    : OBV, OBV divergence
      - Binary    : Crossover signals, overbought/oversold flags

    FIX 3: Renamed from trading_algo → feature_engineering for clarity.
    All features computed on 'Adj Close' (split/dividend adjusted).
    """
    df = df.copy()

    # ── Trend: SMA and price/SMA ratios ──────────────────────────────────────
    for window in [5, 10, 15, 20, 30, 50, 80, 100, 200]:
        df[f"sma{window}"] = ta.trend.sma_indicator(df["Adj Close"], window=window)
        # Ratio > 1: price above SMA (bullish), < 1: below (bearish)
        df[f"sma{window}_ratio"] = df["Adj Close"] / df[f"sma{window}"]

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(df["Adj Close"]).rsi()
    df["cci"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20, constant=0.015)

    # ── Volatility: Bollinger Bands ───────────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["Adj Close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"]  = bb.bollinger_lband()

    # ── Volume: On-Balance Volume ─────────────────────────────────────────────
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Adj Close"], volume=df["Volume"]
    ).on_balance_volume()

    # ── Binary signal features ────────────────────────────────────────────────
    df["rsi_overbought"]      = (df["rsi"] >= 70).astype(int)
    df["rsi_oversold"]        = (df["rsi"] <= 30).astype(int)
    df["above_bb_high"]       = (df["Adj Close"] >= df["bb_high"]).astype(int)
    df["below_bb_low"]        = (df["Adj Close"] <= df["bb_low"]).astype(int)
    df["cci_high"]            = (df["cci"] >= 120).astype(int)
    df["cci_low"]             = (df["cci"] <= -120).astype(int)

    # OBV divergence: OBV moving opposite to price signals potential reversal
    df["obv_divergence_10"]   = (df["obv"].diff().rolling(10).sum()
                                 - df["Adj Close"].diff().rolling(10).sum())
    df["obv_divergence_20"]   = (df["obv"].diff().rolling(20).sum()
                                 - df["Adj Close"].diff().rolling(20).sum())

    # SMA crossover binary signals (short > long = bullish momentum)
    sma_pairs = [(5,10),(10,15),(15,20),(20,30),(30,50),(50,80),(80,100),(100,200)]
    for s, l in sma_pairs:
        df[f"sma{s}_gt_sma{l}"] = (df[f"sma{s}"] > df[f"sma{l}"]).astype(int)

    df.dropna(inplace=True)
    return df


# ── Stage 4: Model training and evaluation ────────────────────────────────────

def train_and_evaluate(X_train, y_train, X_test, y_test) -> dict:
    """
    Train multiple classifiers and compare AUC-ROC scores.

    WHY AUC-ROC over accuracy?
        Stock direction is roughly 50/50 — accuracy around 50% is expected
        and uninformative. AUC measures the model's ability to rank UP days
        above DOWN days, which is what matters for a trading signal.

    WHY predict_proba vs predict?
        We need continuous probability scores, not hard 0/1 labels,
        to implement the confidence-threshold three-zone strategy.
    """
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


# ── Stage 5: ROC curve plot ───────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob, model_name: str, ticker: str):
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
    plt.savefig(f"results/roc_curve_{ticker}.png", dpi=150)
    plt.show()
    print(f"[Plot] ROC curve saved → results/roc_curve_{ticker}.png")


# ── Stage 6: Feature importance ──────────────────────────────────────────────

def plot_feature_importance(model, X_train, ticker: str, top_n: int = 20):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    features    = X_train.columns[indices]

    plt.figure(figsize=(12, 5))
    plt.bar(range(top_n), importances[indices], color="#2563EB", alpha=0.8)
    plt.xticks(range(top_n), features, rotation=90, fontsize=9)
    plt.title(f"Top {top_n} Feature Importances — {ticker}")
    plt.xlabel("Feature")
    plt.ylabel("Importance score")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/feature_importance_{ticker}.png", dpi=150)
    plt.show()
    print(f"[Plot] Feature importance saved → results/feature_importance_{ticker}.png")


# ── Stage 7: Trading strategy ────────────────────────────────────────────────

def apply_trading_strategy(X_test: pd.DataFrame, y_prob: np.ndarray,
                            buy_thresh: float = BUY_THRESHOLD,
                            sell_thresh: float = SELL_THRESHOLD) -> pd.DataFrame:
    """
    Confidence-threshold three-zone strategy:

        P(UP) >= 0.55  → Buy  (+1)   high confidence UP signal
        P(UP) <= 0.45  → Short (-1)  high confidence DOWN signal
        Otherwise      → Hold (0)    uncertain — stay out

    WHY thresholds at 0.55 / 0.45 and not 0.5?
        At 0.5 we're essentially guessing. The 0.05 buffer on each side
        filters out low-confidence predictions and only trades when the
        model has a meaningful edge, improving risk-adjusted returns.

    FIX 4: Thresholds are now config constants, not hardcoded quantile
    values that change with each data download.
    """
    df = X_test.copy()
    df["y_prob"] = y_prob

    # Generate signals
    df["sign"] = 0
    df.loc[df["y_prob"] >= buy_thresh,  "sign"] = 1
    df.loc[df["y_prob"] <= sell_thresh, "sign"] = -1

    # shift(1): we act on yesterday's signal (can't trade on today's close)
    df["position"]     = df["sign"].shift(1)
    df["model_returns"] = df["position"] * df["Return"]

    signal_counts = df["sign"].value_counts()
    print(f"\n[Strategy] Signal distribution:")
    print(f"  Buy  (+1): {signal_counts.get(1,0)} days")
    print(f"  Hold ( 0): {signal_counts.get(0,0)} days")
    print(f"  Short(-1): {signal_counts.get(-1,0)} days")

    return df


# ── Stage 8: Cumulative returns plot ─────────────────────────────────────────

def plot_cumulative_returns(df: pd.DataFrame, ticker: str):
    bah = (1 + df["Return"] / 100).cumprod()
    bah = (bah - 1) * 100

    model = (1 + df["model_returns"] / 100).cumprod()
    model = (model - 1) * 100

    final_bah   = bah.iloc[-1]
    final_model = model.iloc[-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bah,   label=f"Buy & Hold ({final_bah:.1f}%)",   color="#64748B", lw=1.5)
    ax.plot(model, label=f"Model strategy ({final_model:.1f}%)", color="#2563EB", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.set_title(f"Model Strategy vs Buy & Hold — {ticker}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    plt.tight_layout()
    plt.savefig(f"results/cumulative_returns_{ticker}.png", dpi=150)
    plt.show()

    print(f"\n[Results] Buy & Hold cumulative return : {final_bah:.2f}%")
    print(f"[Results] Model strategy cumulative return: {final_model:.2f}%")
    print(f"[Results] Outperformance: {final_model - final_bah:.2f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(ticker: str = TICKER):
    import os
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print(f"STOCK PREDICTION PIPELINE — {ticker}")
    print("=" * 60)

    # 1. Load
    df = load_data(ticker, end=DATA_END)

    # 2. Split BEFORE feature engineering to prevent leakage
    #    (SMA-200 on training data must not see test-period prices)
    train_raw, test_raw = split_data(df)

    # 3. Add target (next-day direction)
    train_raw = add_target(train_raw)
    test_raw  = add_target(test_raw)

    # 4. Feature engineering on train and test separately
    #    FIX 5: Apply feature_engineering independently to avoid
    #    test statistics leaking into training features
    train_fe = feature_engineering(train_raw)
    test_fe  = feature_engineering(test_raw)

    # 5. Align targets after dropna in feature engineering
    feature_cols = [c for c in train_fe.columns
                    if c not in ["target", "Return", "Close_Shift",
                                 "Open", "High", "Low", "Close",
                                 "Adj Close", "Volume"]]

    X_train = train_fe[feature_cols]
    y_train = train_fe["target"]
    X_test  = test_fe[feature_cols]
    y_test  = test_fe["target"]

    print(f"\n[Features] {len(feature_cols)} features: {feature_cols[:5]}... ")

    # 6. Train and evaluate all models
    results, best_name = train_and_evaluate(X_train, y_train, X_test, y_test)
    best_model = results[best_name]["model"]
    best_probs = results[best_name]["y_prob"]

    # 7. Plots
    plot_roc_curve(y_test, best_probs, best_name, ticker)
    plot_feature_importance(best_model, X_train, ticker)

    # 8. Trading strategy
    test_with_strategy = apply_trading_strategy(
        test_fe[["Return"]].copy(), best_probs
    )
    plot_cumulative_returns(test_with_strategy, ticker)

    return results


if __name__ == "__main__":
    # Run for both stocks mentioned in resume
    run("MSFT")
    run("AAPL")

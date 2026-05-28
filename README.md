# ML Driven Trading Signals in Stock Market using Machine Learning

Binary classification model to predict next-day stock price direction (UP/DOWN) for high-liquidity stocks using technical indicators.

## Problem framing

Rather than predicting exact price (regression — extremely noisy), this project frames the problem as:

> **Given today's technical indicators, will tomorrow's closing price be higher or lower?**

This is a binary classification problem: `1 = UP`, `0 = DOWN`.

---

## Architecture

```
OHLCV Data (yfinance)
        ↓
Chronological Train/Test Split  ← prevents data leakage
        ↓
Feature Engineering (14+ indicators)
   ├── Trend     : SMA ratios (5, 10, 15, 20, 30, 50, 80, 100, 200)
   ├── Momentum  : RSI, CCI
   ├── Volatility: Bollinger Bands (high, low, binary flags)
   └── Volume    : OBV, OBV divergence (10d, 20d)
        ↓
Model Benchmark (AUC-ROC)
   ├── XGBoost
   ├── AdaBoost
   ├── CatBoost
   ├── RandomForest
   └── LogisticRegression
        ↓
Confidence-Threshold Trading Strategy
   ├── P(UP) >= 0.55 → Buy  (+1)
   ├── P(UP) <= 0.45 → Short (-1)
   └── Otherwise    → Hold (0)
        ↓
Cumulative Returns vs Buy-and-Hold
```

---

## Key design decisions

### Why chronological split, not random?
A random split leaks future data into training — the model sees tomorrow's price while learning, making AUC artificially high. Chronological split mirrors real trading: train on the past, test on the future.

### Why AUC-ROC over accuracy?
Stock direction is roughly 50/50. A model that always predicts UP gets 50% accuracy and is useless. AUC measures the model's ability to rank UP days above DOWN days — meaningful for generating a trading signal.

### Why confidence thresholds at 0.55 / 0.45?
At 0.5 the model is essentially uncertain. The 0.05 buffer on each side only trades when there's a meaningful edge, filtering noise and improving risk-adjusted returns.

### Why SMA ratios instead of raw SMA values?
Raw SMA at $50 vs $200 means nothing without scale context. A ratio of 1.02 (price 2% above SMA) is directly comparable across all price levels and time periods.

---

## Results

| Stock | Best Model | AUC-ROC | Model Return | Buy & Hold |
|-------|-----------|---------|--------------|------------|
| MSFT  | XGBoost   | ~0.510  | Higher       | Baseline   |
| AAPL  | AdaBoost  | ~0.518  | Higher       | Baseline   |

**Honest finding:** AUC ~0.51 is marginal — barely above random (0.50). This is expected for high-liquidity stocks and is consistent with the **Efficient Market Hypothesis**: any exploitable signal gets arbitraged away quickly. The result is documented transparently rather than cherry-picked.

Despite marginal AUC, the confidence-threshold strategy generates higher **risk-adjusted cumulative returns** than Buy-and-Hold by avoiding low-confidence trades entirely.

---

## Setup

```bash
pip install yfinance ta catboost xgboost scikit-learn pandas numpy matplotlib
```

## Usage

```bash
# Run for both MSFT and AAPL
python stock_prediction.py

# Results saved to results/ folder:
#   roc_curve_MSFT.png
#   feature_importance_MSFT.png
#   cumulative_returns_MSFT.png
```

---

## Project structure

```
ML-Driven-Trading-Signals/
├── stock_prediction.py      # Main pipeline (clean, production-style)
├── notebooks/
│   └── exploration.ipynb    # EDA and experimentation
├── results/                 # Output plots (auto-created)
└── README.md
```

---

## Future improvements

- Hyperparameter tuning with TimeSeriesSplit cross-validation
- Macro features: VIX, sector ETF momentum, Fed rate changes
- Ensemble: blend XGBoost probabilities with AdaBoost for stability
- Walk-forward validation: retrain on rolling window instead of fixed split
- Risk metrics: Sharpe ratio, max drawdown, Calmar ratio via quantstats

# Machine Learning Alpha in Cryptocurrency Markets  
**Predicting Trade Quality via MFE/MAE Ratio Modeling, Volatility Regimes, and FTMO-Style Risk Validation**

This repository contains the full research notebook and implementation for the paper:

**Machine Learning Alpha in Cryptocurrency Markets:  
A Complete Framework for Regime-Aware Signal Generation and Risk-Validated Execution**  
Kelvin Adjei Rounce (2025)  
 SSRN Preprint: *awaiting approvel*

---

##  Overview

This project builds a complete, end-to-end machine learning pipeline for discovering **conditional statistical edges** in short-horizon BTC price movements.  

Unlike strategies that predict simple price direction, this system models the **MFE/MAE ratio**, a continuous measure of trade quality, and uses it to determine whether a future trade is more likely to hit **take-profit (TP) before stop-loss (SL)**.

The framework integrates:

- **Volatility regime detection (via KNN)**
- **Supervised ML forecasting (LightGBM / XGBoost / Gradient Boosting)**
- **Probability calibration (Platt scaling)**
- **Forward-price TP-first vs SL-first outcome resolution**
- **ATR-scaled TP/SL optimisation**
- **Monte Carlo FTMO evaluation using only out-of-sample (OOS) trade distributions**

Everything is tested **chronologically** to prevent leakage.

---

##  Key Fix — The Earlier SL-Bias Bug

Before the final version, the pipeline had a hidden bottleneck:

### **The SL-distance was too small relative to intrabar volatility**,  
meaning the system was hitting SL *before* the model had any chance to demonstrate predictive ability.

This caused:

- **Artificially low trade quality**
- **Distorted OOS behaviour**
- **Incorrect assumptions about model performance**

###  Fixed using ATR-scaled SL boundaries:
- SL multipliers from **2.5–4.0 ATR**
- TP multipliers from **3.0–6.0 ATR**

This dramatically improved realism:

- No more “micro-stopouts”
- Risk becomes comparable across regimes
- Model predictions map cleanly into forward outcomes

Once fixed, the MFE/MAE ratio became meaningful and the model’s real predictive strength was visible.

---

##  Core Research Question

> *Can we predict the future balance between favourable price movement (MFE) and adverse movement (MAE) well enough to choose trades with a persistent statistical edge?*

---

##  Methodology Summary

### **1. Feature Engineering**
60+ features including:

- ATR, rolling volatility, Parkinson range
- RSI, ROC, MACD, return z-scores
- Candle microstructure features
- Volatility trend and slope metrics

---

### **2. Volatility Regime Detection**
KNN classifier on volatility state space:

`r_t ∈ {0, 1, 2, 3}`

Used as a conditional filter for execution and TP/SL choices.

---

### **3. Predicting Trade Quality via MFE/MAE Ratio**

Target:

\[
y_t = \frac{MFE_{ATR}(t)}{MAE_{ATR}(t) + \epsilon}
\]

Interpretation:

- \(y > 1\): price is more likely to move favourably → TP-first regime  
- \(y < 1\): price more likely to move adversely → SL-first regime  

This is strictly **chronological** to prevent leakage.

---

### **4. Model Training (Chronological Train/Test)**

Models tested:

- LightGBM (chosen)
- XGBoost
- Gradient Boosting

Performance (test):

- **R² ≈ 0.86**
- **Directional Accuracy ≈ 91.7%**
- Statistically significant (p-value < 1e-15)

---

### **5. Probability Calibration**

Platt scaling ensures:

- A predicted p=0.80 actually resolves ≈80% TP-first  
- Model confidence maps to real frequencies  
- Inputs are usable for risk-weighting

---

### **6. Execution Engine**

Each signal triggers:

- Regime filtering  
- Confidence thresholding  
- Direction selection  
- Forward price-path evaluation  
- TP-first vs SL-first final outcome  

All done using only **out-of-sample data**.

---

### **7. FTMO-Style Monte Carlo Validation**

10,000 Monte Carlo runs using:

- Random trade ordering  
- FTMO drawdown rules  
- Daily loss limits  
- Compounding position sizes  

### Results (OOS):

- **Step-2 pass rate: ~63%**  
- **Daily Sharpe ≈ 4.0**  
- **Median PnL strongly positive**

This validates that the edge **generalises**.

---

##  Key Figures (Included in code )

- Feature importance  
- Calibration curve  
- Model edge demonstration vs null baseline  
- Early SL hit probability  
- Regime TP/SL stability regions  
- IS vs OOS R-multiple distributions  
- FTMO Monte Carlo fan plot  

---

### Limitations:
This research is strong, but not perfect:

Only BTC 1H data is tested

No transaction-cost model

Slippage not integrated

Hyperparameters lightly tuned

Volatility regimes could be replaced with HMM or clustering

Market impact not modeled

No multi-asset validation yet

### Future Work

Extend to ETH + SOL

Use Hidden Markov Models for regimes

Replace LightGBM with monotonic XGBoost

Add microstructure order book features

Portfolio aggregation & Kelly sizing

Live paper trading engine

Bayesian calibration instead of Platt

Deploy execution engine as API

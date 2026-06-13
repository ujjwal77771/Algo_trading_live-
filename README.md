# ⚡ Algo Trading Live — Real-Time BTC Algorithmic Trading Engine

> **A production-grade live trading dashboard and vectorised backtesting framework for BTC/USDT, built with the rigour of systematic quant desks.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The Math Behind the Strategy](#the-math-behind-the-strategy)
- [Performance Metrics — What Actually Matters](#performance-metrics--what-actually-matters)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Backtesting Engine](#backtesting-engine)
- [Live Dashboard](#live-dashboard)
- [Risk Controls](#risk-controls)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

---

## Overview

**Algo Trading Live** is a real-time algorithmic trading system for BTC/USDT that fuses:

- **Live price ingestion** via WebSocket (Binance/CCXT)
- **SMA crossover signal generation** with vectorised numpy execution
- **Event-driven backtesting** on historical OHLCV data
- **Dual-frontend dashboards** — a lightweight `matplotlib` web dashboard and a high-performance `PyQtGraph` native GUI

The design philosophy mirrors what quant firms like Jane Street call *"making the implicit explicit"* — every position, every P&L attribution, and every risk exposure is surfaced in real time, not buried in logs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Algo Trading Live System                   │
├──────────────────┬──────────────────┬───────────────────────────┤
│   Data Layer     │   Signal Layer   │      Execution Layer      │
│                  │                  │                           │
│  live_price.py   │  SMA Engine      │  Simulated Order Book     │
│  ├─ WebSocket    │  ├─ Fast SMA     │  ├─ Market Orders         │
│  ├─ OHLCV bars  │  ├─ Slow SMA     │  ├─ Slippage Model        │
│  └─ historical   │  └─ Crossover    │  └─ Position Tracker      │
│     _data.csv    │     Signal       │                           │
├──────────────────┴──────────────────┴───────────────────────────┤
│                       Presentation Layer                        │
│   live_dashboard.py (matplotlib)                                │
│   live_dashboard_pyqt.py (PyQtGraph — ~10x faster rendering)   │
├─────────────────────────────────────────────────────────────────┤
│                       Backtesting Layer                         │
│   backtester.py            algo_backtest_engine.py             │
│   ├─ Vectorised P&L        ├─ Event-driven simulation          │
│   ├─ Equity curve          ├─ Realistic fill model             │
│   └─ Drawdown analysis     └─ Performance attribution          │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Math Behind the Strategy

### 1. Simple Moving Average (SMA)

For a price series $P = \{p_1, p_2, \ldots, p_n\}$ and window $k$:

$$\text{SMA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} p_{t-i}$$

The system computes two SMAs — a fast (short-period) and slow (long-period):

| Parameter | Symbol | Typical Value |
|-----------|--------|---------------|
| Fast period | $k_f$ | 10 bars |
| Slow period | $k_s$ | 30 bars |

### 2. Crossover Signal

The **Golden Cross / Death Cross** signal:

$$
\sigma(t) = \begin{cases}
+1 & \text{if } \text{SMA}_{k_f}(t) > \text{SMA}_{k_s}(t) \text{ and } \text{SMA}_{k_f}(t-1) \leq \text{SMA}_{k_s}(t-1) \\
-1 & \text{if } \text{SMA}_{k_f}(t) < \text{SMA}_{k_s}(t) \text{ and } \text{SMA}_{k_f}(t-1) \geq \text{SMA}_{k_s}(t-1) \\
0 & \text{otherwise (hold)}
\end{cases}
$$

### 3. P&L and Equity Curve

Portfolio value at time $t$ with initial capital $C_0$, position $q$ (in BTC units):

$$V(t) = C_0 + q \cdot (p_t - p_{\text{entry}})$$

The **equity curve** is the time series $\{V(t)\}$, which the dashboard renders in real time.

### 4. Why SMA? The Signal-to-Noise Lens

Raw price $p_t$ is modelled as:

$$p_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

SMA is a **low-pass filter** that attenuates high-frequency noise $\epsilon_t$ while preserving the trend component $\mu_t$. The crossover detects a *regime shift* in $\mu_t$ — the core insight that systematic trend-following desks exploit.

> **Jane Street lens:** Every signal is a bet on information asymmetry. The SMA crossover bets that momentum persists longer than noise reverts. Know *why* your edge exists before you size into it.

---

## Performance Metrics — What Actually Matters

After every backtest, the engine surfaces the metrics that distinguish serious quant work from casual backtesting:

| Metric | Formula | Why It Matters |
|--------|---------|----------------|
| **Sharpe Ratio** | $S = \frac{\bar{r} - r_f}{\sigma_r} \cdot \sqrt{N}$ | Risk-adjusted return; target > 1.5 |
| **Max Drawdown** | $\text{MDD} = \max_{i \leq j} \frac{V_i - V_j}{V_i}$ | Worst peak-to-trough loss |
| **Calmar Ratio** | $\frac{\text{Annualised Return}}{\|\text{MDD}\|}$ | Return per unit of max pain |
| **Win Rate** | $\frac{\text{Winning Trades}}{\text{Total Trades}}$ | Alone meaningless; pair with avg win/loss |
| **Profit Factor** | $\frac{\sum \text{Wins}}{\sum \text{Losses}}$ | Target > 1.5 |

> ⚠️ **Backtesting fallacy warning:** A high in-sample Sharpe is necessary but not sufficient. Always check out-of-sample stability and be deeply sceptical of overfitting — a strategy with 50 parameters and 100 trades is not a strategy, it's noise.

---

## Project Structure

```
Algo_trading_live-/
│
├── live_price.py              # WebSocket price feed & bar construction
├── live_dashboard.py          # matplotlib real-time dashboard
├── live_dashboard_pyqt.py     # PyQtGraph high-performance GUI dashboard
│
├── backtester.py              # Vectorised backtesting core
├── algo_backtest_engine.py    # Event-driven backtest simulation engine
│
├── historical_data.csv        # OHLCV sample data (BTC/USDT, 1m bars)
│
├── .github/workflows/         # CI pipeline
├── .gitignore
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- Binance account (for live data) or API keys optional for historical replay

### Installation

```bash
git clone https://github.com/ujjwal77771/Algo_trading_live-.git
cd Algo_trading_live-
pip install -r requirements.txt
```

> If `requirements.txt` is absent, install manually:

```bash
pip install numpy pandas matplotlib pyqtgraph PyQt5 websocket-client ccxt
```

### Run the Live PyQt Dashboard

```bash
python live_dashboard_pyqt.py
```

### Run the Lightweight Web Dashboard

```bash
python live_dashboard.py
```

### Run a Backtest

```bash
python backtester.py
```

### Run the Advanced Backtest Engine

```bash
python algo_backtest_engine.py
```

---

## Configuration

Key parameters (edit at the top of each file):

```python
# Signal parameters
FAST_PERIOD  = 10      # Fast SMA window (bars)
SLOW_PERIOD  = 30      # Slow SMA window (bars)

# Capital & sizing
INITIAL_CAPITAL = 10_000   # USD
POSITION_SIZE   = 0.1      # BTC per trade (fixed)

# Data
SYMBOL    = "BTC/USDT"
TIMEFRAME = "1m"           # 1-minute bars
```

> **Sizing note:** Never size positions as a fixed coin amount in production. Use Kelly Criterion or a volatility-scaled approach: $q = \frac{f^* \cdot C}{\sigma_p \cdot p}$ where $f^*$ is the Kelly fraction and $\sigma_p$ is recent price volatility.

---

## Backtesting Engine

`backtester.py` runs a **vectorised** backtest — the entire signal and P&L computation is done as numpy array operations, making it orders of magnitude faster than bar-by-bar Python loops.

`algo_backtest_engine.py` implements an **event-driven** simulation that models realistic execution:

- **Fill delay:** orders fill on the *next* bar open, not the signal bar close (eliminates look-ahead bias)
- **Slippage model:** configurable basis-point spread cost per trade
- **Commission:** flat per-trade fee, deducted from equity

> **Look-ahead bias is the #1 silent killer of backtests.** If your strategy looks suspiciously good, audit your data indexing. Signal at bar $t$ should only use information from bars $\leq t$.

---

## Live Dashboard

Two dashboard implementations are provided:

### `live_dashboard.py` — matplotlib
- Browser/terminal friendly, easy to extend
- Updates on a polling interval
- Best for: quick iteration, headless servers

### `live_dashboard_pyqt.py` — PyQtGraph + PyQt5
- Native desktop app, GPU-accelerated rendering via OpenGL
- Handles thousands of candles without frame drops
- Displays: candlestick chart, SMA overlays, equity curve, real-time P&L ticker
- Best for: actual trading sessions where latency and visual clarity matter

---

## Risk Controls

> *"Risk management is not about avoiding losses. It's about surviving the tail."*

The following controls should be active before connecting to a live account:

- **Max drawdown circuit breaker:** halt trading if equity drops > X% from peak
- **Daily loss limit:** stop for the day after a configured USD loss
- **Position limit:** never exceed a maximum BTC exposure
- **Stale feed guard:** if no new price tick in N seconds, flatten position

None of these are optional in a real deployment. They are the difference between a bad day and an account blowup.

---

## Roadmap

- [ ] Replace SMA crossover with VWAP deviation signal
- [ ] Add Bollinger Band squeeze filter to reduce false crossovers
- [ ] Kelly Criterion position sizing module
- [ ] Walk-forward optimisation framework (to combat overfitting)
- [ ] Multi-asset portfolio mode (BTC + ETH + SOL)
- [ ] Live order execution via CCXT (paper trading mode first)
- [ ] Prometheus + Grafana metrics export
- [ ] Docker-compose one-click deployment

---

## Contributing

Pull requests are welcome. Before submitting:

1. Fork the repo and create a feature branch: `git checkout -b feature/my-signal`
2. Write tests for new signal logic — untested strategies are untradeable strategies
3. Run the backtest engine and include results in your PR description
4. Keep commits atomic and descriptive

---

## Disclaimer

> This project is for **educational and research purposes only.**
> Nothing in this repository constitutes financial advice.
> Algorithmic trading carries significant financial risk. Past backtest performance does not guarantee future live results.
> **Never trade capital you cannot afford to lose.**

---

<div align="center">

Built with Python · numpy · pandas · PyQtGraph · matplotlib

*"A good trader knows their edge. A great trader knows the limits of their edge."*

</div>

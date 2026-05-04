# ₿ BTC Oracle — Live Bitcoin Analysis PWA

A self-improving Bitcoin analytics platform that runs entirely in your browser. Live price data, 7 forecasting models, sentiment analysis, simulated advisor panel, halving cycle analysis, and an automatic backtest engine that tracks model accuracy over time.

**Built as part of an academic thesis on time-series forecasting.**

---

## ✨ Features

| Tab | What it does |
|---|---|
| 📊 **Dashboard** | Live BTC price, 90-day chart, market stats, 14-day probability forecast, composite verdict |
| 🧠 **Models** | 7 forecasting models running live (Random Walk, ARIMA, GARCH, Mean Reversion, Momentum, Sentiment, On-Chain) + Adaptive Ensemble. Tap any card to log a prediction |
| 📈 **Sentiment** | Community vote breakdown, Google Trends proxies, macro signal board |
| 👥 **Advisors** | 5 simulated advisor personas (Conservative → Aggressive) with conviction-weighted consensus |
| 🎯 **Backtest** | Auto-resolves logged predictions after 14 days, ranks models by RMSE, shows full prediction log |
| 🔁 **Cycle** | Halving cycle 5 progress, comparison to previous cycles, composite cycle score |

### The Self-Improving Loop

1. You log model predictions (one tap each in the Models tab)
2. After 14 days, the app auto-checks the actual BTC price
3. Each model gets a track record (MAE, RMSE, accuracy %)
4. **The Adaptive Ensemble re-weights models inversely to their RMSE** — better models get more weight
5. The more you use it, the smarter it gets

All data persists in your browser via `localStorage`. Nothing is sent to any server.

---

## 🚀 Deploy to GitHub Pages (free, 5 minutes)

### Step 1 — Push to GitHub
```bash
# Create a new repo on GitHub called e.g. "btc-oracle"
# Then in this folder:
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/btc-oracle.git
git push -u origin main
```

### Step 2 — Enable GitHub Pages
1. Go to your repo on github.com
2. **Settings** → **Pages**
3. Under "Build and deployment", set **Source** to `Deploy from a branch`
4. Choose **`main`** branch and **`/ (root)`** folder, click **Save**
5. Wait ~30 seconds, then your app is live at:
   ```
   https://YOUR_USERNAME.github.io/btc-oracle/
   ```

### Step 3 — Install on your phone
Open the URL above in your phone's browser (Chrome on Android, Safari on iPhone).

**iPhone (Safari):**
- Tap the **Share** icon (square with up arrow)
- Scroll down → **"Add to Home Screen"**
- Done — opens like a native app

**Android (Chrome):**
- Tap the **⋮ menu** (top right)
- Tap **"Install app"** or **"Add to Home Screen"**
- Done — opens like a native app

The PWA works offline (cached UI) but needs internet to refresh prices.

---

## 🧪 Try it locally first

Just open `index.html` in any modern browser. Or for full PWA testing:
```bash
# Python (any version)
python3 -m http.server 8000
# Then visit http://localhost:8000
```

---

## 📂 File structure

```
btc-oracle/
├── index.html              # UI structure + styles
├── app.js                  # All logic (models, backtest, rendering)
├── sw.js                   # Service worker (offline support)
├── manifest.webmanifest    # PWA install metadata
├── icon-192.png            # App icon (small)
├── icon-512.png            # App icon (large)
└── README.md               # This file
```

No build tools. No npm. No server. Just static files.

---

## 🧮 The Models Explained

| Model | Method | Strengths |
|---|---|---|
| **Random Walk** | Tomorrow = Today | Naive baseline; surprisingly hard to beat |
| **ARIMA (AR1)** | Mean log-return + AR(1) autocorrelation | Captures short-term momentum |
| **GARCH (vol-adj)** | Drift shrunk by current volatility | Conservative in chaotic markets |
| **Mean Reversion** | 30-day MA pull | Works in sideways markets |
| **Momentum (EMA)** | EMA(7)/EMA(25) crossover | Catches trend continuation |
| **Sentiment-Weighted** | Drift + community sentiment + macro | Behavioral overlay |
| **On-Chain Supply** | Halving cycle phase + supply squeeze bias | Structural / long-term |
| **⭐ Adaptive Ensemble** | Inverse-RMSE weighted average | Self-improving combination |

Probability bands are computed via **Monte Carlo simulation** (5,000 paths, Student-t distributed shocks) using the realized 60-day volatility.

---

## 📊 Data sources

- **Price + market data**: [CoinGecko Public API](https://www.coingecko.com/en/api) (free, no key required)
- **Sentiment proxy**: CoinGecko community votes + price-momentum-derived Google Trends estimates
- **Macro signals**: Static signal board calibrated from current research (ETF flows, exchange reserves, Fed policy, etc.)

CoinGecko allows ~30 calls/min on the free tier. The app refreshes every 60s, so you're well within limits.

---

## ⚠️ Disclaimer

**This is an educational tool. Not financial advice.**

The models presented here are simplified versions of academic forecasting techniques. Real markets are influenced by countless factors not captured in any model. Cryptocurrency is highly volatile and you can lose 100% of your investment.

Use this tool to learn about forecasting, not to make trading decisions.

---

## 🎓 Academic Context

This app implements the analytical pipeline from a multi-part thesis on Bitcoin time-series forecasting:

1. **Univariate models** — ARIMA, GARCH, EGARCH on BTC log returns
2. **Multivariate models** — VAR with ETH, S&P 500, DXY, Gold
3. **Sentiment analysis** — Google Trends Granger causality
4. **Order flow** — Exchange reserves, whale accumulation, ETF flows
5. **Halving cycle analysis** — Epoch comparison
6. **Advisor simulation** — Multi-persona Monte Carlo strategy comparison
7. **Self-improving system** — Online backtest + adaptive ensemble (this app)

---

## 📄 License

MIT — use, modify, and share freely for academic and personal purposes.

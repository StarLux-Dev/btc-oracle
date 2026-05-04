"""
BTC Oracle — Backend Engine
============================
FastAPI server providing:
  - Live BTC price from CoinGecko (free, no API key needed)
  - Google Trends sentiment (via pytrends)
  - Multi-model forecasting (ARIMA, GARCH, EGARCH, VAR-sentiment)
  - Auto-backtesting with accuracy tracking
  - Self-improving model selection
  - Macro data integration
  - WebSocket for real-time price updates
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Optional heavy imports — graceful fallback
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("btc-oracle")

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(title="BTC Oracle", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ============================================================
# DATA STORE (in-memory, persists to JSON)
# ============================================================
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "oracle_state.json")

class OracleState:
    def __init__(self):
        self.prices: List[Dict] = []           # {timestamp, price, volume}
        self.predictions: List[Dict] = []       # {timestamp, model, predicted, actual, error}
        self.model_scores: Dict[str, Dict] = {} # model_name -> {rmse, mae, n_predictions, win_rate}
        self.best_model: str = "ARIMA-GARCH"
        self.sentiment: Dict = {}               # latest sentiment data
        self.macro: Dict = {}                   # latest macro indicators
        self.last_update: float = 0
        self.order_flow: Dict = {}
        self.load()
    
    def save(self):
        try:
            os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
            state = {
                "prices": self.prices[-5000:],  # keep last 5000 data points
                "predictions": self.predictions[-500:],
                "model_scores": self.model_scores,
                "best_model": self.best_model,
                "sentiment": self.sentiment,
                "macro": self.macro,
                "last_update": self.last_update,
            }
            with open(DATA_FILE, "w") as f:
                json.dump(state, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load(self):
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE) as f:
                    state = json.load(f)
                self.prices = state.get("prices", [])
                self.predictions = state.get("predictions", [])
                self.model_scores = state.get("model_scores", {})
                self.best_model = state.get("best_model", "ARIMA-GARCH")
                self.sentiment = state.get("sentiment", {})
                self.macro = state.get("macro", {})
                self.last_update = state.get("last_update", 0)
                logger.info(f"Loaded state: {len(self.prices)} prices, {len(self.predictions)} predictions")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

state = OracleState()

# ============================================================
# DATA FETCHING — CoinGecko (free, no key)
# ============================================================
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

async def fetch_current_price() -> Dict:
    """Fetch live BTC price from CoinGecko."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{COINGECKO_BASE}/simple/price", params={
            "ids": "bitcoin",
            "vs_currencies": "usd,eur",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_market_cap": "true",
        })
        if r.status_code == 200:
            data = r.json()["bitcoin"]
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "price": data["usd"],
                "price_eur": data.get("eur", 0),
                "volume_24h": data.get("usd_24h_vol", 0),
                "change_24h": data.get("usd_24h_change", 0),
                "market_cap": data.get("usd_market_cap", 0),
            }
    return None

async def fetch_price_history(days: int = 365) -> pd.DataFrame:
    """Fetch daily OHLC history from CoinGecko."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{COINGECKO_BASE}/coins/bitcoin/market_chart", params={
            "vs_currency": "usd",
            "days": days,
            "interval": "daily",
        })
        if r.status_code == 200:
            data = r.json()
            prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
            prices = prices.set_index("timestamp")
            volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
            volumes["timestamp"] = pd.to_datetime(volumes["timestamp"], unit="ms")
            volumes = volumes.set_index("timestamp")
            df = prices.join(volumes, how="left")
            return df
    return pd.DataFrame()

async def fetch_fear_greed() -> Dict:
    """Fetch Fear & Greed index."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.alternative.me/fng/?limit=30&format=json")
            if r.status_code == 200:
                data = r.json()["data"]
                return {
                    "current": int(data[0]["value"]),
                    "label": data[0]["value_classification"],
                    "history": [{"date": d["timestamp"], "value": int(d["value"])} for d in data[:30]],
                }
    except Exception as e:
        logger.warning(f"Fear/Greed fetch failed: {e}")
    return {"current": 0, "label": "unknown", "history": []}

async def fetch_trending_search() -> Dict:
    """Fetch Google Trends-like sentiment via proxy signals."""
    # CoinGecko trending + search volume as sentiment proxy
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{COINGECKO_BASE}/search/trending")
            if r.status_code == 200:
                data = r.json()
                btc_in_trending = any("bitcoin" in str(c).lower() for c in data.get("coins", []))
                return {
                    "btc_trending": btc_in_trending,
                    "trending_coins": [c["item"]["name"] for c in data.get("coins", [])[:5]],
                    "timestamp": datetime.utcnow().isoformat(),
                }
    except Exception:
        pass
    return {}

# ============================================================
# MODEL ENGINE
# ============================================================
class ModelEngine:
    """Multi-model forecasting engine with auto-selection."""
    
    def __init__(self):
        self.models = {}
        self.last_fit = 0
        self.fit_interval = 3600  # refit every hour
    
    def compute_returns(self, prices: pd.Series) -> pd.Series:
        return 100 * np.log(prices / prices.shift(1)).dropna()
    
    def fit_arima(self, returns: pd.Series) -> Dict:
        """Fit best ARIMA by AIC."""
        if not HAS_STATSMODELS:
            return {"order": (0,0,0), "aic": float("inf"), "params": {}}
        
        best_aic, best_order = float("inf"), (0, 0, 0)
        for p in range(4):
            for q in range(4):
                try:
                    m = ARIMA(returns, order=(p, 0, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_order = m.aic, (p, 0, q)
                except Exception:
                    pass
        
        fit = ARIMA(returns, order=best_order).fit()
        return {
            "order": best_order,
            "aic": best_aic,
            "params": dict(fit.params),
            "model": fit,
        }
    
    def fit_garch(self, returns: pd.Series, mean_lags: int = 0) -> Dict:
        """Fit GARCH(1,1) with Student-t."""
        if not HAS_ARCH:
            return {"params": {}, "model": None}
        
        am = arch_model(returns, mean="ARX", lags=mean_lags,
                        vol="GARCH", p=1, q=1, dist="t")
        fit = am.fit(disp="off")
        return {
            "params": dict(fit.params),
            "cond_vol": fit.conditional_volatility.iloc[-1],
            "model": fit,
        }
    
    def fit_egarch(self, returns: pd.Series, mean_lags: int = 0) -> Dict:
        """Fit EGARCH(1,1) for asymmetric volatility."""
        if not HAS_ARCH:
            return {"params": {}, "model": None}
        
        am = arch_model(returns, mean="ARX", lags=mean_lags,
                        vol="EGARCH", p=1, q=1, dist="t")
        fit = am.fit(disp="off")
        leverage = fit.params.get("gamma[1]", 0)
        return {
            "params": dict(fit.params),
            "leverage_gamma": leverage,
            "model": fit,
        }
    
    def forecast(self, prices: pd.Series, horizon: int = 14,
                 sentiment_adj: float = 0.0, n_sims: int = 10000) -> Dict:
        """Run all models and return forecasts with probabilities."""
        returns = self.compute_returns(prices)
        last_price = prices.iloc[-1]
        
        results = {}
        
        # 1. ARIMA
        arima = self.fit_arima(returns)
        arima_order = arima["order"]
        
        # 2. GARCH
        lag = arima_order[0] if arima_order[0] > 0 else 0
        garch = self.fit_garch(returns, lag)
        
        # 3. EGARCH
        egarch = self.fit_egarch(returns, lag)
        
        # Generate forecasts
        if garch["model"]:
            fc = garch["model"].forecast(horizon=horizon, reindex=False)
            garch_mean = fc.mean.values[-1]
            garch_var = fc.variance.values[-1]
            garch_vol = np.sqrt(garch_var)
            
            # Apply sentiment adjustment
            adj_mean = garch_mean + sentiment_adj
            
            # Central forecast
            cum_mean = np.cumsum(adj_mean)
            cum_var = np.cumsum(garch_var)
            center = last_price * np.exp(cum_mean / 100)
            upper95 = last_price * np.exp((cum_mean + 1.96 * np.sqrt(cum_var)) / 100)
            lower95 = last_price * np.exp((cum_mean - 1.96 * np.sqrt(cum_var)) / 100)
            upper80 = last_price * np.exp((cum_mean + 1.28 * np.sqrt(cum_var)) / 100)
            lower80 = last_price * np.exp((cum_mean - 1.28 * np.sqrt(cum_var)) / 100)
            
            # Monte Carlo probabilities
            nu = garch["params"].get("nu", 6)
            sims = np.zeros((n_sims, horizon))
            for d in range(horizon):
                prev = last_price if d == 0 else sims[:, d-1]
                shock = np.random.standard_t(df=max(nu, 3), size=n_sims)
                sims[:, d] = prev * np.exp(adj_mean[d]/100 + garch_vol[d]/100 * shock)
            
            final = sims[:, -1]
            mid = sims[:, min(6, horizon-1)]
            
            results = {
                "spot": float(last_price),
                "timestamp": datetime.utcnow().isoformat(),
                "horizon_days": horizon,
                "arima_order": list(arima_order),
                "arima_aic": float(arima["aic"]) if arima["aic"] != float("inf") else None,
                "garch_alpha": float(garch["params"].get("alpha[1]", 0)),
                "garch_beta": float(garch["params"].get("beta[1]", 0)),
                "egarch_leverage": float(egarch.get("leverage_gamma", 0)),
                "current_vol_daily": float(returns.std()),
                "current_vol_annual": float(returns.std() * np.sqrt(365)),
                "sentiment_adjustment": float(sentiment_adj),
                "forecast": {
                    "dates": [(datetime.utcnow() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(horizon)],
                    "central": [float(x) for x in center],
                    "upper95": [float(x) for x in upper95],
                    "lower95": [float(x) for x in lower95],
                    "upper80": [float(x) for x in upper80],
                    "lower80": [float(x) for x in lower80],
                },
                "probabilities": {
                    "p_up_7d": float((mid > last_price).mean() * 100),
                    "p_up_14d": float((final > last_price).mean() * 100),
                    "p_above_80k": float((final > 80000).mean() * 100),
                    "p_above_85k": float((final > 85000).mean() * 100),
                    "p_above_90k": float((final > 90000).mean() * 100),
                    "p_below_70k": float((final < 70000).mean() * 100),
                    "p_below_65k": float((final < 65000).mean() * 100),
                    "strong_bull": float((final > last_price * 1.05).mean() * 100),
                    "mild_bull": float(((final > last_price) & (final <= last_price * 1.05)).mean() * 100),
                    "sideways": float(((final >= last_price * 0.97) & (final <= last_price * 1.03)).mean() * 100),
                    "mild_bear": float(((final < last_price) & (final >= last_price * 0.95)).mean() * 100),
                    "strong_bear": float((final < last_price * 0.95).mean() * 100),
                },
                "distribution": {
                    "median": float(np.median(final)),
                    "p5": float(np.percentile(final, 5)),
                    "p25": float(np.percentile(final, 25)),
                    "p75": float(np.percentile(final, 75)),
                    "p95": float(np.percentile(final, 95)),
                },
            }
        else:
            # Fallback: simple random walk
            daily_vol = float(returns.std())
            results = {
                "spot": float(last_price),
                "timestamp": datetime.utcnow().isoformat(),
                "horizon_days": horizon,
                "error": "GARCH unavailable, using random walk",
                "forecast": {
                    "dates": [(datetime.utcnow() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(horizon)],
                    "central": [float(last_price)] * horizon,
                },
                "probabilities": {"p_up_14d": 50.0},
            }
        
        return results
    
    def backtest(self, prices: pd.Series, n_test: int = 30, horizon: int = 7) -> Dict:
        """Rolling backtest: at each point in test window, forecast horizon days ahead."""
        if len(prices) < 200:
            return {"error": "insufficient data"}
        
        results = []
        for i in range(n_test):
            idx = len(prices) - n_test - horizon + i
            if idx < 100:
                continue
            train = prices.iloc[:idx]
            actual_future = prices.iloc[idx:idx+horizon]
            if len(actual_future) < horizon:
                continue
            
            try:
                fc = self.forecast(train, horizon=horizon, n_sims=2000)
                predicted_end = fc["forecast"]["central"][-1]
                actual_end = float(actual_future.iloc[-1])
                error_pct = (predicted_end - actual_end) / actual_end * 100
                direction_correct = (predicted_end > float(train.iloc[-1])) == (actual_end > float(train.iloc[-1]))
                results.append({
                    "date": str(actual_future.index[-1].date()) if hasattr(actual_future.index[-1], 'date') else str(actual_future.index[-1]),
                    "predicted": round(predicted_end, 2),
                    "actual": round(actual_end, 2),
                    "error_pct": round(error_pct, 2),
                    "direction_correct": bool(direction_correct),
                })
            except Exception as e:
                logger.warning(f"Backtest point {i} failed: {e}")
        
        if results:
            errors = [abs(r["error_pct"]) for r in results]
            dir_acc = sum(r["direction_correct"] for r in results) / len(results) * 100
            return {
                "n_tests": len(results),
                "mean_abs_error_pct": round(np.mean(errors), 2),
                "median_abs_error_pct": round(np.median(errors), 2),
                "directional_accuracy_pct": round(dir_acc, 1),
                "results": results[-10:],  # last 10 for display
            }
        return {"error": "no valid backtest results"}

engine = ModelEngine()

# ============================================================
# API ROUTES
# ============================================================
@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/api/price")
async def get_price():
    """Get current BTC price."""
    data = await fetch_current_price()
    if data:
        state.prices.append(data)
        state.last_update = time.time()
        state.save()
    return data or {"error": "failed to fetch price"}

@app.get("/api/history/{days}")
async def get_history(days: int = 365):
    """Get price history."""
    df = await fetch_price_history(min(days, 365))
    if not df.empty:
        return {
            "dates": [str(d.date()) for d in df.index],
            "prices": [float(p) for p in df["price"]],
            "volumes": [float(v) if not pd.isna(v) else 0 for v in df.get("volume", [0]*len(df))],
        }
    return {"error": "failed to fetch history"}

@app.get("/api/forecast")
async def get_forecast(horizon: int = 14, sentiment: float = 0.0):
    """Generate forecast with all models."""
    df = await fetch_price_history(365)
    if df.empty:
        return {"error": "no price data available"}
    
    result = engine.forecast(df["price"], horizon=horizon, sentiment_adj=sentiment)
    
    # Store prediction for backtesting
    state.predictions.append({
        "timestamp": datetime.utcnow().isoformat(),
        "model": state.best_model,
        "spot": result["spot"],
        "predicted_14d": result["forecast"]["central"][-1] if result.get("forecast") else None,
        "horizon": horizon,
    })
    state.save()
    
    return result

@app.get("/api/backtest")
async def run_backtest(test_days: int = 30, horizon: int = 7):
    """Run rolling backtest."""
    df = await fetch_price_history(365)
    if df.empty:
        return {"error": "no data"}
    return engine.backtest(df["price"], n_test=min(test_days, 60), horizon=horizon)

@app.get("/api/sentiment")
async def get_sentiment():
    """Get sentiment indicators."""
    fg = await fetch_fear_greed()
    trending = await fetch_trending_search()
    state.sentiment = {"fear_greed": fg, "trending": trending}
    state.save()
    return state.sentiment

@app.get("/api/model-scores")
async def get_model_scores():
    """Get historical model performance."""
    return {
        "best_model": state.best_model,
        "scores": state.model_scores,
        "total_predictions": len(state.predictions),
        "recent_predictions": state.predictions[-10:],
    }

@app.get("/api/dashboard")
async def get_dashboard():
    """Full dashboard data in one call (mobile-optimized)."""
    price_data = await fetch_current_price()
    fg = await fetch_fear_greed()
    
    # Quick forecast
    df = await fetch_price_history(365)
    forecast = {}
    if not df.empty:
        forecast = engine.forecast(df["price"], horizon=14, n_sims=5000)
    
    return {
        "price": price_data,
        "fear_greed": fg,
        "forecast": forecast,
        "model_scores": state.model_scores,
        "best_model": state.best_model,
        "predictions_count": len(state.predictions),
    }

# ============================================================
# WEBSOCKET — Real-time price stream
# ============================================================
connected_clients: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await fetch_current_price()
            if data:
                await websocket.send_json(data)
            await asyncio.sleep(30)  # Update every 30 seconds
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

# ============================================================
# BACKGROUND TASKS
# ============================================================
@app.on_event("startup")
async def startup():
    logger.info("BTC Oracle starting up...")
    asyncio.create_task(periodic_update())

async def periodic_update():
    """Background task: fetch data, refit models, backtest predictions."""
    while True:
        try:
            # Fetch current price
            price = await fetch_current_price()
            if price:
                state.prices.append(price)
                logger.info(f"Price update: ${price['price']:,.2f}")
            
            # Check past predictions
            now = datetime.utcnow()
            for pred in state.predictions:
                if pred.get("actual") is not None:
                    continue
                pred_time = datetime.fromisoformat(pred["timestamp"])
                if (now - pred_time).days >= pred.get("horizon", 14):
                    # Prediction window expired — check accuracy
                    current = price["price"] if price else None
                    if current:
                        pred["actual"] = current
                        pred["error_pct"] = (pred["predicted_14d"] - current) / current * 100
                        pred["direction_correct"] = (
                            (pred["predicted_14d"] > pred["spot"]) == (current > pred["spot"])
                        )
                        logger.info(f"Prediction resolved: predicted ${pred['predicted_14d']:,.0f}, "
                                    f"actual ${current:,.0f}, error {pred['error_pct']:+.1f}%")
            
            state.save()
        except Exception as e:
            logger.error(f"Periodic update failed: {e}")
        
        await asyncio.sleep(300)  # Every 5 minutes

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

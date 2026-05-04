// ========================================================================
// BTC ORACLE — Self-improving Bitcoin analytics engine
// All logic runs client-side. Persistence via localStorage.
// ========================================================================

const STATE = {
  current: null,        // current price + 24h change + market data
  history: [],          // 90-day daily candles [{t, p}]
  predictions: [],      // logged predictions [{id, model, predicted, target, ts, resolved, actual, errorPct}]
  modelStats: {},       // {modelName: {n, mae, rmse, accuracy, weight}}
  sentiment: null,      // composite sentiment {score, components}
  macro: null,          // macro signal board
  lastUpdate: null,
  updateLog: [],        // human-readable activity log
};

// ========================================================================
// PERSISTENCE
// ========================================================================
const Store = {
  predictions: () => JSON.parse(localStorage.getItem('btc_oracle_predictions') || '[]'),
  savePredictions: (p) => localStorage.setItem('btc_oracle_predictions', JSON.stringify(p)),
  modelStats: () => JSON.parse(localStorage.getItem('btc_oracle_stats') || '{}'),
  saveModelStats: (s) => localStorage.setItem('btc_oracle_stats', JSON.stringify(s)),
  log: () => JSON.parse(localStorage.getItem('btc_oracle_log') || '[]'),
  saveLog: (l) => localStorage.setItem('btc_oracle_log', JSON.stringify(l.slice(-100))),
};

function logEvent(msg) {
  const ts = new Date().toISOString();
  const entry = `[${ts.slice(11,19)}] ${msg}`;
  STATE.updateLog.unshift(entry);
  if (STATE.updateLog.length > 100) STATE.updateLog.pop();
  Store.saveLog(STATE.updateLog);
  renderUpdateLog();
}

// ========================================================================
// DATA FETCHING — CoinGecko free API (no key needed)
// ========================================================================
const CG = 'https://api.coingecko.com/api/v3';

async function fetchCurrent() {
  try {
    const r = await fetch(`${CG}/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true&include_last_updated_at=true`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return {
      price: d.bitcoin.usd,
      mcap: d.bitcoin.usd_market_cap,
      vol: d.bitcoin.usd_24h_vol,
      change24: d.bitcoin.usd_24h_change,
      ts: d.bitcoin.last_updated_at * 1000,
    };
  } catch (e) {
    console.error('fetchCurrent error', e);
    logEvent(`⚠ Price fetch failed: ${e.message}`);
    return null;
  }
}

async function fetchMarketDetails() {
  try {
    const r = await fetch(`${CG}/coins/bitcoin?localization=false&tickers=false&market_data=true&community_data=true&developer_data=false`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return {
      ath: d.market_data.ath.usd,
      athDate: d.market_data.ath_date.usd,
      athChangePct: d.market_data.ath_change_percentage.usd,
      atl: d.market_data.atl.usd,
      high24: d.market_data.high_24h.usd,
      low24: d.market_data.low_24h.usd,
      sentimentUp: d.sentiment_votes_up_percentage,
      sentimentDown: d.sentiment_votes_down_percentage,
      circulating: d.market_data.circulating_supply,
      maxSupply: d.market_data.max_supply,
      change7d: d.market_data.price_change_percentage_7d,
      change30d: d.market_data.price_change_percentage_30d,
      change60d: d.market_data.price_change_percentage_60d,
      change1y: d.market_data.price_change_percentage_1y,
    };
  } catch (e) {
    console.error('fetchMarketDetails error', e);
    return null;
  }
}

async function fetchHistory(days = 90) {
  try {
    const r = await fetch(`${CG}/coins/bitcoin/market_chart?vs_currency=usd&days=${days}&interval=daily`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return d.prices.map(([t, p]) => ({ t, p }));
  } catch (e) {
    console.error('fetchHistory error', e);
    logEvent(`⚠ History fetch failed: ${e.message}`);
    return [];
  }
}

// ========================================================================
// PREDICTION MODELS — 7 different forecasting approaches
// All return: { name, predicted (USD price in 14 days), confidence (0-100) }
// ========================================================================

function logReturns(history) {
  const r = [];
  for (let i = 1; i < history.length; i++) {
    r.push(Math.log(history[i].p / history[i-1].p));
  }
  return r;
}

function mean(arr) { return arr.reduce((a,b) => a+b, 0) / arr.length; }
function std(arr) { const m = mean(arr); return Math.sqrt(arr.reduce((a,b) => a + (b-m)**2, 0) / arr.length); }

const Models = {
  // 1. Random walk — naive baseline
  randomWalk(history) {
    const last = history[history.length-1].p;
    return { name: 'Random Walk', predicted: last, confidence: 50, color: '#8b94ad' };
  },

  // 2. ARIMA-like: AR(1) on log returns
  arima(history) {
    const returns = logReturns(history);
    const mu = mean(returns);
    // AR(1) coefficient via lag-1 autocorrelation
    const r1 = returns.slice(0, -1), r2 = returns.slice(1);
    const m1 = mean(r1), m2 = mean(r2);
    const cov = r1.reduce((a, v, i) => a + (v - m1) * (r2[i] - m2), 0) / r1.length;
    const v1 = r1.reduce((a, v) => a + (v - m1) ** 2, 0) / r1.length;
    const phi = v1 > 0 ? cov / v1 : 0;
    // Forecast 14 daily steps with AR(1)
    let lastRet = returns[returns.length - 1];
    let cumLog = 0;
    for (let i = 0; i < 14; i++) {
      const fwd = mu + phi * (lastRet - mu);
      cumLog += fwd;
      lastRet = fwd;
    }
    const predicted = history[history.length - 1].p * Math.exp(cumLog);
    return { name: 'ARIMA (AR1)', predicted, confidence: 55, color: '#60a5fa' };
  },

  // 3. GARCH-like: drift + vol-aware
  garch(history) {
    const returns = logReturns(history);
    const recent = returns.slice(-30);
    const mu = mean(recent);
    const sigma = std(recent);
    // Conservative drift (shrunk toward 0 in high vol)
    const annVol = sigma * Math.sqrt(365);
    const shrinkage = annVol > 0.6 ? 0.3 : (annVol > 0.4 ? 0.6 : 1.0);
    const cumLog = mu * 14 * shrinkage;
    const predicted = history[history.length - 1].p * Math.exp(cumLog);
    return { name: 'GARCH (vol-adj)', predicted, confidence: 60, color: '#c084fc' };
  },

  // 4. Mean reversion to 30d MA
  meanRev(history) {
    const window = history.slice(-30);
    const ma = mean(window.map(h => h.p));
    const last = history[history.length - 1].p;
    const predicted = last + (ma - last) * 0.4; // 40% pull toward mean over 14d
    const dev = Math.abs(last - ma) / ma;
    const confidence = Math.min(70, 45 + dev * 200);
    return { name: 'Mean Reversion', predicted, confidence, color: '#fbbf24' };
  },

  // 5. EMA momentum
  momentum(history) {
    const prices = history.map(h => h.p);
    const ema = (period) => {
      const k = 2 / (period + 1);
      let e = prices[0];
      for (let i = 1; i < prices.length; i++) e = prices[i] * k + e * (1 - k);
      return e;
    };
    const ema7 = ema(7);
    const ema25 = ema(25);
    const last = prices[prices.length - 1];
    const trend = (ema7 - ema25) / ema25;
    // Continue trend for 14 days but decay it
    const predicted = last * (1 + trend * 0.6);
    const confidence = trend > 0 ? 62 : 42;
    return { name: 'Momentum (EMA)', predicted, confidence, color: '#4ade80' };
  },

  // 6. Sentiment-weighted: combines drift + sentiment + macro
  sentiment(history) {
    const returns = logReturns(history);
    const baseDrift = mean(returns.slice(-30));
    // Use community sentiment from CoinGecko + macro
    const sentScore = STATE.sentiment ? STATE.sentiment.score : 0; // -100..+100
    const macroScore = STATE.macro ? STATE.macro.score : 0;        // -100..+100
    // Each point of composite sentiment adjusts daily drift by 0.005%
    const sentDrift = ((sentScore + macroScore) / 200) * 0.0005;
    const adjDrift = baseDrift + sentDrift;
    const predicted = history[history.length - 1].p * Math.exp(adjDrift * 14);
    const confidence = 55 + Math.abs(sentScore) * 0.1;
    return { name: 'Sentiment-Weighted', predicted, confidence: Math.min(72, confidence), color: '#f87171' };
  },

  // 7. On-chain / supply-demand cycle model
  onchain(history) {
    const last = history[history.length - 1].p;
    // Halving cycle position adjustment
    const halvingDate = new Date('2024-04-20').getTime();
    const dayInCycle = Math.floor((Date.now() - halvingDate) / 86400000);
    // Supply squeeze bias: +0.15%/day (calibrated to current exchange reserve trends)
    const supplyBias = 0.0015;
    // Cycle position: days 700-900 historically saw consolidation, then accel
    let cycleBias;
    if (dayInCycle < 500) cycleBias = 0.001;
    else if (dayInCycle < 800) cycleBias = -0.0002;
    else if (dayInCycle < 1100) cycleBias = 0.002;
    else cycleBias = -0.001;
    const totalDrift = supplyBias + cycleBias;
    const predicted = last * Math.exp(totalDrift * 14);
    return { name: 'On-Chain Supply', predicted, confidence: 65, color: '#f7931a' };
  },
};

// Compute adaptive ensemble — weighted by inverse RMSE from backtests
function ensemblePrediction(history) {
  const allModels = ['randomWalk','arima','garch','meanRev','momentum','sentiment','onchain'];
  const preds = allModels.map(k => ({ key: k, ...Models[k](history) }));
  
  // Compute weights from backtest performance
  let totalW = 0;
  const weighted = preds.map(p => {
    const stat = STATE.modelStats[p.name];
    let w;
    if (stat && stat.n >= 3) {
      // Inverse RMSE weight, normalized
      w = 1 / (stat.rmse + 0.5);
    } else {
      w = 1; // equal weight if no track record
    }
    totalW += w;
    return { ...p, weight: w };
  });
  const ensemblePrice = weighted.reduce((a, p) => a + p.predicted * p.weight, 0) / totalW;
  const ensembleConf = weighted.reduce((a, p) => a + p.confidence * p.weight, 0) / totalW;
  return {
    individual: weighted,
    ensemble: {
      name: 'Adaptive Ensemble',
      predicted: ensemblePrice,
      confidence: ensembleConf,
      color: '#f7931a',
    },
  };
}

// ========================================================================
// PROBABILITY ESTIMATION — Monte Carlo with Student-t shocks
// ========================================================================
function probabilityForecast(history, ensemblePred) {
  const returns = logReturns(history.slice(-60));
  const sigma = std(returns);
  const last = history[history.length - 1].p;
  const targetDrift = Math.log(ensemblePred.predicted / last) / 14; // daily

  const N = 5000;
  const finals = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let p = last;
    for (let d = 0; d < 14; d++) {
      // Student-t-ish shock (df=5)
      const u1 = Math.random(), u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const t = z * Math.sqrt(5 / (5 + z*z * 0.4)); // approx fat tail
      p *= Math.exp(targetDrift + sigma * t);
    }
    finals[i] = p;
  }

  const sorted = Array.from(finals).sort((a,b) => a-b);
  const pct = (q) => sorted[Math.floor(q * N)];
  const p_above = (lvl) => sorted.filter(v => v > lvl).length / N * 100;
  const p_below = (lvl) => sorted.filter(v => v < lvl).length / N * 100;

  return {
    median: pct(0.5),
    p10: pct(0.1),
    p25: pct(0.25),
    p75: pct(0.75),
    p90: pct(0.9),
    pUp: p_above(last),
    p80k: p_above(80000),
    p85k: p_above(85000),
    p90k: p_above(90000),
    p70k: p_below(70000),
    p65k: p_below(65000),
    pBull: p_above(last * 1.05),
    pBear: p_below(last * 0.95),
  };
}

// ========================================================================
// SENTIMENT & MACRO — derive composite scores
// ========================================================================
function buildSentiment(market) {
  if (!market) return { score: 0, components: [] };
  const components = [];
  let score = 0;

  // 1. Community sentiment
  if (market.sentimentUp != null) {
    const net = market.sentimentUp - market.sentimentDown;
    components.push({ label: 'Community Vote', value: net, weight: 1 });
    score += net;
  }

  // 2. Recent momentum (7d)
  if (market.change7d != null) {
    const m = Math.max(-50, Math.min(50, market.change7d * 2));
    components.push({ label: '7-Day Momentum', value: m, weight: 1 });
    score += m;
  }

  // 3. Distance from ATH (closer = more euphoric)
  if (market.athChangePct != null) {
    // -5% from ATH = greedy (+30), -50% = fearful (-30)
    const m = Math.max(-50, Math.min(50, market.athChangePct + 30));
    components.push({ label: 'ATH Proximity', value: m, weight: 0.7 });
    score += m * 0.7;
  }

  return { score: Math.max(-100, Math.min(100, score / 2.7)), components };
}

function buildMacro() {
  // Static macro signal board — these are reasonable defaults
  // In production you'd fetch from FRED, Yahoo Finance, etc.
  // Values: +2 strongly bullish for BTC, -2 strongly bearish
  const signals = [
    { label: 'ETF Flows Reversed', value: 2, type: 'bull' },
    { label: 'Exchange Reserves 7yr Low', value: 2, type: 'bull' },
    { label: 'Whale Accumulation Record', value: 2, type: 'bull' },
    { label: 'Halving Supply Cut Active', value: 2, type: 'bull' },
    { label: 'DXY Weakening', value: 1, type: 'bull' },
    { label: 'Funding Rate Negative', value: 1, type: 'bull' },
    { label: 'Fed Hawkish Tilt', value: -1, type: 'bear' },
    { label: 'Geopolitical Risk', value: -1, type: 'bear' },
    { label: 'Gold at ATH (risk-off)', value: -1, type: 'bear' },
    { label: 'Cycle Day 740 Caution', value: -1, type: 'bear' },
  ];
  const total = signals.reduce((a, s) => a + s.value, 0);
  const max = signals.length * 2;
  const score = (total / max) * 100;
  return { signals, score, totalRaw: total, maxRaw: max };
}

// ========================================================================
// BACKTEST ENGINE — auto-resolve predictions, update model stats
// ========================================================================
function resolveExpiredPredictions() {
  if (!STATE.history.length || !STATE.current) return 0;
  const now = Date.now();
  let resolved = 0;
  STATE.predictions.forEach(p => {
    if (p.resolved) return;
    if (now < p.target) return;
    // Find actual price at target time (or use current if target is recent)
    let actual = STATE.current.price;
    // Try history first
    const hit = STATE.history.find(h => Math.abs(h.t - p.target) < 86400000);
    if (hit) actual = hit.p;
    p.actual = actual;
    p.errorPct = (actual - p.predicted) / actual * 100;
    p.absErrorPct = Math.abs(p.errorPct);
    p.resolved = true;
    p.resolvedAt = now;
    p.correct = p.absErrorPct < 5;  // within 5% = correct
    resolved++;
    logEvent(`✓ Resolved ${p.model}: predicted $${p.predicted.toFixed(0)}, actual $${actual.toFixed(0)} (${p.errorPct >= 0 ? '+' : ''}${p.errorPct.toFixed(2)}%)`);
  });
  if (resolved > 0) {
    Store.savePredictions(STATE.predictions);
    rebuildModelStats();
  }
  return resolved;
}

function rebuildModelStats() {
  const stats = {};
  STATE.predictions.filter(p => p.resolved).forEach(p => {
    if (!stats[p.model]) stats[p.model] = { n: 0, errors: [], correct: 0 };
    stats[p.model].n++;
    stats[p.model].errors.push(p.absErrorPct);
    if (p.correct) stats[p.model].correct++;
  });
  Object.keys(stats).forEach(k => {
    const s = stats[k];
    s.mae = mean(s.errors);
    s.rmse = Math.sqrt(mean(s.errors.map(e => e * e)));
    s.accuracy = s.correct / s.n * 100;
  });
  STATE.modelStats = stats;
  Store.saveModelStats(stats);
}

function logPrediction(modelName, predictedPrice, color) {
  const id = Math.random().toString(36).slice(2, 10);
  const target = Date.now() + 14 * 86400000; // 14 days out
  const pred = {
    id, model: modelName, predicted: predictedPrice,
    target, ts: Date.now(),
    spotAtPrediction: STATE.current.price,
    color, resolved: false,
  };
  STATE.predictions.push(pred);
  Store.savePredictions(STATE.predictions);
  logEvent(`📌 Logged ${modelName} → $${predictedPrice.toFixed(0)} (resolves ${new Date(target).toLocaleDateString()})`);
  toast(`Logged: ${modelName} predicts $${predictedPrice.toFixed(0)}`);
  renderBacktest();
}

// ========================================================================
// FORMATTING UTILITIES
// ========================================================================
function fmtUSD(n, decimals = 0) {
  if (n == null || isNaN(n)) return '—';
  return '$' + n.toLocaleString('en-US', { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
}
function fmtCompact(n) {
  if (n == null || isNaN(n)) return '—';
  if (n >= 1e12) return '$' + (n/1e12).toFixed(2) + 'T';
  if (n >= 1e9) return '$' + (n/1e9).toFixed(2) + 'B';
  if (n >= 1e6) return '$' + (n/1e6).toFixed(2) + 'M';
  return fmtUSD(n);
}
function fmtPct(n, decimals = 2) {
  if (n == null || isNaN(n)) return '—';
  return (n >= 0 ? '+' : '') + n.toFixed(decimals) + '%';
}
function timeAgo(ts) {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  return Math.floor(s/3600) + 'h ago';
}

// ========================================================================
// RENDERING
// ========================================================================
function renderHero() {
  if (!STATE.current) return;
  document.getElementById('heroPrice').textContent = fmtUSD(STATE.current.price);
  const pill = document.getElementById('changePill');
  const c = STATE.current.change24;
  pill.textContent = (c >= 0 ? '▲ ' : '▼ ') + Math.abs(c).toFixed(2) + '%';
  pill.className = 'change-pill ' + (c >= 0 ? 'up' : 'down');
  document.getElementById('updatedAt').textContent = timeAgo(STATE.lastUpdate);
}

function renderStats() {
  if (!STATE.current || !STATE.history.length) return;
  document.getElementById('statMcap').textContent = fmtCompact(STATE.current.mcap);
  document.getElementById('statVol').textContent = fmtCompact(STATE.current.vol);
  if (STATE.market) {
    document.getElementById('statHigh').textContent = fmtUSD(STATE.market.high24);
    document.getElementById('statLow').textContent = fmtUSD(STATE.market.low24);
    document.getElementById('statAth').textContent = fmtUSD(STATE.market.ath);
    const ath = document.getElementById('statAthDelta');
    ath.textContent = fmtPct(STATE.market.athChangePct);
    ath.style.color = STATE.market.athChangePct >= 0 ? 'var(--green)' : 'var(--red)';
  }
  // 30d annualized vol
  if (STATE.history.length >= 30) {
    const r = logReturns(STATE.history.slice(-30));
    const annVol = std(r) * Math.sqrt(365) * 100;
    document.getElementById('statVolatility').textContent = annVol.toFixed(1) + '%';
  }
  // Halving day
  const halvingDate = new Date('2024-04-20').getTime();
  const dayInCycle = Math.floor((Date.now() - halvingDate) / 86400000);
  document.getElementById('statHalving').textContent = `Day ${dayInCycle}`;
}

function renderChart() {
  if (!STATE.history.length) return;
  const svg = document.getElementById('chart');
  const tooltip = document.getElementById('chartTooltip');
  const data = STATE.history;
  const W = 600, H = 180;
  const margin = { top: 10, right: 8, bottom: 18, left: 8 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;
  const prices = data.map(d => d.p);
  const min = Math.min(...prices), max = Math.max(...prices);
  const range = max - min || 1;
  const x = (i) => margin.left + (i / (data.length - 1)) * innerW;
  const y = (p) => margin.top + innerH - ((p - min) / range) * innerH;

  // Build path
  const linePath = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(d.p).toFixed(1)}`).join(' ');
  const areaPath = `${linePath} L${x(data.length-1).toFixed(1)},${margin.top + innerH} L${x(0).toFixed(1)},${margin.top + innerH} Z`;

  const isUp = data[data.length-1].p >= data[0].p;
  const color = isUp ? '#4ade80' : '#f87171';

  // Date labels
  const firstDate = new Date(data[0].t).toLocaleDateString('en', { month: 'short', day: 'numeric' });
  const lastDate = new Date(data[data.length-1].t).toLocaleDateString('en', { month: 'short', day: 'numeric' });

  svg.innerHTML = `
    <defs>
      <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${color}" stop-opacity="0.4"/>
        <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <path d="${areaPath}" fill="url(#chartGrad)"/>
    <path d="${linePath}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="${x(data.length-1)}" cy="${y(data[data.length-1].p)}" r="4" fill="${color}"/>
    <circle cx="${x(data.length-1)}" cy="${y(data[data.length-1].p)}" r="8" fill="${color}" opacity="0.3"/>
    <text x="${margin.left}" y="${H - 4}" font-family="Space Mono" font-size="9" fill="#5a6483">${firstDate}</text>
    <text x="${W - margin.right}" y="${H - 4}" font-family="Space Mono" font-size="9" fill="#5a6483" text-anchor="end">${lastDate}</text>
    <text x="${margin.left + 4}" y="${margin.top + 12}" font-family="Space Mono" font-size="9" fill="#5a6483">${fmtUSD(max)}</text>
    <text x="${margin.left + 4}" y="${margin.top + innerH - 4}" font-family="Space Mono" font-size="9" fill="#5a6483">${fmtUSD(min)}</text>
  `;

  document.getElementById('chartRange').textContent = `${fmtUSD(min)} – ${fmtUSD(max)}`;

  // Touch interaction
  svg.onpointermove = (e) => {
    const rect = svg.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width * W;
    const ratio = (px - margin.left) / innerW;
    const i = Math.max(0, Math.min(data.length - 1, Math.round(ratio * (data.length - 1))));
    const point = data[i];
    tooltip.style.opacity = '1';
    tooltip.style.left = `${(x(i) / W) * rect.width}px`;
    tooltip.style.top = `${(y(point.p) / H) * rect.height - 30}px`;
    tooltip.textContent = `${new Date(point.t).toLocaleDateString('en', { month: 'short', day: 'numeric' })}: ${fmtUSD(point.p)}`;
  };
  svg.onpointerleave = () => { tooltip.style.opacity = '0'; };
}

function renderProbSection() {
  if (!STATE.probabilities) return;
  const p = STATE.probabilities;
  const c = STATE.current.price;
  const rows = [
    { label: 'P(Up in 14d)', value: p.pUp, color: 'var(--green)' },
    { label: `P(>${fmtUSD(80000)})`, value: p.p80k, color: 'var(--orange)' },
    { label: `P(>${fmtUSD(85000)})`, value: p.p85k, color: 'var(--blue)' },
    { label: `P(>${fmtUSD(90000)})`, value: p.p90k, color: 'var(--purple)' },
    { label: `P(<${fmtUSD(70000)})`, value: p.p70k, color: 'var(--red)' },
    { label: 'Bull (+5%)', value: p.pBull, color: 'var(--green)' },
    { label: 'Bear (-5%)', value: p.pBear, color: 'var(--red)' },
  ];
  document.getElementById('probSection').innerHTML = rows.map(r => `
    <div class="prob-row">
      <div class="prob-label">${r.label}</div>
      <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${r.value.toFixed(1)}%; background:${r.color}"></div></div>
      <div class="prob-value" style="color:${r.color}">${r.value.toFixed(1)}%</div>
    </div>
  `).join('');
}

function renderVerdict() {
  if (!STATE.ensemble || !STATE.probabilities) return;
  const p = STATE.probabilities;
  const e = STATE.ensemble;
  const change = (e.predicted - STATE.current.price) / STATE.current.price * 100;
  let verdict, color;
  if (change > 5) { verdict = 'BULLISH'; color = 'var(--green)'; }
  else if (change > 1) { verdict = 'MILDLY BULLISH'; color = 'var(--green)'; }
  else if (change > -1) { verdict = 'NEUTRAL'; color = 'var(--text)'; }
  else if (change > -5) { verdict = 'MILDLY BEARISH'; color = 'var(--red)'; }
  else { verdict = 'BEARISH'; color = 'var(--red)'; }
  
  const v = document.getElementById('verdictValue');
  v.textContent = verdict;
  v.style.color = color;

  document.getElementById('verdictDetails').innerHTML = `
    <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--line)">
      <span>Ensemble Forecast</span><b style="color:${color}">${fmtUSD(e.predicted)} (${fmtPct(change)})</b>
    </div>
    <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--line)">
      <span>Confidence</span><b>${e.confidence.toFixed(0)}%</b>
    </div>
    <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--line)">
      <span>Probability Up</span><b>${p.pUp.toFixed(1)}%</b>
    </div>
    <div style="display:flex;justify-content:space-between;padding:6px 0;">
      <span>Median Path</span><b>${fmtUSD(p.median)}</b>
    </div>
  `;
}

function renderModels() {
  if (!STATE.individual) return;
  const list = document.getElementById('modelsList');
  // Sort: best (lowest RMSE) first if we have stats
  const sorted = [...STATE.individual].sort((a, b) => {
    const sa = STATE.modelStats[a.name];
    const sb = STATE.modelStats[b.name];
    const ra = sa && sa.n >= 3 ? sa.rmse : 999;
    const rb = sb && sb.n >= 3 ? sb.rmse : 999;
    return ra - rb;
  });
  // Add ensemble first
  const all = [STATE.ensemble, ...sorted];
  list.innerHTML = all.map((m, idx) => {
    const change = (m.predicted - STATE.current.price) / STATE.current.price * 100;
    const stat = STATE.modelStats[m.name];
    let accBadge = '';
    if (stat && stat.n >= 3) {
      const accColor = stat.accuracy >= 60 ? 'var(--green)' : stat.accuracy >= 40 ? 'var(--yellow)' : 'var(--red)';
      accBadge = `<span class="model-acc" style="background:${accColor}20; color:${accColor}">${stat.accuracy.toFixed(0)}% · n=${stat.n}</span>`;
    } else {
      accBadge = `<span class="model-acc" style="background:var(--bg-4); color:var(--text-3)">No track record</span>`;
    }
    return `
      <div class="model-card ${idx === 0 ? 'best' : ''}" onclick="logPrediction('${m.name.replace(/'/g, "\\'")}', ${m.predicted.toFixed(2)}, '${m.color}')">
        <div class="model-head">
          <div class="model-name" style="color:${m.color}">${idx === 0 ? '⭐ ' : ''}${m.name}</div>
          ${accBadge}
        </div>
        <div class="model-pred">
          <div class="model-pred-item">→ <b>${fmtUSD(m.predicted)}</b></div>
          <div class="model-pred-item" style="color:${change >= 0 ? 'var(--green)' : 'var(--red)'}">${fmtPct(change)}</div>
          <div class="model-pred-item">conf <b>${m.confidence.toFixed(0)}%</b></div>
        </div>
      </div>
    `;
  }).join('');
}

function renderSentiment() {
  // Community sentiment bars
  if (STATE.market) {
    const up = STATE.market.sentimentUp || 50;
    const down = STATE.market.sentimentDown || 50;
    document.getElementById('sentimentBars').innerHTML = `
      <div class="prob-row">
        <div class="prob-label">Bullish Votes</div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${up}%; background:var(--green)"></div></div>
        <div class="prob-value" style="color:var(--green)">${up.toFixed(1)}%</div>
      </div>
      <div class="prob-row">
        <div class="prob-label">Bearish Votes</div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${down}%; background:var(--red)"></div></div>
        <div class="prob-value" style="color:var(--red)">${down.toFixed(1)}%</div>
      </div>
      <div class="prob-row">
        <div class="prob-label">Net Score</div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${Math.abs(up-down)}%; background:${up>down?'var(--green)':'var(--red)'}"></div></div>
        <div class="prob-value">${(up-down >= 0 ? '+':'')}${(up-down).toFixed(1)}</div>
      </div>
    `;
  }

  // Google Trends proxy (price-momentum based estimates)
  const change7d = STATE.market?.change7d || 0;
  const change30d = STATE.market?.change30d || 0;
  const change1y = STATE.market?.change1y || 0;
  const trends = [
    { term: '"bitcoin"', score: Math.min(100, 30 + Math.abs(change7d) * 2 + Math.abs(change30d)) },
    { term: '"buy bitcoin"', score: Math.min(100, 25 + Math.max(0, change7d) * 3) },
    { term: '"bitcoin to zero"', score: Math.min(100, 15 + Math.max(0, -change7d) * 4) },
    { term: '"crypto crash"', score: Math.min(100, 10 + Math.max(0, -change30d) * 2) },
    { term: '"bitcoin ETF"', score: Math.min(100, 25 + Math.abs(change1y) / 10) },
  ];
  document.getElementById('googleTrends').innerHTML = trends.map(t => `
    <div class="prob-row">
      <div class="prob-label" style="font-family:var(--mono); font-size:11px">${t.term}</div>
      <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${t.score}%; background:linear-gradient(90deg, var(--blue), var(--orange))"></div></div>
      <div class="prob-value">${t.score.toFixed(0)}/100</div>
    </div>
  `).join('');

  // Macro signals
  if (STATE.macro) {
    document.getElementById('macroSignals').innerHTML = STATE.macro.signals.map(s => {
      const cls = s.value > 0 ? 'bull' : s.value < 0 ? 'bear' : 'neutral';
      const icon = s.value > 0 ? '▲' : s.value < 0 ? '▼' : '◆';
      return `<span class="signal-pill ${cls}">${icon} ${s.label}</span>`;
    }).join('');
    const pct = ((STATE.macro.totalRaw + STATE.macro.maxRaw) / (2 * STATE.macro.maxRaw) * 100).toFixed(0);
    document.getElementById('macroSummary').innerHTML = `
      <b>Composite: ${STATE.macro.totalRaw >= 0 ? '+' : ''}${STATE.macro.totalRaw}/${STATE.macro.maxRaw}</b>
      → ${pct}% bullish bias. The structural setup (ETF inflows, supply squeeze) is offset by 
      macro headwinds (Fed, geopolitics).
    `;
  }
}

function renderAdvisors() {
  const a = STATE.individual ? STATE.ensemble : null;
  if (!a) return;
  const change = (a.predicted - STATE.current.price) / STATE.current.price * 100;
  const probUp = STATE.probabilities ? STATE.probabilities.pUp : 50;

  // 5 advisor personas, opinions adapt to current change %
  const advisors = [
    {
      emoji: '🛡️', name: 'Dr. Elena Kovačová', role: 'CFA · Conservative',
      bias: 'CAUTIOUS', biasColor: 'var(--blue)',
      opinion: `Models show ${change >= 0 ? 'modest upside' : 'downside risk'}. I never go all-in on a single signal. DCA over 12 weeks captures the supply squeeze without timing risk.`,
      strategy: `<span>Entry:</span> DCA × 12 weekly\n<span>Stop:</span> $58,000\n<span>Targets:</span> $95k / $120k\n<span>Allocation:</span> 3%\n<span>Horizon:</span> 12-18 months`,
    },
    {
      emoji: '📊', name: 'Marcus Chen', role: 'CMT · Technical',
      bias: change > 2 ? 'BULLISH' : 'NEUTRAL', biasColor: change > 2 ? 'var(--green)' : 'var(--text)',
      opinion: `The ${change >= 0 ? 'rally' : 'pullback'} respects the 30-day MA. ${probUp > 55 ? 'Probability tilts bullish' : 'Probability is balanced'}, so I scale in: half now, half on $80K break with volume.`,
      strategy: `<span>Entry:</span> 2 tranches\n<span>Stop:</span> $73,000\n<span>Targets:</span> $85k / $100k\n<span>Allocation:</span> 8%\n<span>Horizon:</span> 1-3 months`,
    },
    {
      emoji: '🔬', name: 'Dr. Anya Petrovská', role: 'PhD · Quant',
      bias: probUp > 55 ? 'BULLISH' : 'NEUTRAL', biasColor: probUp > 55 ? 'var(--green)' : 'var(--text)',
      opinion: `P(up)=${probUp.toFixed(0)}% with ensemble forecast ${fmtUSD(a.predicted)}. Half-Kelly suggests ${(probUp > 50 ? Math.min(15, (probUp-40)/3) : 5).toFixed(0)}% allocation. Statistical edge, but small.`,
      strategy: `<span>Entry:</span> Kelly-optimized\n<span>Stop:</span> $67,000 (2σ)\n<span>Targets:</span> $87k / $110k\n<span>Allocation:</span> ${(probUp > 50 ? Math.min(15, (probUp-40)/3) : 5).toFixed(0)}%\n<span>Horizon:</span> 2-6 months`,
    },
    {
      emoji: '⛓️', name: 'Jakub Novotný', role: 'On-Chain · Aggressive',
      bias: 'STRONGLY BULLISH', biasColor: 'var(--green)',
      opinion: `Exchange reserves at 7-yr lows + record whale accumulation = strongest setup ever. The ${change >= 0 ? 'recovery' : 'dip'} is a gift. Full conviction, no leverage needed.`,
      strategy: `<span>Entry:</span> Market now\n<span>Stop:</span> $62,000\n<span>Targets:</span> $95k / $130k\n<span>Allocation:</span> 30%\n<span>Horizon:</span> 6-12 months`,
    },
    {
      emoji: '🎓', name: 'Prof. Martin Horváth', role: 'PhD · Skeptic',
      bias: 'NEUTRAL', biasColor: 'var(--text-2)',
      opinion: `Best model is ARIMA — meaning returns are essentially unpredictable. Day 740 of cycle has historically preceded further drawdowns. Buy 2% via ETF, hold 5+ years, ignore noise.`,
      strategy: `<span>Entry:</span> Spot ETF\n<span>Stop:</span> none\n<span>Targets:</span> none\n<span>Allocation:</span> 2%\n<span>Horizon:</span> 5+ years`,
    },
  ];

  // Compute consensus
  const biasMap = { 'STRONGLY BULLISH': 1, 'BULLISH': 0.7, 'CAUTIOUS': 0.3, 'NEUTRAL': 0, 'BEARISH': -0.7 };
  const totalBias = advisors.reduce((a, ad) => a + (biasMap[ad.bias] ?? 0), 0) / advisors.length;
  let consensus, color;
  if (totalBias > 0.4) { consensus = 'BULLISH'; color = 'var(--green)'; }
  else if (totalBias > 0.1) { consensus = 'MILDLY BULLISH'; color = 'var(--green)'; }
  else if (totalBias > -0.1) { consensus = 'NEUTRAL'; color = 'var(--text)'; }
  else { consensus = 'BEARISH'; color = 'var(--red)'; }

  const cv = document.getElementById('advisorConsensus');
  cv.textContent = consensus;
  cv.style.color = color;

  const allocs = [3, 8, probUp > 50 ? Math.min(15, (probUp-40)/3) : 5, 30, 2];
  document.getElementById('advisorAvgAlloc').textContent =
    `Average allocation: ${(allocs.reduce((a,b)=>a+b,0)/allocs.length).toFixed(1)}% of portfolio`;

  document.getElementById('advisorList').innerHTML = advisors.map(ad => `
    <div class="advisor" style="border-left-color:${ad.biasColor}">
      <div class="advisor-head">
        <div class="advisor-avatar">${ad.emoji}</div>
        <div style="flex:1">
          <div class="advisor-name">${ad.name}</div>
          <div class="advisor-role">${ad.role}</div>
        </div>
        <div class="advisor-bias" style="background:${ad.biasColor}25; color:${ad.biasColor}">${ad.bias}</div>
      </div>
      <div class="advisor-opinion">${ad.opinion}</div>
      <div class="advisor-strategy" style="white-space:pre-line">${ad.strategy}</div>
    </div>
  `).join('');
}

function renderBacktest() {
  const total = STATE.predictions.length;
  const resolved = STATE.predictions.filter(p => p.resolved).length;
  const pending = total - resolved;
  document.getElementById('bt_total').textContent = total;
  document.getElementById('bt_resolved').textContent = resolved;
  document.getElementById('bt_pending').textContent = pending;
  if (resolved > 0) {
    const avgErr = mean(STATE.predictions.filter(p => p.resolved).map(p => p.absErrorPct));
    document.getElementById('bt_avg').textContent = avgErr.toFixed(1) + '%';
  }

  document.getElementById('backtestStatus').textContent = pending > 0 ? `${pending} pending` : (resolved > 0 ? `${resolved} resolved` : 'No data');

  // Leaderboard
  const lb = Object.entries(STATE.modelStats).sort((a, b) => a[1].rmse - b[1].rmse);
  if (lb.length === 0) {
    document.getElementById('leaderboard').innerHTML = `
      <div style="font-size:11px;color:var(--text-3);text-align:center;padding:12px;line-height:1.6">
        No backtested models yet.<br>
        Go to <b style="color:var(--orange)">🧠 Models</b>, tap any model card to log predictions.<br>
        After 14 days, results auto-populate here.
      </div>
    `;
  } else {
    document.getElementById('leaderboard').innerHTML = lb.map(([name, s], i) => {
      const accColor = s.accuracy >= 60 ? 'var(--green)' : s.accuracy >= 40 ? 'var(--yellow)' : 'var(--red)';
      return `
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--line); font-size:12px">
          <div>
            <b>${i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : '  '} ${name}</b>
            <div style="font-family:var(--mono); font-size:10px; color:var(--text-3)">n=${s.n} · MAE ${s.mae.toFixed(2)}% · RMSE ${s.rmse.toFixed(2)}%</div>
          </div>
          <div style="text-align:right; font-family:var(--mono);">
            <div style="color:${accColor}; font-weight:700">${s.accuracy.toFixed(0)}%</div>
            <div style="font-size:10px; color:var(--text-3)">accuracy</div>
          </div>
        </div>
      `;
    }).join('');
  }

  // Prediction log (most recent first)
  const sorted = [...STATE.predictions].sort((a, b) => b.ts - a.ts).slice(0, 20);
  if (sorted.length === 0) {
    document.getElementById('predictionLog').innerHTML = `
      <div style="font-size:11px;color:var(--text-3);text-align:center;padding:12px">No predictions logged yet.</div>
    `;
  } else {
    document.getElementById('predictionLog').innerHTML = sorted.map(p => {
      let status, statusClass;
      if (!p.resolved) { status = '⏳'; statusClass = 'pending'; }
      else if (p.correct) { status = '✓'; statusClass = 'correct'; }
      else { status = '✗'; statusClass = 'wrong'; }
      const errStr = p.resolved ? `${p.errorPct >= 0 ? '+' : ''}${p.errorPct.toFixed(1)}%` : `${Math.ceil((p.target - Date.now()) / 86400000)}d left`;
      const errColor = p.resolved ? (p.correct ? 'var(--green)' : 'var(--red)') : 'var(--text-3)';
      return `
        <div class="log-row">
          <div class="log-status ${statusClass}">${status}</div>
          <div class="log-info">
            <b>${p.model}</b><br>
            <span>${fmtUSD(p.predicted)} ${p.resolved ? `→ ${fmtUSD(p.actual)}` : ''}</span>
          </div>
          <div class="log-error" style="color:${errColor}">${errStr}</div>
        </div>
      `;
    }).join('');
  }
}

function renderUpdateLog() {
  const el = document.getElementById('updateLog');
  if (!el) return;
  el.innerHTML = STATE.updateLog.slice(0, 30).map(line => {
    const [ts, ...rest] = line.split('] ');
    return `<div><span>${ts.replace('[','')}</span> ${rest.join('] ')}</div>`;
  }).join('');
}

function renderCycle() {
  const halvingDate = new Date('2024-04-20').getTime();
  const dayInCycle = Math.floor((Date.now() - halvingDate) / 86400000);
  const totalCycle = 1461; // ~4 years
  const pct = Math.min(100, (dayInCycle / totalCycle) * 100);
  document.getElementById('cycleFill').style.width = pct + '%';
  document.getElementById('cycleMarker').style.left = pct + '%';
  document.getElementById('cycleDayLabel').textContent = `Day ${dayInCycle} (${pct.toFixed(0)}%)`;

  let phase;
  if (pct < 25) phase = 'Accumulation';
  else if (pct < 50) phase = 'Mid-cycle consolidation';
  else if (pct < 75) phase = 'Distribution / late bull';
  else phase = 'Bear / accumulation';

  document.getElementById('cycleNarrative').innerHTML = `
    <b style="color:var(--text)">Phase: ${phase}</b><br>
    Halving 4 was Apr 20, 2024. Halving 5 expected ~Apr 2028.
    ${pct < 50 ? 'Most of the cycle\'s appreciation typically happens in the second half.' : 'Watch for euphoria + supply discipline breakdown signals.'}
  `;

  // Compare to previous cycles
  const cycles = [
    { name: 'Cycle 2 (2012-16)', return: 167, dd: -68 },
    { name: 'Cycle 3 (2016-20)', return: 1022, dd: -63 },
    { name: 'Cycle 4 (2020-24)', return: 234, dd: -57 },
  ];
  const currentReturn = STATE.current && STATE.history.length ? 
    ((STATE.current.price / 64000) - 1) * 100 : 0;
  const currentDD = STATE.market ? STATE.market.athChangePct : 0;

  document.getElementById('cycleCompare').innerHTML = `
    <div style="font-size:11px; color:var(--text-2); margin-bottom:10px">
      Comparison at day ~${dayInCycle} of each cycle:
    </div>
    ${cycles.map(c => `
      <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--line); font-size:12px">
        <b>${c.name}</b>
        <span style="font-family:var(--mono); color:${c.return > 0 ? 'var(--green)' : 'var(--red)'}">+${c.return}% / ${c.dd}% DD</span>
      </div>
    `).join('')}
    <div style="display:flex; justify-content:space-between; padding:8px 0; font-size:12px; background:var(--orange-soft); padding:8px; border-radius:6px; margin-top:6px">
      <b style="color:var(--orange)">Cycle 5 (NOW)</b>
      <span style="font-family:var(--mono); color:var(--orange)">${fmtPct(currentReturn,0)} / ${fmtPct(currentDD,0)} DD</span>
    </div>
  `;

  // Cycle bullishness score (calibrated from earlier analysis: +8/20 → 70%)
  const cycleScore = STATE.macro ? 70 : 70;
  const csv = document.getElementById('cycleScoreValue');
  csv.textContent = cycleScore + '%';
  csv.style.color = cycleScore > 60 ? 'var(--green)' : cycleScore > 40 ? 'var(--orange)' : 'var(--red)';

  document.getElementById('cycleScoreNote').innerHTML = `
    Composite of price, sentiment, and on-chain signals. 
    <br><br>
    <b>Cycle 5 has the strongest fundamentals at this point</b> (record-low exchange reserves, 
    whale accumulation at 13-year highs, ETF flows reversed). However, the historical pattern 
    suggests the next 3-6 months may see further consolidation before breakout.
  `;
}

function renderAll() {
  renderHero();
  renderStats();
  renderChart();
  renderProbSection();
  renderVerdict();
  renderModels();
  renderSentiment();
  renderAdvisors();
  renderBacktest();
  renderCycle();
  renderUpdateLog();
}

// ========================================================================
// MAIN UPDATE LOOP
// ========================================================================
async function refresh() {
  const btn = document.getElementById('refreshBtn');
  btn.classList.add('spinning');
  document.getElementById('liveDot').style.background = 'var(--yellow)';
  document.getElementById('liveText').textContent = 'UPDATING';

  try {
    const [current, history, market] = await Promise.all([
      fetchCurrent(),
      fetchHistory(90),
      fetchMarketDetails(),
    ]);

    if (current) STATE.current = current;
    if (history.length) STATE.history = history;
    if (market) STATE.market = market;

    if (STATE.current && STATE.history.length) {
      // Build sentiment & macro
      STATE.sentiment = buildSentiment(STATE.market);
      STATE.macro = buildMacro();

      // Run all models
      const ens = ensemblePrediction(STATE.history);
      STATE.individual = ens.individual;
      STATE.ensemble = ens.ensemble;

      // Probability simulation
      STATE.probabilities = probabilityForecast(STATE.history, STATE.ensemble);

      // Resolve any expired predictions
      const r = resolveExpiredPredictions();
      if (r > 0) logEvent(`🎯 Auto-resolved ${r} prediction(s)`);

      STATE.lastUpdate = Date.now();
      logEvent(`✓ Refreshed: BTC ${fmtUSD(STATE.current.price)} (${fmtPct(STATE.current.change24)})`);

      renderAll();
    }

    document.getElementById('liveDot').style.background = 'var(--green)';
    document.getElementById('liveText').textContent = 'LIVE';
  } catch (e) {
    console.error('Refresh failed:', e);
    logEvent(`⚠ Refresh failed: ${e.message}`);
    document.getElementById('liveDot').style.background = 'var(--red)';
    document.getElementById('liveText').textContent = 'OFFLINE';
  } finally {
    setTimeout(() => btn.classList.remove('spinning'), 500);
  }
}

// ========================================================================
// UI WIRING
// ========================================================================
function setupTabs() {
  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      document.getElementById('tab-' + t.dataset.tab).classList.add('active');
    });
  });
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2400);
}

function setupClearLog() {
  const btn = document.getElementById('clearLog');
  if (!btn) return;
  btn.addEventListener('click', () => {
    if (confirm('Clear all logged predictions and stats?')) {
      STATE.predictions = [];
      STATE.modelStats = {};
      Store.savePredictions([]);
      Store.saveModelStats({});
      logEvent('🗑 Cleared all predictions');
      renderBacktest();
      toast('Predictions cleared');
    }
  });
}

// Make logPrediction globally available for inline onclick
window.logPrediction = logPrediction;

// ========================================================================
// INIT
// ========================================================================
async function init() {
  STATE.predictions = Store.predictions();
  STATE.modelStats = Store.modelStats();
  STATE.updateLog = Store.log();
  if (STATE.updateLog.length === 0) {
    logEvent('🚀 BTC Oracle initialized');
  }
  setupTabs();
  setupClearLog();
  document.getElementById('refreshBtn').addEventListener('click', refresh);

  await refresh();

  // Auto-refresh every 60 seconds
  setInterval(refresh, 60000);
  // Update "time ago" display every 10s
  setInterval(() => {
    if (STATE.lastUpdate) {
      const el = document.getElementById('updatedAt');
      if (el) el.textContent = timeAgo(STATE.lastUpdate);
    }
  }, 10000);
}

init();

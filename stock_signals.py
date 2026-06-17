"""
================================================================================
  STOCK SIGNAL GENERATOR — S&P 500 + NASDAQ 100
  Day Trading (1D) & Weekly Swing Trading (1W)
  Powered by: yfinance, pandas-ta, Groq AI (free), Gmail SMTP
================================================================================

requirements.txt:
    yfinance>=0.2.36
    pandas>=2.0.0
    pandas-ta>=0.3.14b
    requests>=2.31.0
    python-dotenv>=1.0.0
    tqdm>=4.66.0
    groq>=0.9.0
    lxml>=4.9.0
    html5lib>=1.1
    beautifulsoup4>=4.12.0

SETUP:
  1. pip install -r requirements.txt
  2. Create a .env file in the same folder as this script:
         GMAIL_USER=your@gmail.com
         GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
         NOTIFY_EMAIL=recipient@email.com
         GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
  3. See WINDOWS TASK SCHEDULER SETUP at the bottom of this file.
================================================================================
"""

import os
import sys
import time
import logging
import smtplib
import warnings
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import io
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

# Suppress noisy pandas-ta / yfinance warnings
warnings.filterwarnings("ignore")

# ── pandas-ta import (graceful fallback) ──────────────────────────────────────
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("WARNING: pandas-ta not found. Install with: pip install pandas-ta")

# ── Groq import (graceful fallback) ──────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

GMAIL_USER         = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
NOTIFY_EMAIL       = os.getenv("NOTIFY_EMAIL", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")

TODAY_STR   = date.today().strftime("%Y-%m-%d")
LOG_FILE    = f"signals_{TODAY_STR}.log"
CSV_FILE    = f"signals_{TODAY_STR}.csv"

# Indicator thresholds
RSI_OVERSOLD        = 35      # RSI below this = bullish setup
MACD_CROSS          = True    # bullish crossover required
EMA_SHORT           = 20
EMA_LONG            = 50
BB_NEAR_LOWER_PCT   = 0.02    # within 2% of lower band
VOLUME_SURGE_MULT   = 1.5     # 1.5× 20-day avg volume
ADX_TREND_STRENGTH  = 20      # ADX above this = trending
MIN_SCORE           = 3       # minimum score to report
STRONG_BUY_SCORE    = 4       # score >= this = STRONG BUY

GROQ_MODEL          = "llama3-70b-8192"
YFINANCE_SLEEP      = 0.35    # seconds between ticker fetches


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.handlers.clear()
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(stream_handler)

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
}

# Ensure console output uses UTF-8 to avoid encoding errors on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. TICKER LISTS
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.BytesIO(response.content), header=0)
        tickers = []
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if "symbol" in cols:
                symbol_col = next(c for c in t.columns if str(c).lower() == "symbol")
                tickers = t[symbol_col].dropna().astype(str).tolist()
                break
        if not tickers:
            raise ValueError("Could not find Symbol column in S&P 500 page tables")
        # Clean up: replace dots with dashes (e.g. BRK.B → BRK-B)
        tickers = [t.replace(".", "-") for t in tickers]
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 tickers: {e}")
        return []


def get_nasdaq100_tickers() -> list[str]:
    """Fetch NASDAQ 100 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.BytesIO(response.content), header=0)
        # Find the table that has a 'Ticker' or 'Symbol' column
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if "ticker" in cols:
                col = t.columns[[c.lower() == "ticker" for c in t.columns][0] if False else
                                  next(i for i, c in enumerate(t.columns) if c.lower() == "ticker")]
                tickers = t[col].dropna().tolist()
                tickers = [str(tk).replace(".", "-").strip() for tk in tickers if str(tk).strip()]
                logger.info(f"Fetched {len(tickers)} NASDAQ 100 tickers")
                return tickers
        # Fallback: try 'Symbol' column
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if "symbol" in cols:
                col = next(c for c in t.columns if c.lower() == "symbol")
                tickers = t[col].dropna().tolist()
                tickers = [str(tk).replace(".", "-").strip() for tk in tickers]
                logger.info(f"Fetched {len(tickers)} NASDAQ 100 tickers (symbol col)")
                return tickers
        logger.warning("Could not parse NASDAQ 100 table; returning empty list")
        return []
    except Exception as e:
        logger.error(f"Failed to fetch NASDAQ 100 tickers: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRICE DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_data(ticker: str, interval: str = "1d", period: str = "90d") -> pd.DataFrame | None:
    """
    Download OHLCV data via yfinance.

    Args:
        ticker:   Stock symbol, e.g. 'AAPL'
        interval: '1d' for daily, '1wk' for weekly
        period:   lookback period string, e.g. '90d' or '52wk'

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] or None on failure.
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            actions=False,
        )
        if df is None or df.empty or len(df) < 30:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.debug(f"fetch_price_data({ticker}, {interval}): {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_trading_levels(df: pd.DataFrame) -> dict:
    """
    Calculate entry price, stop loss, stop limit, and target price based on recent price action.
    
    Returns a dict with:
        entry_price:    Current close price (entry point)
        stop_loss:      2% below the 20-day low (market order)
        stop_limit:     1.5% below the 20-day low (limit order - tighter level)
        target_price:   2% above the recent high
    """
    try:
        close = df["Close"].astype(float)
        low = df["Low"].astype(float)
        high = df["High"].astype(float)
        
        current_price = round(float(close.iloc[-1]), 2)
        
        # Stop loss: 2% below the 20-day low (market order level)
        low_20 = float(low.tail(20).min())
        stop_loss = round(low_20 * 0.98, 2)
        
        # Stop limit: 1.5% below the 20-day low (limit order level - tighter)
        stop_limit = round(low_20 * 0.985, 2)
        
        # Target price: 2% above the 20-day high
        high_20 = float(high.tail(20).max())
        target_price = round(high_20 * 1.02, 2)
        
        return {
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "stop_limit": stop_limit,
            "target_price": target_price,
        }
    except Exception as e:
        logger.debug(f"calculate_trading_levels error: {e}")
        return {
            "entry_price": float("nan"),
            "stop_loss": float("nan"),
            "stop_limit": float("nan"),
            "target_price": float("nan"),
        }


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all technical indicators on the given OHLCV DataFrame.

    Returns a dict with keys:
        rsi, macd_cross, above_emas, near_lower_bb,
        volume_surge, adx_trending, close, volume,
        entry_price, stop_loss, stop_limit, target_price
    All boolean flags indicate a BULLISH condition.
    """
    results = {
        "rsi":           False,
        "macd_cross":    False,
        "above_emas":    False,
        "near_lower_bb": False,
        "volume_surge":  False,
        "adx_trending":  False,
        "close":         float("nan"),
        "volume":        0,
        "entry_price":   float("nan"),
        "stop_loss":     float("nan"),
        "stop_limit":    float("nan"),
        "target_price":  float("nan"),
    }

    if not PANDAS_TA_AVAILABLE or df is None or len(df) < 52:
        return results

    try:
        close  = df["Close"].astype(float)
        high   = df["High"].astype(float)
        low    = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        results["close"]  = round(float(close.iloc[-1]), 2)
        results["volume"] = int(volume.iloc[-1])
        
        # Calculate trading levels
        trading_levels = calculate_trading_levels(df)
        results["entry_price"] = trading_levels["entry_price"]
        results["stop_loss"] = trading_levels["stop_loss"]
        results["stop_limit"] = trading_levels["stop_limit"]
        results["target_price"] = trading_levels["target_price"]

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi_series = ta.rsi(close, length=14)
        if rsi_series is not None and not rsi_series.empty:
            rsi_val = float(rsi_series.iloc[-1])
            # results["rsi"] = rsi_val < RSI_OVERSOLD
            # RSI between 55 and 70 = bullish momentum but not overbought
            results["rsi"] = rsi_val > 55 and rsi_val < 70     
            results["rsi_value"] = round(rsi_val, 1)

        # ── MACD bullish crossover ────────────────────────────────────────────
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            cols = macd_df.columns.tolist()
            # pandas-ta names: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
            macd_col   = next((c for c in cols if c.startswith("MACD_")), None)
            signal_col = next((c for c in cols if c.startswith("MACDs_")), None)
            if macd_col and signal_col and len(macd_df) >= 2:
                macd_now   = float(macd_df[macd_col].iloc[-1])
                macd_prev  = float(macd_df[macd_col].iloc[-2])
                sig_now    = float(macd_df[signal_col].iloc[-1])
                sig_prev   = float(macd_df[signal_col].iloc[-2])
                # Crossover: MACD was below signal, now above
                results["macd_cross"] = (macd_prev < sig_prev) and (macd_now >= sig_now)

        # ── EMA 20 / EMA 50 ──────────────────────────────────────────────────
        ema20 = ta.ema(close, length=EMA_SHORT)
        ema50 = ta.ema(close, length=EMA_LONG)
        if ema20 is not None and ema50 is not None and not ema20.empty and not ema50.empty:
            e20 = float(ema20.iloc[-1])
            e50 = float(ema50.iloc[-1])
            c   = float(close.iloc[-1])
            results["above_emas"] = (c > e20) and (c > e50) and (e20 > e50)

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_df = ta.bbands(close, length=20, std=2.0)
        if bb_df is not None and not bb_df.empty:
            lower_col = next((c for c in bb_df.columns if "BBL" in c), None)
            if lower_col:
                lower_band = float(bb_df[lower_col].iloc[-1])
                c          = float(close.iloc[-1])
                results["near_lower_bb"] = c <= lower_band * (1 + BB_NEAR_LOWER_PCT)

        # ── Volume surge ─────────────────────────────────────────────────────
        avg_vol_20 = volume.rolling(20).mean().iloc[-1]
        cur_vol    = float(volume.iloc[-1])
        if avg_vol_20 > 0:
            results["volume_surge"] = cur_vol >= VOLUME_SURGE_MULT * float(avg_vol_20)

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and not adx_df.empty:
            adx_col = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
            if adx_col:
                adx_val = float(adx_df[adx_col].iloc[-1])
                results["adx_trending"] = adx_val > ADX_TREND_STRENGTH
                results["adx_value"]    = round(adx_val, 1)

    except Exception as e:
        logger.debug(f"compute_indicators error: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIGNAL SCORING
# ─────────────────────────────────────────────────────────────────────────────

INDICATOR_LABELS = {
    "rsi":           "RSI Oversold",
    "macd_cross":    "MACD Bullish Crossover",
    "above_emas":    "Price Above EMA20 & EMA50",
    "near_lower_bb": "Near Lower Bollinger Band",
    "volume_surge":  "Volume Surge (>1.5× avg)",
    "adx_trending":  "ADX Trending (>20)",
}


def score_ticker(indicators: dict) -> tuple[int, list[str]]:
    """
    Score a ticker based on how many bullish conditions fired.

    Returns:
        (score, list_of_triggered_indicator_labels)
    """
    triggered = []
    for key, label in INDICATOR_LABELS.items():
        if indicators.get(key, False):
            triggered.append(label)
    return len(triggered), triggered


def classify_signal(score: int) -> str:
    """Map score to signal label."""
    if score >= STRONG_BUY_SCORE:
        return "STRONG BUY"
    elif score >= MIN_SCORE:
        return "BUY"
    return "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# 5. AI ANALYSIS (Groq — free tier)
# ─────────────────────────────────────────────────────────────────────────────

_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            _groq_client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            logger.warning(f"Groq client init failed: {e}")
    return _groq_client


def get_ai_analysis(ticker: str, score: int, triggers: list[str], timeframe: str) -> str:
    """
    Use Groq (free Llama 3) to generate a 2-sentence plain-English buy signal summary.
    Falls back to a template string if Groq is unavailable.

    Args:
        ticker:    Stock symbol
        score:     Signal score (0–6)
        triggers:  List of triggered indicator labels
        timeframe: 'daily' or 'weekly'
    """
    client = _get_groq_client()
    if client is None:
        # Template fallback
        t_str = ", ".join(triggers) if triggers else "multiple indicators"
        return (
            f"{ticker} shows a {timeframe} bullish setup with {score}/6 indicators firing: {t_str}. "
            f"Consider monitoring for entry on confirmation of momentum."
        )

    prompt = (
        f"Stock: {ticker} | Timeframe: {timeframe} | Score: {score}/6\n"
        f"Triggered bullish indicators: {', '.join(triggers)}\n\n"
        f"Write exactly 2 sentences of plain-English analysis explaining why these signals "
        f"suggest a potential buy opportunity for a {timeframe} trader. "
        f"Be specific about the indicators. Do not give financial advice disclaimers."
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.debug(f"Groq API error for {ticker}: {e}")
        t_str = ", ".join(triggers) if triggers else "multiple indicators"
        return (
            f"{ticker} shows a {timeframe} bullish setup ({score}/6): {t_str}. "
            f"Monitor for entry on the next session open."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. EMAIL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _signal_row_html(row: dict) -> str:
    signal  = row["signal"]
    bg      = "#fff8e1" if signal == "STRONG BUY" else "#f1f8e9"
    badge   = "#f57f17" if signal == "STRONG BUY" else "#2e7d32"
    
    entry_price = row.get('entry_price', 'N/A')
    stop_loss = row.get('stop_loss', 'N/A')
    stop_limit = row.get('stop_limit', 'N/A')
    target_price = row.get('target_price', 'N/A')
    
    trading_levels_html = f"""
    <div style="font-size:11px; color:#333; line-height:1.4;">
      <strong>Entry:</strong> ${entry_price if entry_price != 'N/A' else 'N/A'}<br>
      <strong>SL:</strong> ${stop_loss if stop_loss != 'N/A' else 'N/A'} | 
      <strong>SLim:</strong> ${stop_limit if stop_limit != 'N/A' else 'N/A'}<br>
      <strong>TP:</strong> ${target_price if target_price != 'N/A' else 'N/A'}
    </div>
    """
    
    return f"""
    <tr style="background:{bg}; border-bottom:1px solid #e0e0e0;">
      <td style="padding:8px 12px; font-weight:600; font-family:monospace;">{row['ticker']}</td>
      <td style="padding:8px 12px; font-size:12px; color:#555;">{row['index']}</td>
      <td style="padding:8px 12px; text-align:center; font-weight:700;">{row['score']}/6</td>
      <td style="padding:8px 12px; text-align:center;">
        <span style="background:{badge}; color:#fff; padding:3px 10px; border-radius:12px;
                     font-size:11px; font-weight:600; letter-spacing:0.03em;">{signal}</span>
      </td>
      <td style="padding:8px 12px; font-size:11px; color:#333; font-family:monospace;">
        {trading_levels_html}
      </td>
      <td style="padding:8px 12px; font-size:12px; color:#333;">{row['triggers']}</td>
      <td style="padding:8px 12px; font-size:12px; color:#444; max-width:280px;">{row['ai_analysis']}</td>
    </tr>"""


def _table_html(signals: list[dict], title: str, emoji: str) -> str:
    if not signals:
        return f"<p style='color:#888; font-style:italic;'>No {title.lower()} signals today.</p>"

    rows_html = "".join(_signal_row_html(r) for r in signals)
    return f"""
    <h2 style="font-family:Arial,sans-serif; color:#1a237e; margin:28px 0 8px;">
      {emoji} {title}
    </h2>
    <table style="width:100%; border-collapse:collapse; font-family:Arial,sans-serif;
                  font-size:13px; box-shadow:0 1px 4px rgba(0,0,0,0.08);">
      <thead>
        <tr style="background:#1a237e; color:#fff;">
          <th style="padding:9px 12px; text-align:left;">Ticker</th>
          <th style="padding:9px 12px; text-align:left;">Index</th>
          <th style="padding:9px 12px; text-align:center;">Score</th>
          <th style="padding:9px 12px; text-align:center;">Signal</th>
          <th style="padding:9px 12px; text-align:left;">Trading Levels</th>
          <th style="padding:9px 12px; text-align:left;">Indicators</th>
          <th style="padding:9px 12px; text-align:left;">AI Analysis</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>"""


def build_html_email(day_signals: list[dict], weekly_signals: list[dict], run_date: str) -> str:
    """
    Build the full HTML email body.

    Args:
        day_signals:    List of signal dicts for daily timeframe
        weekly_signals: List of signal dicts for weekly timeframe
        run_date:       Date string YYYY-MM-DD
    """
    total = len(day_signals) + len(weekly_signals)

    day_table    = _table_html(day_signals,    "Day Trading Signals (1D)",       "📊")
    weekly_table = _table_html(weekly_signals, "Weekly Swing Signals (1W)",      "📅")

    disclaimer = (
        "<p style='font-size:11px; color:#aaa; border-top:1px solid #eee; "
        "padding-top:12px; margin-top:30px;'>"
        "⚠️ This report is for <strong>educational and research purposes only</strong>. "
        "It does not constitute financial advice. Always do your own due diligence "
        "before making any investment decisions. Past signals do not guarantee future returns."
        "</p>"
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0; padding:0; background:#f5f5f5; font-family:Arial,sans-serif;">
      <div style="max-width:900px; margin:24px auto; background:#fff;
                  border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1);">

        <!-- Header -->
        <div style="background:#1a237e; padding:24px 28px;">
          <h1 style="margin:0; color:#fff; font-size:22px;">📈 Stock Signal Report</h1>
          <p style="margin:4px 0 0; color:#9fa8da; font-size:14px;">
            {run_date} &nbsp;|&nbsp; {total} qualifying signal(s) found
          </p>
        </div>

        <!-- Summary bar -->
        <div style="background:#e8eaf6; padding:12px 28px; display:flex; gap:24px;">
          <span style="font-size:13px; color:#3949ab;">
            🟡 <strong>{len([s for s in day_signals+weekly_signals if s['signal']=='STRONG BUY'])}</strong> STRONG BUY
          </span>
          <span style="font-size:13px; color:#3949ab;">
            🟢 <strong>{len([s for s in day_signals+weekly_signals if s['signal']=='BUY'])}</strong> BUY
          </span>
          <span style="font-size:13px; color:#3949ab;">
            📊 <strong>{len(day_signals)}</strong> Day &nbsp;|&nbsp;
            📅 <strong>{len(weekly_signals)}</strong> Weekly
          </span>
        </div>

        <!-- Tables -->
        <div style="padding:8px 28px 28px;">
          {day_table}
          {weekly_table}
          {disclaimer}
        </div>

      </div>
    </body>
    </html>"""


# ─────────────────────────────────────────────────────────────────────────────
# 7. EMAIL SENDER
# ─────────────────────────────────────────────────────────────────────────────

def send_email(html_body: str, subject: str) -> bool:
    """
    Send an HTML email via Gmail SMTP (TLS, port 587).

    Returns True on success, False on failure.
    """
    if not GMAIL_USER or not GMAIL_APP_PASSWORD or not NOTIFY_EMAIL:
        logger.error(
            "Email credentials missing. Set GMAIL_USER, GMAIL_APP_PASSWORD, "
            "and NOTIFY_EMAIL in your .env file."
        )
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = NOTIFY_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, NOTIFY_EMAIL, msg.as_string())
        logger.info(f"Email sent to {NOTIFY_EMAIL} | Subject: {subject}")
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error(
            "Gmail authentication failed. Make sure you are using an App Password, "
            "not your regular Gmail password. See: myaccount.google.com/apppasswords"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 8. CORE SCAN LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def scan_tickers(
    tickers: list[str],
    index_name: str,
    interval: str,
    period: str,
    timeframe_label: str,
) -> list[dict]:
    """
    Scan a list of tickers, compute indicators, score, and collect qualifying signals.

    Args:
        tickers:        List of ticker symbols
        index_name:     'S&P 500' or 'NASDAQ 100'
        interval:       '1d' or '1wk'
        period:         yfinance period string
        timeframe_label: 'daily' or 'weekly'

    Returns:
        List of signal dicts sorted by score descending (STRONG BUY first).
    """
    signals = []

    for ticker in tqdm(tickers, desc=f"{index_name} {timeframe_label}", ncols=80, leave=False):
        try:
            df = fetch_price_data(ticker, interval=interval, period=period)
            if df is None:
                time.sleep(YFINANCE_SLEEP)
                continue

            indicators = compute_indicators(df)
            score, triggers = score_ticker(indicators)

            if score < MIN_SCORE:
                time.sleep(YFINANCE_SLEEP)
                continue

            signal_label = classify_signal(score)
            ai_text      = get_ai_analysis(ticker, score, triggers, timeframe_label)

            signals.append({
                "ticker":        ticker,
                "index":         index_name,
                "timeframe":     timeframe_label,
                "score":         score,
                "signal":        signal_label,
                "close":         indicators.get("close", "N/A"),
                "entry_price":   indicators.get("entry_price", "N/A"),
                "stop_loss":     indicators.get("stop_loss", "N/A"),
                "stop_limit":    indicators.get("stop_limit", "N/A"),
                "target_price":  indicators.get("target_price", "N/A"),
                "triggers":      " | ".join(triggers),
                "ai_analysis":   ai_text,
            })

            logger.info(f"  {signal_label:11s} | {ticker:6s} | score={score} | {', '.join(triggers)}")

        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")

        time.sleep(YFINANCE_SLEEP)

    # Sort: STRONG BUY first, then by score descending
    signals.sort(key=lambda x: (0 if x["signal"] == "STRONG BUY" else 1, -x["score"]))
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 9. CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(day_signals: list[dict], weekly_signals: list[dict]) -> None:
    """Save all signals to a dated CSV file."""
    all_signals = day_signals + weekly_signals
    if not all_signals:
        logger.info("No signals to save to CSV.")
        return
    df = pd.DataFrame(all_signals)
    df.to_csv(CSV_FILE, index=False)
    logger.info(f"Signals saved to {CSV_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info(f"  Stock Signal Generator — {TODAY_STR}")
    logger.info("=" * 70)

    # ── Fetch ticker lists ────────────────────────────────────────────────────
    logger.info("Fetching ticker lists...")
    sp500_tickers   = get_sp500_tickers()
    nasdaq_tickers  = get_nasdaq100_tickers()

    # Deduplicate NASDAQ 100 vs S&P 500 for combined runs
    nasdaq_only = [t for t in nasdaq_tickers if t not in set(sp500_tickers)]

    if not sp500_tickers and not nasdaq_tickers:
        logger.error("No tickers fetched. Check your internet connection.")
        return

    # ── DAY TRADING SCAN (1D) ─────────────────────────────────────────────────
    logger.info("\n── DAY TRADING SIGNALS (1D) ──────────────────────────────────────")

    day_signals_sp500 = scan_tickers(
        sp500_tickers, "S&P 500", interval="1d", period="90d", timeframe_label="daily"
    )
    day_signals_ndx = scan_tickers(
        nasdaq_only, "NASDAQ 100", interval="1d", period="90d", timeframe_label="daily"
    )
    day_signals = day_signals_sp500 + day_signals_ndx
    day_signals.sort(key=lambda x: (0 if x["signal"] == "STRONG BUY" else 1, -x["score"]))

    logger.info(f"Day trading signals found: {len(day_signals)}")

    # ── WEEKLY SWING SCAN (1W) ────────────────────────────────────────────────
    logger.info("\n── WEEKLY SWING SIGNALS (1W) ─────────────────────────────────────")

    weekly_signals_sp500 = scan_tickers(
        sp500_tickers, "S&P 500", interval="1wk", period="2y", timeframe_label="weekly"
    )
    weekly_signals_ndx = scan_tickers(
        nasdaq_only, "NASDAQ 100", interval="1wk", period="2y", timeframe_label="weekly"
    )
    weekly_signals = weekly_signals_sp500 + weekly_signals_ndx
    weekly_signals.sort(key=lambda x: (0 if x["signal"] == "STRONG BUY" else 1, -x["score"]))

    logger.info(f"Weekly swing signals found: {len(weekly_signals)}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(day_signals, weekly_signals)

    # ── Build & send email ────────────────────────────────────────────────────
    total = len(day_signals) + len(weekly_signals)

    if total == 0:
        subject   = f"📈 Stock Signals — {TODAY_STR} | No qualifying signals today"
        html_body = build_html_email([], [], TODAY_STR)
        logger.info("No qualifying signals found. Sending empty report email.")
    else:
        subject   = f"📈 Stock Signals — {TODAY_STR} | {total} BUY Opportunities"
        html_body = build_html_email(day_signals, weekly_signals, TODAY_STR)

    send_email(html_body, subject)

    logger.info("\n── SUMMARY ───────────────────────────────────────────────────────")
    logger.info(f"  Total signals : {total}")
    logger.info(f"  STRONG BUY    : {len([s for s in day_signals+weekly_signals if s['signal']=='STRONG BUY'])}")
    logger.info(f"  BUY           : {len([s for s in day_signals+weekly_signals if s['signal']=='BUY'])}")
    logger.info(f"  Day trading   : {len(day_signals)}")
    logger.info(f"  Weekly swing  : {len(weekly_signals)}")
    logger.info(f"  Log file      : {LOG_FILE}")
    logger.info(f"  CSV file      : {CSV_FILE}")
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()


# =============================================================================
# WINDOWS TASK SCHEDULER SETUP
# =============================================================================
#
# STEP 1 — INSTALL DEPENDENCIES
# ──────────────────────────────
#   Open Command Prompt and run:
#       pip install yfinance pandas pandas-ta requests python-dotenv tqdm groq lxml html5lib beautifulsoup4
#
# STEP 2 — CREATE YOUR .env FILE
# ────────────────────────────────
#   In the same folder as stock_signals.py, create a file named ".env" with:
#
#       GMAIL_USER=your.address@gmail.com
#       GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
#       NOTIFY_EMAIL=recipient@email.com
#       GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
#
#   To get a Gmail App Password:
#     1. Enable 2-Step Verification at myaccount.google.com/security
#     2. Go to myaccount.google.com/apppasswords
#     3. Create a new App Password → select "Mail" + "Windows Computer"
#     4. Copy the 16-character password into .env (no spaces)
#
#   To get a free Groq API key:
#     1. Go to console.groq.com → sign up free
#     2. Click "API Keys" → Create API Key
#     3. Paste into GROQ_API_KEY in .env
#
# STEP 3 — FIND YOUR PYTHON PATH
# ────────────────────────────────
#   Run in Command Prompt:
#       where pythonw.exe
#   Example result: C:\Users\YourName\AppData\Local\Programs\Python\Python312\pythonw.exe
#   (pythonw.exe runs silently — no console window pops up)
#
# STEP 4 — REGISTER IN TASK SCHEDULER
# ──────────────────────────────────────
#   Option A — Via GUI:
#     1. Press Win + S → search "Task Scheduler" → Open
#     2. Click "Create Basic Task" in the right panel
#     3. Name: "Stock Signal Generator"
#     4. Trigger: Daily → set time to 09:00
#     5. Action: "Start a program"
#        Program:   C:\Users\YourName\AppData\Local\Programs\Python\Python312\pythonw.exe
#        Arguments: "C:\path\to\stock_signals.py"
#        Start in:  C:\path\to\your\script\folder
#     6. Finish → Right-click the task → Properties → Triggers:
#        - Edit trigger → set "Repeat task" or add days: Mon, Tue, Wed, Thu, Fri
#
#   Option B — Via PowerShell (run as Administrator):
#       $action  = New-ScheduledTaskAction `
#                    -Execute "C:\Users\YourName\AppData\Local\Programs\Python\Python312\pythonw.exe" `
#                    -Argument "C:\path\to\stock_signals.py" `
#                    -WorkingDirectory "C:\path\to\script\folder"
#
#       $trigger = New-ScheduledTaskTrigger `
#                    -Weekly `
#                    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
#                    -At 9:00AM
#
#       Register-ScheduledTask `
#         -TaskName "StockSignalGenerator" `
#         -Action $action `
#         -Trigger $trigger `
#         -RunLevel Highest
#
# STEP 5 — RECOMMENDED TRIGGERS
# ──────────────────────────────
#   • Day trading signals : Mon–Fri at 09:00 (30 min before US market open)
#   • Weekly swing signals: Friday at 18:00  (after US market close)
#   Tip: You can create two separate scheduled tasks pointing to the same script;
#        it will run both scans every time regardless of trigger.
#
# STEP 6 — TEST THE SCRIPT
# ─────────────────────────
#   Before scheduling, run manually:
#       python stock_signals.py
#   Check the log file (signals_YYYY-MM-DD.log) and your email inbox.
#
# STEP 7 — TROUBLESHOOTING
# ─────────────────────────
#   • Script runs but no email → check .env credentials; test with python -c "import smtplib"
#   • pandas-ta error → run: pip install --upgrade pandas-ta
#   • yfinance returns empty data → ticker may be delisted; check on finance.yahoo.com
#   • Groq error → your free tier may be rate-limited; the script will use template fallback
#   • Task doesn't run → in Task Scheduler properties, set "Run whether user is logged on or not"
#                        and check "Run with highest privileges"
# =============================================================================

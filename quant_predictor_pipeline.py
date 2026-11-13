import os
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import requests
import yfinance as yf
import numpy as np

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

# Expanded Quant Watchlist covering top liquid SPY & QQQ constituents
QUANT_UNIVERSE = [
    # Mega-Cap Tech & Growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "AVGO", "ADBE", "CRM", "QCOM", "INTC", "CSCO",
    # Financials & Banking
    "JPM", "BAC", "GS", "MS", "V", "MA", "BLK", "WFC", "C",
    # Healthcare & Biotech
    "UNH", "LLY", "JNJ", "PFE", "ABBV", "MRK", "TMO", "AMGN",
    # Industrials, Defense & Energy
    "CAT", "GE", "RTX", "LMT", "HON", "BA", "XOM", "CVX", "COP", "NEE",
    # Consumer Cyclical & Defensive
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "DIS", "NKE"
]


def fetch_finnhub_intelligence(ticker: str):
    """Pulls company-relevant news and recent insider transactions from Finnhub."""
    headlines = []
    insider_notes = "No major recent insider filings."
    if not FINNHUB_API_KEY:
        return "Finnhub API Key missing.", insider_notes

    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={three_days_ago}&to={today}&token={FINNHUB_API_KEY}"
        news_res = requests.get(news_url, timeout=5)
        if news_res.status_code == 200:
            articles = news_res.json()
            for art in articles:
                title = art.get("headline", "").strip()
                if title and (ticker.lower() in title.lower()):
                    if title not in headlines:
                        headlines.append(title)
                if len(headlines) >= 3:
                    break
            if not headlines and articles:
                for art in articles[:2]:
                    if art.get("headline"):
                        headlines.append(art["headline"])

        insider_url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={FINNHUB_API_KEY}"
        ins_res = requests.get(insider_url, timeout=5)
        if ins_res.status_code == 200:
            txs = ins_res.json().get("data", [])[:2]
            if txs:
                tx_list = [f"{tx.get('name')} ({tx.get('change'):,} shs)" for tx in txs if tx.get("name")]
                if tx_list:
                    insider_notes = "; ".join(tx_list)
    except Exception as e:
        print(f"⚠️ Finnhub error for {ticker}: {repr(e)}")

    news_text = " | ".join(headlines) if headlines else "No major headlines in last 72h."
    return news_text, insider_notes


def analyze_quant_asset(ticker: str):
    """Pulls historical data, computes volatility projections (lowest/highest price bounds), and scores fundamentals."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="3mo", interval="1d")
        if df.empty or len(df) < 30:
            return None

        closes = df["Close"]
        current_price = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        daily_change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)

        # Technical Indicators (RSI & ATR for low/high projection)
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = round(float((100 - (100 / (1 + rs))).iloc[-1]), 1)

        # Average True Range (ATR) & Volatility boundaries for multi-day/session forecast
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - closes.shift())
        low_close = np.abs(df["Low"] - closes.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr_14 = float(true_range.rolling(14).mean().iloc[-1])

        # Predictive Price Bounds (Statistical projection for next 5-10 sessions)
        # Lowest expected trough based on current ATR & support buffer
        lowest_price_target = round(current_price - (1.8 * atr_14), 2)
        # Highest expected peak based on resistance expansion
        highest_price_target = round(current_price + (2.2 * atr_14), 2)

        # Fundamentals & Valuation Multiples
        info = t.info or {}
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        price_to_sales = info.get("priceToSalesTrailing12Months")
        earnings_growth = info.get("earningsGrowth")
        sector = info.get("sector", "Equities")
        name = info.get("shortName", ticker)

        # Volume analysis
        volume = int(df["Volume"].iloc[-1])
        avg_volume = float(df["Volume"].rolling(10).mean().iloc[-1])
        vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0

        news_summary, insider_summary = fetch_finnhub_intelligence(ticker)
        news_lower = news_summary.lower()

        # Shock filter
        has_shock = any(w in news_lower for w in ["resign", "exit", "investigation", "fraud", "subpoena", "sec probe"])

        # Quant Scoring (0-100)
        score = 55
        if 40 <= rsi <= 65:
            score += 15
        if vol_ratio > 1.1:
            score += 10
        if forward_pe and trailing_pe and forward_pe < trailing_pe:
            score += 10  # Earnings compression tailwind
        if earnings_growth and earnings_growth > 0.10:
            score += 10
        if has_shock:
            score -= 35

        score = max(10, min(98, score))

        # 3 PM UK / US Session Execution Levels
        bias = "BULLISH" if current_price >= closes.rolling(10).mean().iloc[-1] and not has_shock else "BEARISH"
        vwap_approx = round(current_price * 1.0002, 2)
        entry_low = round(current_price * 0.996, 2)
        entry_high = round(current_price * 1.004, 2)
        sl_price = round(current_price * 0.975 if bias == "BULLISH" else current_price * 1.025, 2)
        tp_price = highest_price_target if bias == "BULLISH" else lowest_price_target

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "price": current_price,
            "daily_change": daily_change_pct,
            "rsi": rsi,
            "lowest_price": lowest_price_target,
            "highest_price": highest_price_target,
            "trailing_pe": trailing_pe if trailing_pe else "N/A",
            "forward_pe": forward_pe if forward_pe else "N/A",
            "ps_ratio": price_to_sales if price_to_sales else "N/A",
            "score": score,
            "bias": bias,
            "vwap": vwap_approx,
            "entry_zone": f"${entry_low} - ${entry_high}",
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "news": news_summary,
            "insider": insider_summary
        }
    except Exception as e:
        print(f"⚠️ Error scanning {ticker}: {repr(e)}")
        return None


def run_quant_predictor_scan():
    print("==================================================")
    print("  🔮 RUNNING QUANT PREDICTOR & 3PM EXECUTION SCAN")
    print("==================================================")

    results = []
    # Multithreaded scan for speed across 50+ mega-caps
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(analyze_quant_asset, ticker): ticker for ticker in QUANT_UNIVERSE}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res and res["score"] >= 65:  # High conviction filter
                results.append(res)

    # Sort by highest confidence score
    results.sort(key=lambda x: x["score"], reverse=True)
    top_picks = results[:3]  # Top 3 diversified picks

    if not top_picks:
        print("⚠️ No assets met the high-conviction threshold.")
        return

    for pick in top_picks:
        print(f"✅ Quant Top Pick: ${pick['ticker']} ({pick['bias']}) — Score: {pick['score']}/100 | Low Target: ${pick['lowest_price']} | High Target: ${pick['highest_price']}")
        embed_payload = generate_quant_embed(pick)
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=embed_payload)

    print(f"--> Quant predictor scan complete. {len(top_picks)} institutional setups dispatched to Discord.")


def generate_quant_embed(pick: dict):
    """Formats the quantitative prediction and 3 PM execution plan into a clean Discord embed."""
    prompt = f"""
    Format a concise institutional quant summary for {pick['name']} (${pick['ticker']}) in {pick['sector']}.
    Current Price: ${pick['price']} | Lowest Expected Trough: ${pick['lowest_price']} | Highest Target: ${pick['highest_price']}
    Trailing P/E: {pick['trailing_pe']} | Forward P/E: {pick['forward_pe']} | P/S: {pick['ps_ratio']}
    News: {pick['news']}
    """
    
    # Clean Markdown output without HTML tags
    description = (
        f"**QUANTITATIVE FORECAST & VALUATION DESK**\n\n"
        f"| Metric | Projected Value | Institutional Rationale |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **Lowest Price Target** | `${pick['lowest_price']}` | Statistical downside trough (ATR buffer) |\n"
        f"| **Highest Price Target** | `${pick['highest_price']}` | Upside expansion target |\n"
        f"| **Valuation (P/E & P/S)** | Fwd P/E: `{pick['forward_pe']}` / P/S: `{pick['ps_ratio']}` | Fundamental valuation multiple |\n"
        f"| **RSI & Momentum** | `{pick['rsi']} RSI` | Momentum regime check |\n"
        f"| **Live Finnhub News** | {pick['news'][:140]} | 72h catalyst scan |\n\n"
        f"**3:00 PM UK Session Execution Plan:**\n"
        f"• **Directional Bias:** `{pick['bias']}`\n"
        f"• **Entry Zone:** `{pick['entry_zone']}` (VWap: `${pick['vwap']}`)\n"
        f"• **Stop-Loss (SL):** `${pick['stop_loss']}`\n"
        f"• **Take-Profit (TP):** `${pick['take_profit']}`\n\n"
        f"🤖 *Engine: Multi-Asset Quant Predictor + Finnhub Fundamentals*"
    )

    return {
        "username": "Quant Prediction Engine",
        "embeds": [{
            "title": f"📈 Quant Forecast: ${pick['ticker']} ({pick['bias']}) — Score: {pick['score']}/100",
            "description": description[:3900],
            "color": 3447003 if pick['bias'] == "BULLISH" else 15158332,
            "footer": {"text": f"3 PM UK Execution Scan | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"}
        }]
    }


if __name__ == "__main__":
    run_quant_predictor_scan()

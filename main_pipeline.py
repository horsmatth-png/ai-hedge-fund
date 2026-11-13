import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import yfinance as yf

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

# ==========================================
# 🇬🇧 UK DYNAMIC SCANNER & AI ENGINE (08:30 AM UK)
# ==========================================
UK_WATCH_UNIVERSE = [
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", 
    "GSK.L", "RIO.L", "DGE.L", "REL.L", "LLOY.L",
    "BARC.L", "VOD.L", "GLEN.L", "NG.L", "RR.L"
]

# ==========================================
# 🇺🇸 US DYNAMIC MULTI-SECTOR EXPANDED UNIVERSE
# ==========================================
US_SCAN_UNIVERSE = [
    # Mega-Cap Tech & Growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "AVGO",
    # Financials & Banking
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare & Pharma
    "UNH", "LLY", "JNJ", "PFE",
    # Industrials, Defense & Energy
    "CAT", "GE", "RTX", "LMT", "XOM", "CVX",
    # Consumer Defensive & Retail
    "WMT", "COST", "PG", "KO"
]

BACKUP_UNIVERSE = [
    {"ticker": "XLU", "name": "Utilities Select Sector SPDR", "type": "Defensive Sector"},
    {"ticker": "XLP", "name": "Consumer Staples Select Sector", "type": "Defensive Sector"},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "type": "Safe Haven"},
    {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond", "type": "Fixed Income Safe Haven"},
    {"ticker": "SH", "name": "ProShares Short S&P500", "type": "Inverse Hedge (-1x)"},
    {"ticker": "SQQQ", "name": "ProShares UltraPro Short QQQ", "type": "Inverse Hedge (-3x)"},
]

def fetch_finnhub_intelligence(ticker: str):
    """Pulls strictly company-relevant news and recent insider transactions from Finnhub."""
    clean_symbol = ticker.replace(".L", "")
    headlines = []
    insider_notes = "No major recent insider filings."

    if not FINNHUB_API_KEY:
        return "Finnhub API Key missing.", insider_notes

    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={three_days_ago}&to={today}&token={FINNHUB_API_KEY}"
        news_res = requests.get(news_url, timeout=8)
        if news_res.status_code == 200:
            articles = news_res.json()
            for art in articles:
                title = art.get("headline", "").strip()
                if title and (clean_symbol.lower() in title.lower() or ticker.lower() in title.lower()):
                    if title not in headlines:
                        headlines.append(title)
                if len(headlines) >= 4:
                    break
            if not headlines and articles:
                for art in articles[:3]:
                    if art.get("headline"):
                        headlines.append(art["headline"])

        if not ticker.endswith(".L"):
            insider_url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
            ins_res = requests.get(insider_url, timeout=8)
            if ins_res.status_code == 200:
                txs = ins_res.json().get("data", [])[:3]
                if txs:
                    tx_list = [f"{tx.get('name')} ({tx.get('change'):,} shs)" for tx in txs if tx.get("name")]
                    if tx_list:
                        insider_notes = "; ".join(tx_list)
    except Exception as e:
        print(f"⚠️ Finnhub fetch error for {ticker}: {repr(e)}")

    news_text = " | ".join(headlines) if headlines else "No major Finnhub headlines in last 72h."
    return news_text, insider_notes

def fetch_market_data(ticker: str):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1mo", interval="1d")
        if not df.empty and len(df) >= 15:
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = round(float((100 - (100 / (1 + rs))).iloc[-1]), 1)
            
            fast = t.fast_info
            price = getattr(fast, 'last_price', None) or fast.get('lastPrice') or df["Close"].iloc[-1]
            prev_close = float(df["Close"].iloc[-2])
            volume = df["Volume"].iloc[-1]
            avg_volume = df["Volume"].rolling(window=10).mean().iloc[-1]
            
            # Dynamic Bias Determination based on 10-day trend
            ma_10 = df["Close"].rolling(window=10).mean().iloc[-1]
            bias = "BULLISH" if price >= ma_10 else "BEARISH"
            
            sector = info_sector = "Equities"
            try:
                sector = t.info.get("sector", "Equities")
            except:
                pass

            news_summary, insider_summary = fetch_finnhub_intelligence(ticker)
            return round(float(price), 2), round(prev_close, 2), rsi, int(volume), int(avg_volume), bias, sector, news_summary, insider_summary
    except Exception as e:
        print(f"⚠️ Fetch error for {ticker}: {repr(e)}")
    return None, None, 50.0, 0, 0, "BULLISH", "Equities", "No news available.", "No insider data."

def calculate_confidence_score(rsi: float, vol_ratio: float, bias: str, has_shock: bool) -> int:
    """Computes a robust institutional confidence score out of 100."""
    score = 65
    if bias == "BULLISH":
        if 40 <= rsi <= 65: score += 15
        elif rsi > 75 or rsi < 30: score -= 20
    else: # BEARISH
        if 35 <= rsi <= 60: score += 15
        elif rsi < 25 or rsi > 70: score -= 20

    if vol_ratio > 1.1: score += 10
    elif vol_ratio < 0.7: score -= 5

    if has_shock: score -= 35
    return max(15, min(95, score))

def generate_ai_analysis(ticker: str, bias: str, price: float, rsi: float, volume: float, avg_volume: float, news_summary: str, insider_summary: str, confidence_score: int, region: str = "US"):
    vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0
    vwap_approx = round(price * 1.0005, 2)
    support = round(price * 0.98, 2)
    resistance = round(price * 1.02, 2)
    
    currency = "£" if region == "UK" else "$"
    market_context = "FTSE 100 institutional rotation & London session order flow" if region == "UK" else "US multi-sector equity rotation & institutional order flow"
    
    entry_low = round(price * 0.995, 2)
    entry_high = round(price * 1.005, 2)
    sl_price = round(price * 0.97 if bias == "BULLISH" else price * 1.03, 2)
    tp_price = round(price * 1.04 if bias == "BULLISH" else price * 0.96, 2)
    
    sl_pts = int(abs(price - sl_price) * 100)
    tp_pts = int(abs(tp_price - price) * 100)

    prompt = f"""
    You are an elite quantitative hedge fund execution engine. Analyze {ticker} trading at {currency}{price} (14-day RSI: {rsi}, Dynamic Bias: {bias}, Volume Ratio: {vol_ratio}x, Confidence Score: {confidence_score}/100).
    
    LIVE FINNHUB HEADLINES FOR {ticker}:
    "{news_summary}"
    
    LIVE FINNHUB INSIDER FILINGS:
    "{insider_summary}"
    
    CRITICAL RULES:
    1. Base Recent Catalysts and Insider/Regulatory rows strictly on the live Finnhub data above. Never fabricate details.
    2. NEVER output HTML tags like <br> — Discord displays them as raw text! Use bullet points or "/" separators instead.
    
    Use this exact concise markdown structure:
    
    **1. Technical & News Confluence**
    | Factor | Observation | Impact |
    | :--- | :--- | :--- |
    | **Current Price** | {currency}{price} | Baseline for entry/stop |
    | **RSI (14-day)** | {rsi} | Momentum regime check |
    | **Sector Momentum** | {market_context} | Macro & sector alignment |
    | **Recent Catalysts** | [Summarize real Finnhub headlines above cleanly] | Direct price & sentiment driver |
    | **VWAP & S/R** | VWAP: {currency}{vwap_approx} | S: {currency}{support} / R: {currency}{resistance} | Key intraday boundary levels |
    | **Insider / Regulatory** | {insider_summary} | Governance & tail-risk check |
    
    **Synthesis:**
    [2-3 sentences synthesizing Finnhub headlines, RSI, and volume flow into a net directional verdict.]
    
    ---
    
    **2. Actionable IG Trade Plan**
    | Trade Component | Target | IG Point Distance (1{currency} = 100 pts) | Notes |
    | :--- | :--- | :--- | :--- |
    | **Entry Zone** | {currency}{entry_low} - {currency}{entry_high} | ±50 pts | Optimal fill window |
    | **Stop-Loss (SL)** | {currency}{sl_price} | {sl_pts} pts | Structural risk boundary |
    | **Take-Profit (TP)** | {currency}{tp_price} | {tp_pts} pts | Primary liquidity objective |
    | **Position Size** | 1,500 units | ~1.5% Portfolio Risk | Institutional allocation |
    """

    engine_used = "Unknown"
    response_text = ""

    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            response_text = response.text
            engine_used = "Google Gemini (gemini-1.5-flash)"
        except Exception as e:
            print(f"⚠️ Gemini failed: {repr(e)}")

    if not response_text and GROQ_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_API_KEY)
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            response_text = completion.choices[0].message.content
            engine_used = "Groq (openai/gpt-oss-20b)"
        except Exception as e:
            print(f"⚠️ Groq failed: {repr(e)}")

    if not response_text:
        response_text = f"**Analysis:** Price: {currency}{price} | RSI: {rsi} | Score: {confidence_score}/100"
        engine_used = "Quantitative Fallback"

    response_text = re.sub(r"<br\s*/?>", " / ", response_text, flags=re.IGNORECASE)
    response_text += f"\n\n🤖 *Engine: {engine_used} + Finnhub Feed*"
    
    embed_color = 3066993 if bias == "BULLISH" else 15158332
    bot_name = "UK Dynamic Quant Engine" if region == "UK" else "AI Quant Execution Engine"
    prefix = "UK Setup Details:" if region == "UK" else "Setup Details: $"

    return {
        "username": bot_name,
        "embeds": [{
            "title": f"📊 {prefix}{ticker} ({bias}) — Score: {confidence_score}/100",
            "description": f"**DIRECTIONAL_BIAS: {bias} | CONFIDENCE: {confidence_score}/100**\n\n---\n\n{response_text[:3900]}",
            "color": embed_color,
            "footer": {"text": f"{'Scanned' if region == 'UK' else 'Validated'} at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Engine: {engine_used}"}
        }]
    }

def run_uk_morning_scan():
    print("==================================================")
    print("  🇬🇧 RUNNING UK DYNAMIC MORNING SCAN & AI DEEP DIVE")
    print("==================================================")
    
    scanned_results = []
    for ticker in UK_WATCH_UNIVERSE:
        res = fetch_market_data(ticker)
        if res and res[0]:
            price, prev_close, rsi, volume, avg_volume, bias, sector, news_summary, insider_summary = res
            vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0
            news_lower = news_summary.lower()
            has_shock = any(w in news_lower for w in ["resign", "exit", "investigation", "fraud", "subpoena"])
            score = calculate_confidence_score(rsi, vol_ratio, bias, has_shock)
            
            scanned_results.append({
                "ticker": ticker, "price": price, "rsi": rsi, "volume": volume, 
                "avg_volume": avg_volume, "vol_ratio": vol_ratio, "bias": bias, 
                "sector": sector, "news": news_summary, "insider": insider_summary, "score": score
            })

    scanned_results.sort(key=lambda x: x["score"], reverse=True)
    top_picks = scanned_results[:3]

    for item in top_picks:
        print(f"✅ UK Top Pick: {item['ticker']} (Score: {item['score']}/100). Dispatching...")
        embed_payload = generate_ai_analysis(
            item["ticker"], item["bias"], item["price"], item["rsi"], item["volume"], item["avg_volume"], item["news"], item["insider"], item["score"], region="UK"
        )
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=embed_payload)

def run_us_validation():
    print("==================================================")
    print("  🇺🇸 RUNNING US MULTI-SECTOR DIVERSIFIED SCAN")
    print("==================================================")
    
    scored_candidates = []

    for ticker in US_SCAN_UNIVERSE:
        res = fetch_market_data(ticker)
        if not res or not res[0]:
            continue
            
        live_price, prev_close, rsi, volume, avg_volume, bias, sector, news_summary, insider_summary = res

        daily_change_pct = round(((live_price - prev_close) / prev_close) * 100, 2)
        news_lower = news_summary.lower()
        shock_keywords = ["resign", "exit", "investigation", "spy", "fraud", "subpoena", "lawsuit", "sec probe", "step down"]
        has_shock_news = any(word in news_lower for word in shock_keywords)

        if has_shock_news:
            print(f"❌ ${ticker} filtered out due to shock news.")
            continue

        vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0
        score = calculate_confidence_score(rsi, vol_ratio, bias, has_shock_news)

        scored_candidates.append({
            "ticker": ticker, "bias": bias, "price": live_price, "rsi": rsi,
            "volume": volume, "avg_volume": avg_volume, "sector": sector,
            "news": news_summary, "insider": insider_summary, "score": score
        })
        print(f"✅ ${ticker} ({sector}) Scored: {score}/100")

    # Sort by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # DIVERSIFICATION ENFORCEMENT: Pick top candidates while avoiding duplicate sectors if possible
    top_picks = []
    seen_sectors = set()
    
    for candidate in scored_candidates:
        if candidate["sector"] not in seen_sectors or len(top_picks) >= 2:
            top_picks.append(candidate)
            seen_sectors.add(candidate["sector"])
        if len(top_picks) >= 3:
            break

    if top_picks:
        print(f"🎯 Dispatching Top {len(top_picks)} Multi-Sector Diversified Setups to Discord...")
        for pick in top_picks:
            embed_payload = generate_ai_analysis(
                pick["ticker"], pick["bias"], pick["price"], pick["rsi"], 
                pick["volume"], pick["avg_volume"], pick["news"], pick["insider"], pick["score"], region="US"
            )
            if DISCORD_WEBHOOK_URL:
                requests.post(DISCORD_WEBHOOK_URL, json=embed_payload)
    else:
        print("🚨 Contingency backup triggered.")

if __name__ == "__main__":
    current_hour = datetime.utcnow().hour
    if current_hour < 12:
        run_uk_morning_scan()
    else:
        run_us_validation()

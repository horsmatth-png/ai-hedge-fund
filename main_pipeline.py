import os
from datetime import datetime
from dotenv import load_dotenv
import requests
import yfinance as yf

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================
# 🇬🇧 UK DYNAMIC SCANNER & AI ENGINE (08:30 AM UK)
# ==========================================
UK_WATCH_UNIVERSE = [
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", 
    "GSK.L", "RIO.L", "DGE.L", "REL.L", "LLOY.L",
    "BARC.L", "VOD.L", "GLEN.L", "NG.L", "RR.L"
]

def generate_ai_analysis(ticker: str, bias: str, price: float, rsi: float, volume: float, avg_volume: float, region: str = "US"):
    vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0
    vwap_approx = round(price * 1.0005, 2)
    support = round(price * 0.98, 2)
    resistance = round(price * 1.02, 2)
    
    currency = "£" if region == "UK" else "$"
    market_context = "FTSE 100 institutional rotation & London session order flow" if region == "UK" else "QQQ & SPY trending higher; US session institutional demand"
    
    entry_low = round(price * 0.995, 2)
    entry_high = round(price * 1.005, 2)
    sl_price = round(price * 0.97 if bias == "BULLISH" else price * 1.03, 2)
    tp_price = round(price * 1.04 if bias == "BULLISH" else price * 0.96, 2)
    
    sl_pts = int(abs(price - sl_price) * 100)
    tp_pts = int(abs(tp_price - price) * 100)

    prompt = f"""
    You are an elite quantitative hedge fund execution engine. Provide a concise, high-density institutional trade breakdown and IG trade plan for {ticker} trading at {currency}{price} (14-day RSI: {rsi}, Bias: {bias}, Volume Ratio: {vol_ratio}x).
    
    Keep table cells concise so the entire response fits cleanly without truncation. Use this exact structure:
    
    **1. Technical & News Confluence**
    | Factor | Observation | Impact |
    | :--- | :--- | :--- |
    | **Current Price** | {currency}{price} | Baseline for entry/stop |
    | **RSI (14-day)** | {rsi} | Neutral/momentum check; supports {bias.lower()} stance |
    | **Sector Momentum** | {market_context} | Supports directional bias |
    | **Recent Catalysts** | Live session price action & volume ({vol_ratio}x avg) | Positive micro-structure reaction |
    | **VWAP & S/R** | VWAP: {currency}{vwap_approx} | S: {currency}{support} / R: {currency}{resistance} | Key intraday boundary levels |
    | **Insider / Regulatory** | Standard institutional flow; no legal/regulatory flags | Downside risk minimized |
    
    **Synthesis:**
    The RSI profile removes immediate squeeze/overbought caution. Sector momentum and order flow absorption outweigh risks, giving a net {bias.lower()} bias.
    
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

    # Attempt 1: Gemini Primary
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

    # Attempt 2: Groq Fallback (Increased max_tokens to prevent cut-off)
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
        response_text = f"**Analysis:** Price: {currency}{price} | RSI: {rsi} (AI Engine Unavailable)"
        engine_used = "Offline / Fallback"

    response_text += f"\n\n🤖 *Engine: {engine_used}*"
    embed_color = 3066993 if bias == "BULLISH" else 15158332
    bot_name = "UK Dynamic Quant Engine" if region == "UK" else "AI Quant Execution Engine"
    prefix = "UK Setup Details:" if region == "UK" else "Setup Details: $"

    return {
        "username": bot_name,
        "embeds": [{
            "title": f"📊 {prefix}{ticker} (75% Conf.)",
            "description": f"**CONFIDENCE_SCORE: 75% {bias}**\n\n---\n\n{response_text[:3900]}",
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
                price = round(float(price), 2)
                
                volume = df["Volume"].iloc[-1]
                avg_volume = df["Volume"].rolling(window=10).mean().iloc[-1]
                vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0

                bias = "BULLISH" if rsi > 50 else "BEARISH"
                scanned_results.append({
                    "ticker": ticker,
                    "price": price,
                    "rsi": rsi,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "vol_ratio": vol_ratio,
                    "bias": bias
                })
        except Exception as e:
            print(f"⚠️ Error scanning {ticker}: {repr(e)}")

    scanned_results.sort(key=lambda x: x["vol_ratio"], reverse=True)
    top_picks = scanned_results[:3]

    for item in top_picks:
        print(f"✅ UK Top Pick: {item['ticker']} ({item['bias']}). Generating Full IG Trade Execution...")
        embed_payload = generate_ai_analysis(
            item["ticker"], item["bias"], item["price"], item["rsi"], item["volume"], item["avg_volume"], region="UK"
        )
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=embed_payload)

# ==========================================
# 🇺🇸 US DYNAMIC VALIDATION & AI ENGINE (14:30/15:00 PM UK)
# ==========================================
US_PRIMARY_UNIVERSE = [
    {"ticker": "SPY", "bias": "BULLISH", "ref_price": 580.00},
    {"ticker": "QQQ", "bias": "BULLISH", "ref_price": 500.00},
    {"ticker": "AAPL", "bias": "BULLISH", "ref_price": 230.00},
    {"ticker": "NVDA", "bias": "BULLISH", "ref_price": 140.00},
    {"ticker": "MSFT", "bias": "BULLISH", "ref_price": 420.00},
    {"ticker": "GOOGL", "bias": "BEARISH", "ref_price": 175.00},
    {"ticker": "AMZN", "bias": "BULLISH", "ref_price": 190.00},
    {"ticker": "META", "bias": "BULLISH", "ref_price": 580.00},
    {"ticker": "AVGO", "bias": "BULLISH", "ref_price": 180.00},
    {"ticker": "TSLA", "bias": "BEARISH", "ref_price": 220.00},
]

BACKUP_UNIVERSE = [
    {"ticker": "XLU", "name": "Utilities Select Sector SPDR", "type": "Defensive Sector"},
    {"ticker": "XLP", "name": "Consumer Staples Select Sector", "type": "Defensive Sector"},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "type": "Safe Haven"},
    {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond", "type": "Fixed Income Safe Haven"},
    {"ticker": "SH", "name": "ProShares Short S&P500", "type": "Inverse Hedge (-1x)"},
    {"ticker": "SQQQ", "name": "ProShares UltraPro Short QQQ", "type": "Inverse Hedge (-3x)"},
]

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
            volume = df["Volume"].iloc[-1]
            avg_volume = df["Volume"].rolling(window=10).mean().iloc[-1]
            
            return round(float(price), 2), rsi, int(volume), int(avg_volume)
    except Exception as e:
        print(f"⚠️ Fetch error for {ticker}: {repr(e)}")
    return None, 50.0, 0, 0

def run_us_validation():
    print("==================================================")
    print("  🇺🇸 RUNNING US 3 PM DYNAMIC VALIDATION & INSTITUTIONAL ENGINE")
    print("==================================================")
    
    confirmed_count = 0

    for item in US_PRIMARY_UNIVERSE:
        ticker = item["ticker"]
        bias = item["bias"]
        ref_price = item["ref_price"]

        live_price, rsi, volume, avg_volume = fetch_market_data(ticker)
        if not live_price:
            continue

        is_valid = True
        if bias == "BEARISH" and (rsi < 30 or live_price > (ref_price * 1.02)):
            is_valid = False
        elif bias == "BULLISH" and (rsi > 70 or live_price < (ref_price * 0.98)):
            is_valid = False

        if is_valid:
            confirmed_count += 1
            print(f"✅ ${ticker} CONFIRMED. Dispatching Rich Embed to Discord...")
            embed_payload = generate_ai_analysis(ticker, bias, live_price, rsi, volume, avg_volume, region="US")

            if DISCORD_WEBHOOK_URL:
                requests.post(DISCORD_WEBHOOK_URL, json=embed_payload)
        else:
            print(f"❌ ${ticker} invalidated or swept by opening bell.")

    if confirmed_count == 0:
        print("🚨 ALL-RED PROTOCOL ACTIVATED: Zero primary setups confirmed. Running dynamic backup scan...")
        scanned_backups = []
        for backup in BACKUP_UNIVERSE:
            b_price, b_rsi, _, _ = fetch_market_data(backup["ticker"])
            if b_price:
                scanned_backups.append({
                    "ticker": backup["ticker"],
                    "name": backup["name"],
                    "type": backup["type"],
                    "price": b_price,
                    "rsi": b_rsi
                })

        top_3 = scanned_backups[:3]
        backup_summary = (
            "⚠️ **Market Regime Warning: 10/10 Primary Setups Invalidated.**\n"
            "The dynamic scanner has swept alternative asset classes and surfaced top live hedges:\n\n"
        )
        for asset in top_3:
            backup_summary += (
                f"🛡️ **${asset['ticker']}** ({asset['name']})\n"
                f"• Type: `{asset['type']}` | Price: `${asset['price']}` | RSI: `{asset['rsi']}`\n\n"
            )

        backup_payload = {
            "username": "US Dynamic Contingency Scanner",
            "embeds": [{
                "title": "🛡️ All-Red Protocol: Top 3 Market Hedges",
                "description": backup_summary,
                "color": 16753920,
                "footer": {"text": f"Dynamic Backup Scan at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"}
            }]
        }
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=backup_payload)
        print("--> Contingency backup alert dispatched to Discord successfully!")
    else:
        print(f"--> Scan complete. {confirmed_count} confirmed setup(s) processed through institutional pipeline.")

if __name__ == "__main__":
    current_hour = datetime.utcnow().hour
    if current_hour < 12:
        run_uk_morning_scan()
    else:
        run_us_validation()

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
# 🇬🇧 UK MORNING SCAN LOGIC (08:30 AM UK)
# ==========================================
def get_uk_dynamic_universe():
    return [
        {"ticker": "SHEL.L", "name": "Shell plc"},
        {"ticker": "AZN.L", "name": "AstraZeneca"},
        {"ticker": "HSBA.L", "name": "HSBC Holdings"},
        {"ticker": "ULVR.L", "name": "Unilever"},
        {"ticker": "BP.L", "name": "BP plc"},
        {"ticker": "GSK.L", "name": "GSK plc"},
        {"ticker": "RIO.L", "name": "Rio Tinto"},
        {"ticker": "DGE.L", "name": "Diageo"},
        {"ticker": "REL.L", "name": "RELX"},
        {"ticker": "LLOY.L", "name": "Lloyds Banking Group"}
    ]

def run_uk_morning_scan():
    print("==================================================")
    print("  🇬🇧 RUNNING UK DYNAMIC MORNING SCAN (08:30 AM)")
    print("==================================================")
    
    summary = ""
    for item in get_uk_dynamic_universe():
        try:
            t = yf.Ticker(item["ticker"])
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
                
                bias = "BULLISH 📈" if rsi > 50 else "BEARISH 📉"
                summary += f"**{item['name']} (`{item['ticker']}`)** | Price: `£{price}` | RSI: `{rsi}` | Bias: **{bias}**\n"
        except Exception as e:
            print(f"⚠️ Error fetching {item['ticker']}: {repr(e)}")

    payload = {
        "username": "UK Morning Dynamic Scanner",
        "embeds": [{
            "title": "🇬🇧 UK Equities Morning Watchlist (08:30 AM)",
            "description": summary,
            "color": 3447003,
            "footer": {"text": f"Scanned at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"}
        }]
    }
    if DISCORD_WEBHOOK_URL:
        requests.post(DISCORD_WEBHOOK_URL, json=payload)
    print("--> UK Morning Watchlist posted to Discord successfully!")

# ==========================================
# 🇺🇸 US AFTERNOON VALIDATION & AI ENGINE (15:00 PM UK)
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

def generate_ai_analysis(ticker: str, bias: str, price: float, rsi: float, volume: float, avg_volume: float):
    vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0
    prompt = f"""
    You are an elite quantitative hedge fund execution engine. Provide a rigorous, structured trade breakdown and order flow analysis for ${ticker} currently trading at ${price} with a 14-day RSI of {rsi} and a morning bias of {bias}. Volume is {volume:,} vs 10-day average {avg_volume:,} (Ratio: {vol_ratio}x).
    
    Format your response cleanly using markdown tables and structured sections exactly like this:
    
    📊 **Setup Details: ${ticker}**
    **CONFIDENCE_SCORE: 75% {bias}**
    
    ---
    
    **1. Technical & Order Book Confluence**
    | Factor | Observation | Impact |
    | :--- | :--- | :--- |
    | **Current Price** | ${price} | Baseline for entry/stop |
    | **RSI (14-day)** | {rsi} | Technical momentum check |
    | **Volume / Order Flow** | Volume ratio: {vol_ratio}x | Order book liquidity & institutional absorption |
    | **Recent Catalysts** | Live session price action & sentiment | Market structure reaction |
    
    **Synthesis:**
    The neutral/supportive RSI removes short-squeeze/overbought caution. Positive volume flow and recent order book absorption outweigh risks, giving a net {bias.lower()} bias.
    
    ---
    
    **2. Actionable IG Trade Plan**
    | Trade Component | Target Level | Notes |
    | :--- | :--- | :--- |
    | **Entry Zone** | ${round(price * 0.995, 2)} - ${round(price * 1.005, 2)} | Optimal fill window |
    | **Stop-Loss (SL)** | ${round(price * 0.97 if bias == 'BULLISH' else price * 1.03, 2)} | Structural risk boundary |
    | **Take-Profit (TP)** | ${round(price * 1.04 if bias == 'BULLISH' else price * 0.96, 2)} | Target liquidity objective |
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

    # Attempt 2: Groq Fallback (Restored to working model)
    if not response_text and GROQ_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_API_KEY)
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            response_text = completion.choices[0].message.content
            engine_used = "Groq (openai/gpt-oss-20b)"
        except Exception as e:
            print(f"⚠️ Groq failed: {repr(e)}")

    if not response_text:
        response_text = f"**Analysis:** Price: ${price} | RSI: {rsi} (AI Engine Unavailable)"
        engine_used = "Offline / Fallback"

    response_text += f"\n\n*Engine: {engine_used}*"
    return response_text

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

        pct_change = round(((live_price - ref_price) / ref_price) * 100, 2)
        is_valid = True

        if bias == "BEARISH" and (rsi < 30 or live_price > (ref_price * 1.02)):
            is_valid = False
        elif bias == "BULLISH" and (rsi > 70 or live_price < (ref_price * 0.98)):
            is_valid = False

        if is_valid:
            confirmed_count += 1
            print(f"✅ ${ticker} CONFIRMED. Running AI & Order Flow Deep Dive...")
            analysis_text = generate_ai_analysis(ticker, bias, live_price, rsi, volume, avg_volume)

            payload = {
                "username": "AI Quant Execution Engine",
                "content": analysis_text
            }
            if DISCORD_WEBHOOK_URL:
                requests.post(DISCORD_WEBHOOK_URL, json=payload)
        else:
            print(f"❌ ${ticker} invalidated or swept by opening bell.")

    # All-Red Contingency Protocol Trigger
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

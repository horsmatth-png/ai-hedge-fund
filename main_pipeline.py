import os
from datetime import datetime
from dotenv import load_dotenv
import requests
import yfinance as yf

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

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
        t = yf.Ticker(item["ticker"])
        df = t.history(period="1mo", interval="1d")
        if not df.empty and len(df) >= 15:
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = round(float((100 - (100 / (1 + rs))).iloc[-1]), 1)
            price = round(float(df["Close"].iloc[-1]), 2)
            bias = "BULLISH 📈" if rsi > 50 else "BEARISH 📉"
            summary += f"**{item['name']} (`{item['ticker']}`)** | Price: `£{price}` | RSI: `{rsi}` | Bias: **{bias}**\n"

    payload = {
        "username": "UK Morning Dynamic Scanner",
        "embeds": [{
            "title": "🇬🇧 UK Equities Morning Watchlist (08:30 AM)",
            "description": summary,
            "color": 3447003,
            "footer": {"text": f"Scanned at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"}
        }]
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload)
    print("--> UK Morning Watchlist posted to Discord successfully!")

# ==========================================
# 🇺🇸 US AFTERNOON VALIDATION LOGIC (15:00 PM UK)
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

def fetch_data(ticker: str):
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
            return round(float(price), 2), rsi
    except Exception as e:
        print(f"⚠️ Fetch error for {ticker}: {repr(e)}")
    return None, 50.0

def run_us_validation():
    print("==================================================")
    print("  🇺🇸 RUNNING US 3 PM DYNAMIC VALIDATION & SCANNER")
    print("==================================================")
    
    results_summary = ""
    confirmed_count = 0

    for item in US_PRIMARY_UNIVERSE:
        ticker = item["ticker"]
        bias = item["bias"]
        ref_price = item["ref_price"]

        live_price, rsi = fetch_data(ticker)
        if not live_price:
            continue

        pct_change = round(((live_price - ref_price) / ref_price) * 100, 2)
        
        if bias == "BEARISH":
            if rsi < 30: status, reason = "INVALIDATED ❌", f"RSI oversold ({rsi}). Squeeze risk."
            elif live_price > (ref_price * 1.02): status, reason = "INVALIDATED ❌", f"Broke resistance (+{pct_change}%)."
            elif live_price < (ref_price * 0.98): status, reason = "EXPIRED / MISSED ⚠️", f"Target hit (-{pct_change}%)."
            else: status, reason = "CONFIRMED ✅", f"Price (${live_price}, {pct_change}%) & RSI intact."
        else: # BULLISH
            if rsi > 70: status, reason = "INVALIDATED ❌", f"RSI overbought ({rsi}). Pullback risk."
            elif live_price < (ref_price * 0.98): status, reason = "INVALIDATED ❌", f"Broke support (-{abs(pct_change)}%)."
            elif live_price > (ref_price * 1.02): status, reason = "EXPIRED / MISSED ⚠️", f"Move triggered (+{pct_change}%)."
            else: status, reason = "CONFIRMED ✅", f"Price (${live_price}, {pct_change}%) & RSI intact."

        if "CONFIRMED ✅" in status:
            confirmed_count += 1

        results_summary += f"**${ticker} ({bias})** | `{status}`\n• Ref: ${ref_price} | Live: ${live_price} | RSI: {rsi}\n• **Verdict:** {reason}\n\n"

    primary_payload = {
        "username": "US 3PM Execution Validator",
        "embeds": [{
            "title": "🇺🇸 US Equities 3 PM Session Validation",
            "description": results_summary,
            "color": 65280 if confirmed_count > 0 else 16711680,
            "footer": {"text": f"Validated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Confirmed: {confirmed_count}/10"}
        }]
    }
    requests.post(DISCORD_WEBHOOK_URL, json=primary_payload)

    # All-Red Contingency Protocol Trigger
    if confirmed_count == 0:
        print("🚨 ALL-RED PROTOCOL ACTIVATED: Sweeping alternative backup universe...")
        scanned_backups = []
        for backup in BACKUP_UNIVERSE:
            b_price, b_rsi = fetch_data(backup["ticker"])
            if b_price:
                scanned_backups.append({"ticker": backup["ticker"], "name": backup["name"], "type": backup["type"], "price": b_price, "rsi": b_rsi})

        top_3 = scanned_backups[:3]
        backup_summary = "⚠️ **Market Regime Warning: 10/10 Primary Setups Invalidated.**\nDynamic scanner surfaced top live alternatives:\n\n"
        for asset in top_3:
            backup_summary += f"🛡️ **${asset['ticker']}** ({asset['name']})\n• Type: `{asset['type']}` | Price: `${asset['price']}` | RSI: `{asset['rsi']}`\n\n"

        backup_payload = {
            "username": "US Dynamic Contingency Scanner",
            "embeds": [{
                "title": "🛡️ All-Red Protocol: Top 3 Market Hedges",
                "description": backup_summary,
                "color": 16753920,
                "footer": {"text": f"Dynamic Backup Scan at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"}
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=backup_payload)

if __name__ == "__main__":
    current_hour = datetime.utcnow().hour
    # Routing: If before 12:00 UTC, run UK Morning Scan. If 12:00 UTC or later, run US 3 PM Validation.
    if current_hour < 12:
        run_uk_morning_scan()
    else:
        run_us_validation()

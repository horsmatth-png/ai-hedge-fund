import os
from datetime import datetime
from dotenv import load_dotenv
import requests
import yfinance as yf

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
VALIDATION_VERSION = "v3.0-us-3pm-validation"

# Top 10 US Equities Watchlist with baseline morning estimates/references
US_TARGET_CANDIDATES = [
    {"ticker": "SPY", "morning_bias": "BULLISH", "morning_price": 580.00},
    {"ticker": "QQQ", "morning_bias": "BULLISH", "morning_price": 500.00},
    {"ticker": "AAPL", "morning_bias": "BULLISH", "morning_price": 230.00},
    {"ticker": "NVDA", "morning_bias": "BULLISH", "morning_price": 140.00},
    {"ticker": "MSFT", "morning_bias": "BULLISH", "morning_price": 420.00},
    {"ticker": "GOOGL", "morning_bias": "BEARISH", "morning_price": 175.00},
    {"ticker": "AMZN", "morning_bias": "BULLISH", "morning_price": 190.00},
    {"ticker": "META", "morning_bias": "BULLISH", "morning_price": 580.00},
    {"ticker": "AVGO", "morning_bias": "BULLISH", "morning_price": 180.00},
    {"ticker": "TSLA", "morning_bias": "BEARISH", "morning_price": 220.00},
]


def fetch_us_data(ticker: str):
  """Pulls live US price and 14-period RSI."""
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
      price = (
          getattr(fast, "last_price", None)
          or fast.get("lastPrice")
          or df["Close"].iloc[-1]
      )
      return round(float(price), 2), rsi
  except Exception as e:
    print(f"⚠️ Fetch error for {ticker}: {repr(e)}")
  return None, 50.0


def validate_trade_setup(
    ticker: str, bias: str, morning_price: float, live_price: float, rsi: float
):
  """Filters out opening-bell volatility traps."""
  if not live_price:
    return "DISCARD", "Unable to pull live price feed."

  pct_change = round(((live_price - morning_price) / morning_price) * 100, 2)

  if bias == "BEARISH":
    if rsi < 30:
      return (
          "INVALIDATED ❌",
          f"RSI oversold ({rsi}). High squeeze risk post-open.",
      )
    if live_price > (morning_price * 1.02):
      return (
          "INVALIDATED ❌",
          f"Opening bell broke resistance (+{pct_change}%). Setup failed.",
      )
    if live_price < (morning_price * 0.98):
      return (
          "EXPIRED / MISSED ⚠️",
          f"Target already hit (-{pct_change}%). Do not chase.",
      )
    return (
        "CONFIRMED ✅",
        f"Price (${live_price}, {pct_change}%) & RSI ({rsi}) intact. Ready for"
        " execution.",
    )

  if bias == "BULLISH":
    if rsi > 70:
      return (
          "INVALIDATED ❌",
          f"RSI overbought ({rsi}). High pullback risk.",
      )
    if live_price < (morning_price * 0.98):
      return (
          "INVALIDATED ❌",
          f"Opening bell broke support ({pct_change}%). Setup failed.",
      )
    if live_price > (morning_price * 1.02):
      return (
          "EXPIRED / MISSED ⚠️",
          f"Move already triggered (+{pct_change}%). Do not chase.",
      )
    return (
        "CONFIRMED ✅",
        f"Price (${live_price}, {pct_change}%) & RSI ({rsi}) intact. Ready for"
        " execution.",
    )

  return "NEUTRAL", "No clear bias."


def run_us_validation():
  current_hour = datetime.utcnow().hour
  print("==================================================")
  print(f"  🇺🇸 RUNNING US 3 PM EXECUTION VALIDATION [{VALIDATION_VERSION}]")
  print(f"  🕒 Current UTC Hour: {current_hour}")
  print("==================================================")

  results_summary = ""

  for item in US_TARGET_CANDIDATES:
    ticker = item["ticker"]
    bias = item["morning_bias"]
    m_price = item["morning_price"]

    live_price, rsi = fetch_us_data(ticker)
    status, reason = validate_trade_setup(ticker, bias, m_price, live_price, rsi)

    print(f"[{status}] ${ticker} ({bias}) -> {reason}")

    results_summary += (
        f"**${ticker} ({bias})** | `{status}`\n"
        f"• Ref: ${m_price} | Live: ${live_price} | RSI: {rsi}\n"
        f"• **Verdict:** {reason}\n\n"
    )

  title_suffix = (
      " (Manual Run)" if (current_hour < 12 and current_hour >= 8) else ""
  )

  payload = {
      "username": "US 3PM Execution Validator",
      "embeds": [{
          "title": f"🇺🇸 Top 10 US Equities 3 PM Verdicts{title_suffix}",
          "description": results_summary,
          "color": 65280 if "CONFIRMED ✅" in results_summary else 16711680,
          "footer": {
              "text": f"Validated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
          },
      }],
  }
  requests.post(DISCORD_WEBHOOK_URL, json=payload)
  print("\n--> US 3 PM Validation Verdicts sent to Discord successfully!")


if __name__ == "__main__":
  run_us_validation()

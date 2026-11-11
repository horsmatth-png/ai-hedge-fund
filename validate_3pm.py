import os
import time
from datetime import datetime
from dotenv import load_dotenv
import requests
import yfinance as yf

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
VALIDATION_VERSION = "v2.1-smart-validation"

# Dynamic candidate list loaded from morning baseline
TARGET_CANDIDATES = [
    {"ticker": "NVDA", "morning_bias": "BEARISH", "morning_price": 173.90},
    {"ticker": "TSLA", "morning_bias": "BEARISH", "morning_price": 385.00},
    {"ticker": "QQQ", "morning_bias": "BEARISH", "morning_price": 500.00},
    {"ticker": "AMD", "morning_bias": "BULLISH", "morning_price": 160.00},
    {"ticker": "JPM", "morning_bias": "BULLISH", "morning_price": 220.00},
]


def fetch_live_data(ticker: str):
  """Pulls live price and 14-period RSI."""
  try:
    t = yf.Ticker(ticker)
    df = t.history(period="1mo", interval="1d")
    if not df.empty and len(df) >= 15:
      delta = df["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      latest_rsi = round(float((100 - (100 / (1 + rs))).iloc[-1]), 1)

      fast = t.fast_info
      latest_price = (
          getattr(fast, "last_price", None)
          or fast.get("lastPrice")
          or df["Close"].iloc[-1]
      )
      return round(float(latest_price), 2), latest_rsi
  except Exception as e:
    print(f"⚠️ Fetch error for {ticker}: {repr(e)}")
  return None, 50.0


def validate_trade_setup(
    ticker: str, bias: str, morning_price: float, live_price: float, rsi: float
):
  """Validates if setup survived the opening bell volatility sweep."""
  if not live_price:
    return "DISCARD", "Unable to pull live price feed."

  pct_change = round(((live_price - morning_price) / morning_price) * 100, 2)

  # BEARISH / SHORT RULES
  if bias == "BEARISH":
    if rsi < 30:
      return (
          "INVALIDATED ❌",
          f"RSI is oversold ({rsi}). High risk of short-squeeze bounce.",
      )
    if live_price > (morning_price * 1.02):
      return (
          "INVALIDATED ❌",
          f"Opening bell swept structural resistance (+{pct_change}% higher)."
          " Stop would have been hit.",
      )
    if live_price < (morning_price * 0.98):
      return (
          "EXPIRED / MISSED ⚠️",
          f"Target already reached (-{pct_change}% lower). Do not chase.",
      )
    return (
        "CONFIRMED ✅",
        f"Price (${live_price}, {pct_change}%) and RSI ({rsi}) intact. Safe to"
        " execute short on IG.",
    )

  # BULLISH / LONG RULES
  if bias == "BULLISH":
    if rsi > 70:
      return (
          "INVALIDATED ❌",
          f"RSI is overbought ({rsi}). High risk of pullback.",
      )
    if live_price < (morning_price * 0.98):
      return (
          "INVALIDATED ❌",
          f"Opening bell swept support ({pct_change}% lower). Setup broken.",
      )
    if live_price > (morning_price * 1.02):
      return (
          "EXPIRED / MISSED ⚠️",
          f"Move already triggered (+{pct_change}% higher). Do not chase.",
      )
    return (
        "CONFIRMED ✅",
        f"Price (${live_price}, {pct_change}%) and RSI ({rsi}) intact. Safe to"
        " execute long on IG.",
    )

  return "NEUTRAL", "No clear bias."


def run_validation():
  current_hour = datetime.utcnow().hour
  print("\n==================================================")
  print(f"  🤖 RUNNING EXECUTION VALIDATION [{VALIDATION_VERSION}]")
  print(f"  🕒 Current UTC Hour: {current_hour}")
  print("==================================================\n")

  results_summary = ""

  for item in TARGET_CANDIDATES:
    ticker = item["ticker"]
    bias = item["morning_bias"]
    m_price = item["morning_price"]

    live_price, rsi = fetch_live_data(ticker)
    status, reason = validate_trade_setup(ticker, bias, m_price, live_price, rsi)

    print(f"[{status}] ${ticker} ({bias}) -> {reason}")

    results_summary += (
        f"**${ticker} ({bias})** | `{status}`\n"
        f"• Reference: ${m_price} | Live Price: ${live_price} | RSI: {rsi}\n"
        f"• **Verdict:** {reason}\n\n"
    )

  # Append manual run notice if triggered outside standard 3 PM slot
  title_suffix = (
      " (Manual Run)" if (current_hour < 12 and current_hour >= 9) else ""
  )

  payload = {
      "username": "AI Quant Execution Engine",
      "embeds": [{
          "title": f"🎯 Trade Validation Verdicts{title_suffix}",
          "description": results_summary,
          "color": 65280 if "CONFIRMED ✅" in results_summary else 16711680,
          "footer": {
              "text": (
                  f"Validated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
              )
          },
      }],
  }
  requests.post(DISCORD_WEBHOOK_URL, json=payload)
  print("\n--> Validation Verdicts sent to Discord successfully!")


if __name__ == "__main__":
  run_validation()

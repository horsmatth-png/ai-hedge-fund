import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dotenv import load_dotenv
from edgar import Company, set_identity
from google import genai
from groq import Groq
import requests
import yfinance as yf

load_dotenv()

# Set identity for SEC EDGAR rate limits
set_identity("QuantTrader research@example.com")

# Pipeline Version Tag (Forces git diff change)
PIPELINE_VERSION = "v2.0-morning-prep"

# Environment Keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Multi-Asset Watchlist: Indexes + Top US Equities
WATCHLIST = [
    "SPY",
    "QQQ",
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "LLY",
    "WMT",
    "JPM",
    "MU",
    "AMD",
    "V",
    "XOM",
]


def fetch_live_price_and_rsi(ticker: str):
  """Pulls real-time price and calculates 14-period RSI using yfinance history."""
  try:
    t = yf.Ticker(ticker)
    df = t.history(period="1mo", interval="1d")

    if not df.empty and len(df) >= 15:
      delta = df["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      rsi_series = 100 - (100 / (1 + rs))
      latest_rsi = round(float(rsi_series.iloc[-1]), 1)

      fast = t.fast_info
      latest_price = (
          getattr(fast, "last_price", None)
          or fast.get("lastPrice")
          or df["Close"].iloc[-1]
      )
      return (
          round(float(latest_price), 2),
          latest_rsi,
          "yfinance Consolidated + RSI",
      )
  except Exception as e:
    print(f"⚠️ yfinance fetch/RSI error for {ticker}: {repr(e)}")

  try:
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
    res = requests.get(url, timeout=5)
    if res.status_code == 200:
      data = res.json()
      current_price = data.get("c")
      if current_price and current_price > 0:
        return round(float(current_price), 2), 50.0, "Finnhub Quote (No RSI)"
  except Exception as e:
    print(f"⚠️ Finnhub fallback error for {ticker}: {repr(e)}")

  return None, 50.0, "Unavailable"


def fetch_enhanced_news(ticker: str) -> str:
  """Combines Finnhub news with Google News RSS."""
  collected_headlines = []

  today = datetime.now().strftime("%Y-%m-%d")
  start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
  url_ticker = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={today}&token={FINNHUB_API_KEY}"

  try:
    res = requests.get(url_ticker, timeout=5)
    if res.status_code == 200:
      articles = res.json()
      for item in articles[:3]:
        headline = item.get("headline", "")
        summary = item.get("summary", "")[:100]
        if headline:
          collected_headlines.append(f"• [Finnhub] {headline}: {summary}...")
  except Exception as e:
    print(f"⚠️ Finnhub News Error for {ticker}: {repr(e)}")

  try:
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:2d&hl=en-US&gl=US&ceid=US:en"
    rss_res = requests.get(rss_url, timeout=5)
    if rss_res.status_code == 200:
      root = ET.fromstring(rss_res.content)
      for item in root.findall(".//item")[:3]:
        title = item.find("title").text if item.find("title") is not None else ""
        if title:
          collected_headlines.append(f"• [Breaking Catalyst] {title}")
  except Exception as e:
    print(f"⚠️ RSS News Fetch Error for {ticker}: {repr(e)}")

  if collected_headlines:
    return "\n".join(collected_headlines)

  return "No recent breaking news retrieved."


def fetch_sec_form4(ticker: str):
  """Pulls recent SEC Form 4 insider transactions."""
  if ticker in ["SPY", "QQQ"]:
    return "N/A (Index ETF)"
  try:
    filings = Company(ticker).get_filings(form="4")
    if filings and len(filings) > 0:
      f4 = filings[0].obj()
      summary = f4.get_ownership_summary()
      return f"Insider: {summary.insider_name} | Action: {summary.primary_activity} | Net Value: ${summary.net_value:,.0f}"
  except Exception as e:
    print(f"⚠️ SEC Edgar Error for {ticker}: {repr(e)}")
  return "No recent SEC Form 4 insider filings."


def analyze_with_fallback(prompt_text: str):
  """Sends prompt to Gemini 2.5 Flash with Groq fallback."""
  try:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    response = ai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt_text
    )
    if response and response.text:
      return response.text, "Gemini 2.5 Flash"
  except Exception as e:
    print(f"⚠️ Gemini error: {repr(e)}. Switching to Groq fallback...")

  try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert quantitative market analyst evaluating"
                    " risk and momentum."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content, "Groq (openai/gpt-oss-20b)"
  except Exception as e:
    return f"Error: Both engines failed. Details: {repr(e)}", "None"


def extract_confidence(report_text: str) -> int:
  """Scans LLM output text for confidence percentages."""
  match = re.search(
      r"(?:confidence|score)[:\s*]+(\d{1,3})\s*%?", report_text, re.IGNORECASE
  )
  if match:
    val = int(match.group(1))
    return min(val, 100)

  fallback_match = re.search(r"(\d{1,3})\s*%", report_text)
  if fallback_match:
    return min(int(fallback_match.group(1)), 100)

  return 50


def run_pipeline():
  print(f"🚀 Initializing Morning Prep Scan [{PIPELINE_VERSION}]")
  scanned_results = []

  for ticker in WATCHLIST:
    print(f"\n--> [08:15 Scan] Ingesting data for ${ticker}...")

    price, rsi, price_source = fetch_live_price_and_rsi(ticker)
    news_summary = fetch_enhanced_news(ticker)
    insider_summary = fetch_sec_form4(ticker)

    prompt = f"""
        You are an institutional AI Quantitative Trading Assistant analyzing ticker ${ticker}.

        ### REAL-TIME & TECHNICAL FEEDS
        - Live Market Price: ${price if price else 'N/A'} (Source: {price_source})
        - 14-Day Relative Strength Index (RSI): {rsi}
        - SEC Form 4 Insider Sentiment: {insider_summary}
        - Breaking News & Catalysts:
        {news_summary}

        ### ANALYSIS TASK & CRITICAL RULES
        1. Your response MUST begin its first section with an explicit numerical confidence rating: "CONFIDENCE_SCORE: [XX]%" (e.g., CONFIDENCE_SCORE: 88% BEARISH or CONFIDENCE_SCORE: 85% BULLISH).
        2. TECHNICAL & RSI FILTER:
           - If RSI < 30 (Oversold), DO NOT give a high-confidence BEARISH / SHORT score (cap below 65%). Short-squeeze relief rallies are statistically probable. Flag as "Wait for Bounce / Retrace".
           - If RSI > 70 (Overbought), DO NOT give a high-confidence BULLISH / LONG score (cap below 65%).
        3. Actionable IG Trade Plan MUST explicitly output:
           - Target Entry Zone ($)
           - Structural Invalidation Stop-Loss ($)
           - IG Stop Distance (where $1.00 move = 100 IG points)
           - IG Limit Distance (where $1.00 move = 100 IG points)

        ### OUTPUT FORMAT
        1. **Confidence Score & Bias:** (e.g., 88% BEARISH)
        2. **Technical & News Confluence:** Synthesize RSI levels, catalysts, and momentum.
        3. **Actionable IG Trade Plan:** Specific entry zone, invalidation stop, and target distances in IG points.
        """

    report_text, engine_used = analyze_with_fallback(prompt)
    confidence = extract_confidence(report_text)

    scanned_results.append({
        "ticker": ticker,
        "confidence": confidence,
        "report": report_text,
        "engine": engine_used,
        "price": price,
        "rsi": rsi,
    })

    time.sleep(1.5)

  scanned_results.sort(key=lambda x: x["confidence"], reverse=True)
  top_5 = scanned_results[:5]

  print("\n--> Dispatching Morning Watchlist to Discord...")

  shortlist_description = (
      "### 📋 08:15 AM Morning Watchlist (Pending 15:00 Validation)\n"
      + "⚠️ **Do NOT place active orders yet.** Setups will be re-validated at"
      + " 15:00 UK to filter out pre-market sweeps.\n\n"
  )

  for i, item in enumerate(top_5, 1):
    shortlist_description += (
        f"**{i}. ${item['ticker']}** — `{item['confidence']}% Confidence` |"
        f" Price: ${item['price']} | RSI: {item['rsi']}\n"
    )

  payload = {
      "username": "AI Quant Engine (08:15 Pre-Market)",
      "embeds": [{
          "title": "🌅 Morning Watchlist Generated",
          "description": shortlist_description,
          "color": 3447003,
          "footer": {
              "text": (
                  f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC',"
                  " time.gmtime())}"
              )
          },
      }],
  }
  requests.post(DISCORD_WEBHOOK_URL, json=payload)

  for item in top_5:
    detail_payload = {
        "username": "AI Quant Engine (08:15 Pre-Market)",
        "embeds": [{
            "title": (
                f"📊 Morning Setup: ${item['ticker']} ({item['confidence']}%"
                " Conf.)"
            ),
            "description": (
                f"{item['report']}\n\n*🤖 Engine: **{item['engine']}***"
            ),
            "color": 5763719,
        }],
    }
    requests.post(DISCORD_WEBHOOK_URL, json=detail_payload)
    time.sleep(1)


if __name__ == "__main__":
  run_pipeline()

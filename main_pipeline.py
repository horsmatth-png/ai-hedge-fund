import os
import re
import time
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from edgar import Company, set_identity
from google import genai
from groq import Groq
import requests

load_dotenv()

set_identity("QuantTrader research@example.com")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Comprehensive Watchlist: Major Indexes + Top 15 US Market-Cap Leaders
WATCHLIST = [
    "SPY",
    "QQQ",  # Major Market Indexes
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",  # Mega-Cap Tech & AI
    "META",
    "AVGO",
    "TSLA",
    "LLY",
    "WMT",  # Growth & Consumer Giants
    "JPM",
    "MU",
    "AMD",
    "V",
    "XOM",  # Financials, Semiconductors & Energy
]


def fetch_alpaca_data(ticker: str):
  try:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    start_date = datetime.now() - timedelta(days=5)
    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start_date,
        limit=15,
    )
    bars = client.get_stock_bars(request_params)

    if ticker in bars and len(bars[ticker]) > 0:
      recent_bars = bars[ticker]
      latest_close = recent_bars[-1].close
      volume_sum = sum([b.volume for b in recent_bars])
      return latest_close, volume_sum
  except Exception as e:
    print(f"⚠️ Alpaca Fetch Error for {ticker}: {repr(e)}")
  return None, None


def fetch_finnhub_news(ticker: str):
  today = datetime.now().strftime("%Y-%m-%d")
  start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
  url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={today}&token={FINNHUB_API_KEY}"

  try:
    res = requests.get(url)
    if res.status_code == 200:
      articles = res.json()
      if articles:
        headlines = []
        for item in articles[:3]:
          headlines.append(
              f"• {item.get('headline', '')}: {item.get('summary', '')[:100]}..."
          )
        return "\n".join(headlines)
  except Exception as e:
    print(f"⚠️ Finnhub Fetch Error for {ticker}: {repr(e)}")
  return "No recent news retrieved."


def fetch_sec_form4(ticker: str):
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
  # 1. Try Gemini Primary
  try:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    response = ai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt_text
    )
    if response and response.text:
      return response.text, "Gemini 2.5 Flash"
  except Exception as e:
    print(f"⚠️ Gemini error: {repr(e)}. Switching to Groq fallback...")

  # 2. Fallback to Groq
  try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert quantitative market analyst.",
            },
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content, "Groq (openai/gpt-oss-20b)"
  except Exception as e:
    return f"Error: Both engines failed. Details: {repr(e)}", "None"


def extract_confidence(report_text: str) -> int:
  """Scans the LLM text output for a percentage confidence score."""
  match = re.search(r"(\d{1,3})\s*%", report_text)
  if match:
    val = int(match.group(1))
    return min(val, 100)
  return 50  # Default baseline if missing


def run_pipeline():
  scanned_results = []

  for ticker in WATCHLIST:
    print(f"\n--> Processing multi-source scan for ${ticker}...")

    price, volume = fetch_alpaca_data(ticker)
    news_summary = fetch_finnhub_news(ticker)
    insider_summary = fetch_sec_form4(ticker)

    prompt = f"""
        You are an institutional AI Quantitative Trading Assistant analyzing ticker ${ticker}.

        ### MARKET DATA & FEEDS
        - Latest Price Feed: ${price if price else 'N/A'}
        - Recent Volume Profile: {volume if volume else 'N/A'}
        - SEC Form 4 Insider Sentiment: {insider_summary}
        - Recent News & Macro Catalysts:
        {news_summary}

        ### ANALYSIS TASK
        Provide a concise trading breakdown. YOU MUST include an explicit confidence percentage score in your first line (e.g., "Confidence Score: 88% - BULLISH").
        1. **Confidence Score & Bias:** (e.g., 85% BULLISH / 70% BEARISH / NEUTRAL)
        2. **Order Book / Flow Confluence:** Assess price action against broader market momentum (SPY/QQQ) and headline flow.
        3. **Actionable Trade Plan:** Entry zone, strict stop-loss/invalidation level, and profit targets.
        """

    report_text, engine_used = analyze_with_fallback(prompt)
    confidence = extract_confidence(report_text)

    scanned_results.append({
        "ticker": ticker,
        "confidence": confidence,
        "report": report_text,
        "engine": engine_used,
    })

    time.sleep(1.5)

  # Sort all results by confidence score descending and slice top 5
  scanned_results.sort(key=lambda x: x["confidence"], reverse=True)
  top_5 = scanned_results[:5]

  print("\n--> Generating and dispatching Top 5 Shortlist to Discord...")

  shortlist_description = (
      "### 🏆 Top 5 Ranked High-Confidence Setups\n"
      + "Synthesized across major indices and top US equities:\n\n"
  )

  for i, item in enumerate(top_5, 1):
    shortlist_description += (
        f"**{i}. ${item['ticker']}** — `{item['confidence']}% Confidence`\n"
    )

  payload = {
      "username": "AI Quant Engine",
      "embeds": [
          {
              "title": "🚨 Top 5 Quant Trading Shortlist",
              "description": shortlist_description,
              "color": 3447003,
              "footer": {
                  "text": f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
              },
          }
      ],
  }

  requests.post(DISCORD_WEBHOOK_URL, json=payload)

  for item in top_5:
    detail_payload = {
        "username": "AI Quant Engine",
        "embeds": [{
            "title": (
                f"📊 Setup Details: ${item['ticker']}"
                f" ({item['confidence']} Conf.)"
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

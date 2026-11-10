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

# Environment Keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Multi-Asset Watchlist: Indexes + Top US Equities
WATCHLIST = [
    "SPY",
    "QQQ",  # Broad Market Indexes
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",  # Tech & AI Leaders
    "META",
    "AVGO",
    "TSLA",
    "LLY",
    "WMT",  # Growth & Staples
    "JPM",
    "MU",
    "AMD",
    "V",
    "XOM",  # Financials, Semiconductors & Energy
]


def fetch_live_price(ticker: str):
  """Pulls real-time consolidated market price (regular + extended hours) via yfinance with Finnhub fallback."""
  # Primary: yfinance consolidated ticker data
  try:
    t = yf.Ticker(ticker)
    # fast_info provides real-time / post-market price
    fast = t.fast_info
    price = fast.get("lastPrice") or fast.get("regularMarketPreviousClose")
    if price and price > 0:
      return round(float(price), 2), "yfinance Consolidated"
  except Exception as e:
    print(f"⚠️ yfinance Fetch Error for {ticker}: {repr(e)}")

  # Fallback: Finnhub Real-Time Quote API
  try:
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
    res = requests.get(url, timeout=5)
    if res.status_code == 200:
      data = res.json()
      current_price = data.get("c")  # Current / Close Price
      if current_price and current_price > 0:
        return round(float(current_price), 2), "Finnhub Quote"
  except Exception as e:
    print(f"⚠️ Finnhub Fallback Error for {ticker}: {repr(e)}")

  return None, "Unavailable"


def fetch_enhanced_news(ticker: str) -> str:
  """Combines Finnhub company news with Google News RSS to capture front-page

  governance scandals, executive risks, and macro catalysts.
  """
  collected_headlines = []

  # Source 1: Finnhub Company News
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

  # Source 2: Google News RSS Search (Captures High-Impact Breaking Scandals/Probes)
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
  """Pulls recent SEC Form 4 insider buying/selling transactions."""
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
  """Scans LLM output text for confidence percentages across various formats."""
  match = re.search(
      r"(?:confidence|score)[:\s*]+(\d{1,3})\s*%?", report_text, re.IGNORECASE
  )
  if match:
    val = int(match.group(1))
    return min(val, 100)

  fallback_match = re.search(r"(\d{1,3})\s*%", report_text)
  if fallback_match:
    return min(int(fallback_match.group(1)), 100)

  return 50  # Default baseline if unparseable


def run_pipeline():
  scanned_results = []

  for ticker in WATCHLIST:
    print(f"\n--> Ingesting real-time data feeds for ${ticker}...")

    price, price_source = fetch_live_price(ticker)
    news_summary = fetch_enhanced_news(ticker)
    insider_summary = fetch_sec_form4(ticker)

    prompt = f"""
        You are an institutional AI Quantitative Trading Assistant analyzing ticker ${ticker}.

        ### REAL-TIME & STRUCTURAL FEEDS
        - Live Market/Close Price: ${price if price else 'N/A'} (Source: {price_source})
        - SEC Form 4 Insider Sentiment: {insider_summary}
        - Breaking News, Scandals & Geopolitical Catalysts:
        {news_summary}

        ### ANALYSIS TASK
        Synthesize these signals into a concise quantitative trade breakdown. 
        CRITICAL INSTRUCTIONS:
        1. Your response MUST begin its first section with an explicit numerical confidence rating in this format: "CONFIDENCE_SCORE: [XX]%" (e.g., CONFIDENCE_SCORE: 88% BEARISH or CONFIDENCE_SCORE: 85% BULLISH).
        2. Evaluate executive governance risks, legal probes, regulatory investigations, and headline scandals with FULL weight. Do not shy away from giving BEARISH / SHORT setups high confidence scores (80%+) when downside momentum and negative catalysts align.

        1. **Confidence Score & Bias:** (e.g., 88% BEARISH / 85% BULLISH)
        2. **Order Book / Flow Confluence:** Combine technical price action with news catalysts and macro/sector momentum (SPY/QQQ).
        3. **Actionable Trade Plan:** Specific entry setup, strict stop-loss/invalidation level, and profit targets.
        """

    report_text, engine_used = analyze_with_fallback(prompt)
    confidence = extract_confidence(report_text)

    scanned_results.append({
        "ticker": ticker,
        "confidence": confidence,
        "report": report_text,
        "engine": engine_used,
    })

    time.sleep(1.5)  # Throttle to respect API rate limits

  # Sort by confidence descending and select Top 5
  scanned_results.sort(key=lambda x: x["confidence"], reverse=True)
  top_5 = scanned_results[:5]

  print("\n--> Dispatching Top 5 Ranked Shortlist to Discord...")

  shortlist_description = (
      "### 🏆 Top 5 Ranked High-Confidence Setups\n"
      + "Synthesized across major indices, mega-cap tech, and market leaders:\n\n"
  )

  for i, item in enumerate(top_5, 1):
    shortlist_description += (
        f"**{i}. ${item['ticker']}** — `{item['confidence']}% Confidence`\n"
    )

  # Post Primary Shortlist Summary
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

  # Post Individual Detailed Embeds for the Top 5
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

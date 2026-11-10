import os
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


def fetch_alpaca_data(ticker: str):
  try:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    start_date = datetime.now() - timedelta(days=5)
    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start_date,
        limit=10,
    )
    bars = client.get_stock_bars(request_params)

    if ticker in bars and len(bars[ticker]) > 0:
      recent_bars = bars[ticker]
      latest_close = recent_bars[-1].close
      volume_sum = sum([b.volume for b in recent_bars])
      return latest_close, volume_sum
  except Exception as e:
    print(f"⚠️ Alpaca Fetch Error: {repr(e)}")
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
              f"• {item.get('headline', '')}: {item.get('summary', '')[:120]}..."
          )
        return "\n".join(headlines)
  except Exception as e:
    print(f"⚠️ Finnhub Fetch Error: {repr(e)}")
  return "No recent news retrieved."


def fetch_sec_form4(ticker: str):
  try:
    filings = Company(ticker).get_filings(form="4")
    if filings and len(filings) > 0:
      f4 = filings[0].obj()
      summary = f4.get_ownership_summary()
      return f"Insider: {summary.insider_name} | Action: {summary.primary_activity} | Net Value: ${summary.net_value:,.0f}"
  except Exception as e:
    print(f"⚠️ SEC Edgar Error: {repr(e)}")
  return "No recent SEC Form 4 insider filings."


def analyze_with_fallback(prompt_text: str):
  # 1. Try Gemini Primary with explicit API key
  try:
    print("--> Attempting analysis with Gemini 2.5 Flash...")
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    response = ai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt_text
    )
    if response and response.text:
      return response.text, "Gemini 2.5 Flash"
  except Exception as e:
    print(f"⚠️ Gemini error: {repr(e)}. Switching to Groq fallback...")

  # 2. Fallback to Groq using the current valid production model ID
  try:
    print("--> Attempting analysis with Groq (openai/gpt-oss-20b)...")
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
    print(f"❌ Groq fallback also failed: {repr(e)}")
    return f"Error: Both engines failed. Details: {repr(e)}", "None"


def execute_pipeline(ticker: str):
  print(f"--> Ingesting multi-source data feeds for ${ticker}...")

  price, volume = fetch_alpaca_data(ticker)
  news_summary = fetch_finnhub_news(ticker)
  insider_summary = fetch_sec_form4(ticker)

  prompt = f"""
    You are an AI Trading Assistant synthesizing pre-market and intraday market data for ticker ${ticker}.

    ### REAL-TIME & STRUCTURAL FEEDS
    - Latest Price (Alpaca IEX): ${price if price else 'N/A'}
    - 10-Min Volume (IEX): {volume if volume else 'N/A'}
    - SEC Form 4 Insider Bias: {insider_summary}
    - Recent Breaking News (Finnhub):
    {news_summary}

    ### ANALYSIS TASK
    Synthesize these signals and format a concise report:
    1. **Overall Bias:** (BULLISH / BEARISH / NEUTRAL)
    2. **Signal Confluence:** Combine insider activity with news catalysts and price action.
    3. **Key Levels & Trade Plan:** Direct entry setup, invalidation/stop-loss level, and initial target.
    """

  report_text, engine_used = analyze_with_fallback(prompt)
  return report_text, engine_used


def send_discord_webhook(ticker: str, report_text: str, engine_used: str):
  payload = {
      "username": "AI Quant Engine",
      "embeds": [{
          "title": f"🚨 Market Signal Report: ${ticker}",
          "description": (
              f"{report_text}\n\n*🤖 Analyzed via: **{engine_used}***"
          ),
          "color": 3447003,
          "footer": {
              "text": (
                  f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
              )
          },
      }],
  }

  res = requests.post(DISCORD_WEBHOOK_URL, json=payload)
  if res.status_code in [200, 204]:
    print("--> Webhook successfully dispatched to Discord!")
  else:
    print(f"--> Failed to send Webhook: Status {res.status_code}")


if __name__ == "__main__":
  target_ticker = "NVDA"
  report, engine = execute_pipeline(target_ticker)
  send_discord_webhook(target_ticker, report, engine)

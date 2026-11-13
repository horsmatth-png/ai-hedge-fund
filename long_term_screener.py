import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import yfinance as yf

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

# Broad multi-sector universe for long-term fundamental & valuation screening
LONG_TERM_UNIVERSE = [
    # US Tech & Semiconductors
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "AMD", "QCOM", "CRM", "ADBE",
    # US Healthcare, Financials, Industrials & Consumer
    "UNH", "LLY", "JNJ", "JPM", "V", "MA", "BRK-B", "CAT", "GE", "COST", "WMT", "PG",
    # UK FTSE 100 Core Compounders
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "REL.L", "RIO.L", "GSK.L", "DGE.L"
]


def fetch_finnhub_fundamentals_and_news(ticker: str):
    """Pulls basic financials (P/E, P/S, margins), 7-day news, and insider activity from Finnhub."""
    clean_symbol = ticker.replace(".L", "")
    fh_metrics = {}
    headlines = []
    insider_summary = "No recent major insider filings."

    if not FINNHUB_API_KEY:
        return fh_metrics, "Finnhub API Key missing.", insider_summary

    try:
        # 1. Finnhub Basic Financials (P/E, P/S, ROE, Revenue Growth)
        metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={clean_symbol}&metric=all&token={FINNHUB_API_KEY}"
        m_res = requests.get(metrics_url, timeout=8)
        if m_res.status_code == 200:
            fh_metrics = m_res.json().get("metric", {}) or {}

        # 2. 7-Day Company News for Macro & Thesis Catalysts
        today = datetime.utcnow().strftime("%Y-%m-%d")
        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={week_ago}&to={today}&token={FINNHUB_API_KEY}"
        n_res = requests.get(news_url, timeout=8)
        if n_res.status_code == 200:
            for art in n_res.json()[:5]:
                if art.get("headline"):
                    headlines.append(art["headline"])

        # 3. Insider Transactions (US Equities)
        if not ticker.endswith(".L"):
            ins_url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
            i_res = requests.get(ins_url, timeout=8)
            if i_res.status_code == 200:
                txs = i_res.json().get("data", [])[:3]
                tx_list = [f"{tx.get('name')} ({tx.get('change'):,} shs)" for tx in txs if tx.get("name")]
                if tx_list:
                    insider_summary = "; ".join(tx_list)
    except Exception as e:
        print(f"⚠️ Finnhub error for {ticker}: {repr(e)}")

    news_text = " | ".join(headlines) if headlines else "No major Finnhub headlines in last 7 days."
    return fh_metrics, news_text, insider_summary


def analyze_stock_fundamentals(ticker: str):
    """Combines yfinance valuation ratios + Finnhub metrics and computes an institutional GARP/Value score."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            hist = t.history(period="5d")
            if hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])

        fh_metrics, news_text, insider_summary = fetch_finnhub_fundamentals_and_news(ticker)

        # Valuation Multiples (yfinance primary, Finnhub fallback)
        trailing_pe = info.get("trailingPE") or fh_metrics.get("peBasicExclExtraTTM")
        forward_pe = info.get("forwardPE") or fh_metrics.get("peNormalizedAnnual")
        price_to_sales = info.get("priceToSalesTrailing12Months") or fh_metrics.get("psTTM")
        peg_ratio = info.get("pegRatio")

        # Quality & Growth Metrics
        rev_growth = info.get("revenueGrowth")  # decimal, e.g., 0.15 = 15%
        earnings_growth = info.get("earningsGrowth")
        profit_margin = info.get("profitMargins")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        div_yield = info.get("dividendYield")
        sector = info.get("sector", "Equities")
        name = info.get("shortName", ticker)

        # Skip companies without positive earnings or missing core multiples
        if not trailing_pe or trailing_pe <= 0 or not price_to_sales:
            return None

        # ==========================================
        # INSTITUTIONAL GARP & VALUE SCORING (0-100)
        # ==========================================
        score = 50

        # 1. Price-to-Earnings (P/E) & Forward Compression
        if trailing_pe < 18:
            score += 15
        elif trailing_pe < 26:
            score += 8
        elif trailing_pe > 45:
            score -= 12

        if forward_pe and forward_pe < trailing_pe:
            score += 8  # Earnings expected to expand over next 12M

        # 2. Price-to-Sales (P/S) Discipline
        if price_to_sales < 2.5:
            score += 12
        elif price_to_sales < 5.5:
            score += 6
        elif price_to_sales > 12.0:
            score -= 10

        # 3. Growth & Profitability Quality
        if rev_growth and rev_growth > 0.10:
            score += 10
        elif rev_growth and rev_growth < 0:
            score -= 8

        if profit_margin and profit_margin > 0.18:
            score += 8

        if peg_ratio and 0 < peg_ratio < 1.3:
            score += 7

        # 4. Governance / Event Shock Filter
        shock_words = ["resign", "investigation", "fraud", "subpoena", "sec probe", "accounting"]
        if any(w in news_text.lower() for w in shock_words):
            score -= 25

        score = max(0, min(100, score))

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "currency": "£" if ticker.endswith(".L") else "$",
            "price": round(float(price), 2),
            "trailing_pe": round(float(trailing_pe), 2),
            "forward_pe": round(float(forward_pe), 2) if forward_pe else "N/A",
            "price_to_sales": round(float(price_to_sales), 2),
            "peg_ratio": round(float(peg_ratio), 2) if peg_ratio else "N/A",
            "rev_growth_pct": f"{round(rev_growth * 100, 1)}%" if rev_growth is not None else "N/A",
            "margin_pct": f"{round(profit_margin * 100, 1)}%" if profit_margin is not None else "N/A",
            "debt_to_equity": round(float(debt_to_equity), 1) if debt_to_equity is not None else "N/A",
            "div_yield_pct": f"{round(div_yield * 100, 2)}%" if div_yield else "0.0%",
            "score": score,
            "news": news_text,
            "insider": insider_summary,
        }
    except Exception as e:
        print(f"⚠️ Fundamental scan error for {ticker}: {repr(e)}")
        return None


def generate_long_term_thesis(stock: dict):
    """Synthesizes P/E, P/S, growth, and live Finnhub catalysts into an institutional investment memo."""
    c = stock["currency"]
    p = stock["price"]
    acc_low = round(p * 0.94, 2)
    acc_high = round(p * 1.01, 2)
    target_12m = round(p * 1.22, 2)
    invalidation_sl = round(p * 0.85, 2)

    prompt = f"""
    You are an institutional long-term equity portfolio manager (12-to-36 month horizon).
    Evaluate {stock['name']} ({stock['ticker']}) in the {stock['sector']} sector.

    FUNDAMENTAL & VALUATION DATA:
    - Current Price: {c}{p}
    - Trailing P/E: {stock['trailing_pe']}x | Forward P/E: {stock['forward_pe']}x
    - Price-to-Sales (P/S): {stock['price_to_sales']}x | PEG Ratio: {stock['peg_ratio']}
    - YoY Revenue Growth: {stock['rev_growth_pct']} | Net Profit Margin: {stock['margin_pct']}
    - Debt-to-Equity: {stock['debt_to_equity']} | Dividend Yield: {stock['div_yield_pct']}
    - Institutional GARP/Value Score: {stock['score']}/100

    LIVE 7-DAY FINNHUB NEWS HEADLINES:
    "{stock['news']}"

    RECENT INSIDER ACTIVITY:
    "{stock['insider']}"

    Output a clean, concise markdown report using this exact layout:

    **1. Valuation & Fundamental Scorecard**
    | Metric | Reading | Institutional Assessment |
    | :--- | :--- | :--- |
    | **P/E (Trailing / Fwd)** | {stock['trailing_pe']}x / {stock['forward_pe']}x | [Assess earnings multiple & forward compression] |
    | **Price-to-Sales (P/S)** | {stock['price_to_sales']}x | [Assess revenue multiple relative to margin profile] |
    | **Growth & Margins** | Rev: {stock['rev_growth_pct']} | Margin: {stock['margin_pct']} | [Assess durability & moat] |
    | **Balance Sheet / Yield** | D/E: {stock['debt_to_equity']} | Yield: {stock['div_yield_pct']} | [Assess capital discipline] |
    | **News & Insider Flow** | [Summarize real Finnhub news/insiders] | [Assess long-term catalyst or governance risk] |

    **Long-Term Earnings Thesis:**
    [2-3 crisp sentences on why P/E, P/S, and earnings trajectory make this a compelling multi-quarter compounder.]

    ---

    **2. Long-Term Accumulation & Position Plan**
    | Plan Component | Target Level | Notes |
    | :--- | :--- | :--- |
    | **Accumulation Zone** | {c}{acc_low} - {c}{acc_high} | Staged tranches on pullbacks |
    | **Thesis Invalidation** | {c}{invalidation_sl} (-15%) | Fundamental/structural stop |
    | **12-18M Fair Value Target** | {c}{target_12m} (+22%) | Multiple expansion + EPS growth |
    """

    engine_used = "Unknown"
    response_text = ""

    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
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
                temperature=0.25,
                max_tokens=1500,
            )
            response_text = completion.choices[0].message.content
            engine_used = "Groq (openai/gpt-oss-20b)"
        except Exception as e:
            print(f"⚠️ Groq failed: {repr(e)}")

    if not response_text:
        response_text = f"**Valuation:** P/E {stock['trailing_pe']}x | P/S {stock['price_to_sales']}x | News: {stock['news']}"
        engine_used = "Offline / Fallback"

    response_text += f"\n\n🤖 *Engine: {engine_used} + Finnhub Fundamentals*"

    return {
        "username": "Institutional Fundamental Desk",
        "embeds": [{
            "title": f"🏛️ Long-Term Pick: {stock['name']} ({stock['ticker']}) — Score: {stock['score']}/100",
            "description": f"**VALUATION_PROFILE: P/E {stock['trailing_pe']}x | P/S {stock['price_to_sales']}x | Fwd P/E {stock['forward_pe']}x**\n\n---\n\n{response_text[:3900]}",
            "color": 3447003,  # Institutional Deep Blue
            "footer": {"text": f"Long-Term Fundamental Scan: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Engine: {engine_used}"}
        }]
    }


def run_long_term_pipeline():
    print("==================================================")
    print("  🏛️ RUNNING LONG-TERM FUNDAMENTAL & VALUATION SCAN")
    print("==================================================")

    candidates = []
    for ticker in LONG_TERM_UNIVERSE:
        print(f"🔍 Screening {ticker} fundamentals (P/E, P/S, Growth, Finnhub News)...")
        result = analyze_stock_fundamentals(ticker)
        if result:
            candidates.append(result)

    # Rank by highest composite GARP/Value score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_picks = candidates[:4]

    if not top_picks:
        print("⚠️ No candidates passed fundamental filters.")
        return

    for pick in top_picks:
        print(f"✅ Top Long-Term Pick: {pick['ticker']} (Score: {pick['score']}/100, P/E: {pick['trailing_pe']}, P/S: {pick['price_to_sales']})")
        embed_payload = generate_long_term_thesis(pick)
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=embed_payload)

    print(f"--> Long-term fundamental scan complete. {len(top_picks)} institutional picks dispatched to Discord.")


if __name__ == "__main__":
    run_long_term_pipeline()

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

LONG_TERM_UNIVERSE = [
    # US Tech & Semiconductors
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "AMD", "QCOM", "CRM", "ADBE",
    # US Healthcare, Financials, Industrials & Consumer
    "UNH", "LLY", "JNJ", "JPM", "V", "MA", "BRK-B", "CAT", "GE", "COST", "WMT", "PG",
    # UK FTSE 100 Core Compounders
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "REL.L", "RIO.L", "GSK.L", "DGE.L"
]


def fetch_finnhub_fundamentals_and_news(ticker: str):
    """Pulls Finnhub basic financials, analyst recommendation trends, 7-day news, and insider activity."""
    clean_symbol = ticker.replace(".L", "")
    fh_metrics = {}
    headlines = []
    insider_summary = "No recent major insider filings."
    rec_summary = "No Finnhub consensus data."

    if not FINNHUB_API_KEY:
        return fh_metrics, "Finnhub API Key missing.", insider_summary, rec_summary

    try:
        # 1. Finnhub Basic Financials
        metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={clean_symbol}&metric=all&token={FINNHUB_API_KEY}"
        m_res = requests.get(metrics_url, timeout=8)
        if m_res.status_code == 200:
            fh_metrics = m_res.json().get("metric", {}) or {}

        # 2. Finnhub Wall Street Recommendation Trends (Forward-Looking Consensus)
        if not ticker.endswith(".L"):
            rec_url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
            r_res = requests.get(rec_url, timeout=8)
            if r_res.status_code == 200 and r_res.json():
                latest = r_res.json()[0]
                rec_summary = (
                    f"StrongBuy:{latest.get('strongBuy', 0)} | Buy:{latest.get('buy', 0)} | "
                    f"Hold:{latest.get('hold', 0)} | Sell:{latest.get('sell', 0) + latest.get('strongSell', 0)}"
                )

        # 3. 7-Day Company News
        today = datetime.utcnow().strftime("%Y-%m-%d")
        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={week_ago}&to={today}&token={FINNHUB_API_KEY}"
        n_res = requests.get(news_url, timeout=8)
        if n_res.status_code == 200:
            for art in n_res.json()[:6]:
                if art.get("headline"):
                    headlines.append(art["headline"])

        # 4. Insider Transactions
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
    return fh_metrics, news_text, insider_summary, rec_summary


def analyze_stock_fundamentals(ticker: str):
    """Scores stocks on Forward Growth + Analyst Consensus + Valuation Discipline + Negative Catalyst Penalties."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            hist = t.history(period="5d")
            if hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])

        fh_metrics, news_text, insider_summary, rec_summary = fetch_finnhub_fundamentals_and_news(ticker)

        # Valuation Multiples
        trailing_pe = info.get("trailingPE") or fh_metrics.get("peBasicExclExtraTTM")
        forward_pe = info.get("forwardPE") or fh_metrics.get("peNormalizedAnnual")
        price_to_sales = info.get("priceToSalesTrailing12Months") or fh_metrics.get("psTTM")
        peg_ratio = info.get("pegRatio")

        # Forward-Looking Growth & Street Targets
        rev_growth = info.get("revenueGrowth")
        earnings_growth = info.get("earningsGrowth")
        profit_margin = info.get("profitMargins")
        debt_to_equity = info.get("debtToEquity")
        div_yield = info.get("dividendYield")
        target_mean = info.get("targetMeanPrice")
        rec_mean = info.get("recommendationMean")  # 1.0 = Strong Buy, 3.0 = Hold, 5.0 = Sell
        sector = info.get("sector", "Equities")
        name = info.get("shortName", ticker)

        if not trailing_pe or trailing_pe <= 0 or not price_to_sales:
            return None

        target_upside_pct = round(((target_mean - price) / price) * 100, 1) if target_mean else None

        # ==========================================
        # FORWARD-LOOKING GARP & CATALYST SCORE (0-100)
        # ==========================================
        score = 45

        # 1. Forward Growth & Earnings Trajectory (Weighted heavily over static cheapness)
        if rev_growth is not None:
            if rev_growth > 0.15:
                score += 14
            elif rev_growth > 0.08:
                score += 8
            elif rev_growth < 0:
                score -= 12

        if earnings_growth is not None:
            if earnings_growth > 0.15:
                score += 12
            elif earnings_growth < -0.05:
                score -= 10

        # 2. Forward Wall Street Consensus & Price Target Upside
        if target_upside_pct is not None:
            if target_upside_pct > 18:
                score += 12
            elif target_upside_pct > 8:
                score += 6
            elif target_upside_pct < 0:
                score -= 12  # Street sees stock as overvalued

        if rec_mean is not None:
            if rec_mean <= 2.0:
                score += 8   # Strong Buy / Buy consensus
            elif rec_mean >= 2.8:
                score -= 10  # Hold / Underperform drift

        # 3. Valuation Discipline (P/E, Forward Compression & P/S)
        if forward_pe and forward_pe < trailing_pe:
            score += 8  # Forward EPS expansion compressing multiple
        if trailing_pe < 22:
            score += 6
        elif trailing_pe > 50:
            score -= 8

        if price_to_sales < 4.5:
            score += 6
        elif price_to_sales > 14.0:
            score -= 8

        if peg_ratio and 0 < peg_ratio < 1.4:
            score += 7

        # 4. Strict Bearish News & Downgrade Filter (Fixes Sentiment Blindness)
        news_lower = news_text.lower()
        severe_shocks = ["resign", "investigation", "fraud", "subpoena", "sec probe", "accounting"]
        downgrade_words = [
            "downgrade", "price target cut", "cuts target", "lowers target",
            "underperform", "sell rating", "guidance cut", "weak outlook",
            "slowdown", "headwind", "disruption", "antitrust"
        ]

        if any(w in news_lower for w in severe_shocks):
            score -= 30
        if any(w in news_lower for w in downgrade_words):
            score -= 15  # Penalize active analyst downgrades & negative revisions

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
            "eps_growth_pct": f"{round(earnings_growth * 100, 1)}%" if earnings_growth is not None else "N/A",
            "margin_pct": f"{round(profit_margin * 100, 1)}%" if profit_margin is not None else "N/A",
            "debt_to_equity": round(float(debt_to_equity), 1) if debt_to_equity is not None else "N/A",
            "div_yield_pct": f"{round(div_yield * 100, 2)}%" if div_yield else "0.0%",
            "target_mean": round(float(target_mean), 2) if target_mean else "N/A",
            "target_upside": f"{target_upside_pct}%" if target_upside_pct is not None else "N/A",
            "rec_summary": rec_summary,
            "score": score,
            "news": news_text,
            "insider": insider_summary,
        }
    except Exception as e:
        print(f"⚠️ Fundamental scan error for {ticker}: {repr(e)}")
        return None


def generate_long_term_thesis(stock: dict):
    """Synthesizes forward catalysts, valuation multiples, and explicit bear-case risks."""
    c = stock["currency"]
    p = stock["price"]
    acc_low = round(p * 0.94, 2)
    acc_high = round(p * 1.01, 2)
    target_12m = stock["target_mean"] if stock["target_mean"] != "N/A" else round(p * 1.20, 2)
    invalidation_sl = round(p * 0.85, 2)

    prompt = f"""
    You are a skeptical institutional equity portfolio manager evaluating {stock['name']} ({stock['ticker']}) in {stock['sector']} for a 12-to-36 month horizon.

    FORWARD & FUNDAMENTAL METRICS:
    - Current Price: {c}{p} | Street Mean Target: {c}{stock['target_mean']} ({stock['target_upside']} implied upside)
    - Analyst Consensus: {stock['rec_summary']}
    - Trailing P/E: {stock['trailing_pe']}x | Forward P/E: {stock['forward_pe']}x | PEG: {stock['peg_ratio']}
    - Price-to-Sales (P/S): {stock['price_to_sales']}x
    - Revenue Growth: {stock['rev_growth_pct']} | EPS Growth: {stock['eps_growth_pct']} | Net Margin: {stock['margin_pct']}
    - Debt-to-Equity: {stock['debt_to_equity']} | Dividend Yield: {stock['div_yield_pct']}
    - Forward GARP Score: {stock['score']}/100

    LIVE 7-DAY FINNHUB NEWS HEADLINES:
    "{stock['news']}"

    RECENT INSIDER ACTIVITY:
    "{stock['insider']}"

    CRITICAL INSTRUCTIONS:
    1. Do NOT spin analyst downgrades, price-target cuts, or cyclical peaks as positives! Report negative headlines honestly.
    2. Balance backward valuation (P/E, P/S) with forward runway (secular tailwinds, product cycle, Street target upside).
    3. Explicitly highlight the Bear Case / Value-Trap Risk (e.g., AI disruption, cyclical commodity peak, margin compression).

    Use this exact concise markdown structure:

    **1. Forward Opportunity & Valuation Matrix**
    | Factor | Data / Reading | Institutional Assessment |
    | :--- | :--- | :--- |
    | **Valuation (P/E & P/S)** | P/E: {stock['trailing_pe']}x (Fwd: {stock['forward_pe']}x) | P/S: {stock['price_to_sales']}x | [Cheap for a reason or genuine GARP discount?] |
    | **Forward Growth Runway** | Rev: {stock['rev_growth_pct']} | EPS: {stock['eps_growth_pct']} | [Assess 12-24M secular expansion & moat] |
    | **Wall Street Consensus** | Target: {c}{stock['target_mean']} ({stock['target_upside']}) | {stock['rec_summary']} | [Assess institutional conviction] |
    | **Live News & Catalysts** | [Summarize real Finnhub headlines objectively] | [Highlight both tailwinds & downgrades] |
    | **Bear Case / Key Negatives** | [Identify top 1-2 structural, cyclical, or competitive risks] | [Value-trap & downside risk check] |

    **Portfolio Manager Verdict:**
    [2-3 crisp sentences weighing the forward growth opportunity against the key negatives.]

    ---

    **2. Long-Term Accumulation & Risk Plan**
    | Plan Component | Target Level | Notes |
    | :--- | :--- | :--- |
    | **Accumulation Zone** | {c}{acc_low} - {c}{acc_high} | Staged entries on weakness |
    | **Thesis Invalidation** | {c}{invalidation_sl} (-15%) | Exit if forward thesis breaks |
    | **12-18M Target Price** | {c}{target_12m} | Street consensus / fair value objective |
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

    response_text += f"\n\n🤖 *Engine: {engine_used} + Finnhub Forward Consensus*"

    return {
        "username": "Institutional Fundamental Desk",
        "embeds": [{
            "title": f"🏛️ Long-Term Pick: {stock['name']} ({stock['ticker']}) — Score: {stock['score']}/100",
            "description": f"**FORWARD_PROFILE: Fwd P/E {stock['forward_pe']}x | P/S {stock['price_to_sales']}x | Upside {stock['target_upside']}**\n\n---\n\n{response_text[:3900]}",
            "color": 3447003,
            "footer": {"text": f"Forward Fundamental Scan: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Engine: {engine_used}"}
        }]
    }


def run_long_term_pipeline():
    print("==================================================")
    print("  🏛️ RUNNING FORWARD-LOOKING FUNDAMENTAL & GARP SCAN")
    print("==================================================")

    candidates = []
    for ticker in LONG_TERM_UNIVERSE:
        print(f"🔍 Screening {ticker} (Forward EPS, Street Targets, P/E, P/S, News)...")
        result = analyze_stock_fundamentals(ticker)
        if result:
            candidates.append(result)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_picks = candidates[:4]

    if not top_picks:
        print("⚠️ No candidates passed fundamental filters.")
        return

    for pick in top_picks:
        print(f"✅ Top Forward Pick: {pick['ticker']} (Score: {pick['score']}/100, Fwd P/E: {pick['forward_pe']}, Upside: {pick['target_upside']})")
        embed_payload = generate_long_term_thesis(pick)
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=embed_payload)

    print(f"--> Forward fundamental scan complete. {len(top_picks)} institutional picks dispatched to Discord.")


if __name__ == "__main__":
    run_long_term_pipeline()

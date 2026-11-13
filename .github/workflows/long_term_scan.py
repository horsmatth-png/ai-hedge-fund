name: Long-Term Fundamental & Valuation Screener

on:
  schedule:
    # Runs every Sunday at 18:00 UTC ahead of the weekly market open
    - cron: '0 18 * * 0'
  workflow_dispatch:

jobs:
  run-long-term-screener:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          pip install requests yfinance python-dotenv groq google-generativeai

      - name: Run Long-Term Fundamental Screener
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
          FINNHUB_KEY: ${{ secrets.FINNHUB_KEY }}
        run: python long_term_screener.py

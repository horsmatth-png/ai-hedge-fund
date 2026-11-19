name: IG Dual-Session Trade Execution (UK 8:30 & US 15:00)

on:
  schedule:
    # 08:30 UK / UTC (adjust for BST if needed) and 15:00 UK / UTC
    - cron: '30 8 * * 1-5'
    - cron: '0 15 * * 1-5'
  workflow_dispatch:
    inputs:
      region:
        description: 'Force region (UK or US, leave blank for auto)'
        required: false
        default: ''

permissions:
  contents: write

jobs:
  execute-trades:
    runs-on: ubuntu-latest
    # Prevent concurrent runs of this workflow from colliding
    concurrency:
      group: ig-bot-state
      cancel-in-progress: false

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests pandas numpy yfinance python-dotenv trading-ig groq google-generativeai lxml html5lib beautifulsoup4

      - name: Run Daily Spread Betting Execution
        env:
          RUN_MODE: execute_daily
          REGION: ${{ github.event.inputs.region }}
          IG_USERNAME: ${{ secrets.IG_USERNAME }}
          IG_PASSWORD: ${{ secrets.IG_PASSWORD }}
          IG_API_KEY: ${{ secrets.IG_API_KEY }}
          IG_ACC_TYPE: ${{ secrets.IG_ACC_TYPE }}
          IG_ACC_NUMBER: ${{ secrets.IG_ACC_NUMBER }}
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
        run: python ig_execution_bot.py

      - name: Sync Remote State Before Commit
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add active_trades.json || true
          git stash --include-untracked || true
          git pull --rebase origin main
          git stash pop || true

      - name: Commit and push state update
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: update active_trades.json after execution"
          file_pattern: active_trades.json

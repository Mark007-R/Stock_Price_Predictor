# Stock-Price-Forecaster

A stock-price forecasting app: walk-forward evaluation of several forecasters against persistence and buy-and-hold baselines, with cost-aware long/flat backtests and conformal prediction intervals. Flask UI, FastAPI service, Streamlit ops dashboard, Docker.

> 🔗 **Live demo:** https://iambatman07-stock-price-forecaster.hf.space · [HF Space](https://huggingface.co/spaces/IamBatman07/Stock-Price-Forecaster)
>
> ⚡ **60-second local demo:** `python demo.py` (offline, ~8s)

---

## Run it

```bash
pip install -r requirements.txt

python demo.py                                # offline demo
pytest tests -q                               # test suite
python app.py                                 # Flask UI on :5000
uvicorn src.serving.api:app --port 8000       # API on :8000
streamlit run src/serving/dashboard.py        # ops dashboard
docker compose up                             # API + Redis + MLflow + dashboard
```

Example API call:

```bash
curl -s -X POST localhost:8000/backtest -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "model": "arima", "cost_bps": 5}'
```

`/backtest` refuses zero-cost runs at the schema level and always returns buy-and-hold alongside the strategy.

---

## Limitations

Survivorship-biased ticker universe (all survivors); US large-cap daily bars only; long/flat execution model (no shorting, slippage modelled as a flat bps charge); interval coverage assumes exchangeable errors.

**Nothing in this repo is investment advice.**

---

## License

MIT — see [LICENSE](LICENSE).

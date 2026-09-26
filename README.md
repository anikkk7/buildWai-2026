# National PHC Health Resource & Supply Chain Platform

An AI platform giving real-time visibility into medicine stock, bed availability,
and staff attendance across India's Primary Health Centre (PHC) network — with
demand forecasting, automated early warnings for stockouts, AI-recommended
cross-district resource redistribution, and a Gemini-powered multilingual
situation report for health officials.

Built as a hackathon prototype. Data is synthetic (generated to mirror real
HMIS patterns) so the demo runs instantly without needing live government data
access — see "Scaling to real data" below for how to swap in the real thing.

## How this maps to the challenge

| Requirement | Where it lives |
|---|---|
| Real-time visibility (stock/beds/staff) | `data_generator.py` produces the feed, `app.py` displays it live |
| Demand forecasting | `forecast.py` — per-PHC, per-medicine linear trend model |
| Early warnings for stockouts | `forecast.py` risk tiers (Critical / Warning / Watch), shown in `app.py` |
| Automated cross-district redistribution | `redistribution.py` — greedy nearest-surplus matcher |
| Mandatory Google AI integration | `gemini_assistant.py` — Gemini turns the raw tables into a plain-language brief |
| Multilingual support | Gemini situation report is generated directly in the selected Indian language |
| Built for India, scales across states | Dataset spans 7 states / 21 districts / 63 PHCs; architecture is state-agnostic |

## Project structure

```
phc-ai-platform/
├── data_generator.py     # synthetic PHC dataset (stock, beds, staff) across Indian states
├── forecast.py            # demand forecasting + stockout risk tiers
├── redistribution.py      # cross-district transfer recommendations
├── gemini_assistant.py    # Google Gemini situation-report generator (multilingual)
├── app.py                 # Streamlit dashboard — the demo entry point
├── requirements.txt
├── .env.example
└── data/
    └── phc_stock_data.csv # generated dataset (created by data_generator.py)
```

## Run it locally

```bash
pip install -r requirements.txt
python data_generator.py          # generates data/phc_stock_data.csv
export GEMINI_API_KEY="your-key"  # optional — get one free at https://aistudio.google.com/apikey
streamlit run app.py
```

Without a `GEMINI_API_KEY` set, the app still runs fully — the AI situation
report falls back to a template-based summary so you can demo offline, but
for the hackathon submission you'll want the real key active so judges see
live Gemini output.

## Deploying a live link (for the "Deployed link" submission requirement)

The simplest free option for a Streamlit app:

1. Push this folder to a public (or access-granted) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click "New app".
3. Point it at your repo and `app.py`.
4. Add `GEMINI_API_KEY` under the app's "Secrets" settings.
5. Deploy — you'll get a public `*.streamlit.app` URL to submit.

Alternative: containerize with a simple Dockerfile and deploy to **Google Cloud
Run** (fits naturally with the "Google AI" theme of the challenge, and scales
to zero when idle).

## Scaling to real data

Replace `data_generator.py`'s output with a live or scheduled pull from:
- State HMIS (Health Management Information System) exports
- [data.gov.in](https://data.gov.in) health infrastructure datasets
- For production-scale forecasting, swap `forecast.py`'s linear model for
  **Vertex AI AutoML Forecasting**, and route redistribution through
  **BigQuery** for national-scale queries across states.

## Next steps for the full submission package

This repo covers the working end-to-end prototype. Still needed:
- [ ] Demo video (3–5 min walkthrough of `app.py`)
- [ ] Pitch deck (10–12 slides)
- [ ] 2–3 line solution description
- [ ] Live deployed link (see above)

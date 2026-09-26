"""
gemini_assistant.py
--------------------
The mandatory Google AI / GenAI layer. Raw tables of stockout risk and
redistribution suggestions are useful to a data analyst but not to a
busy district health officer. This module asks Gemini to turn the
numbers into a short, plain-language "situation report" — optionally
in a regional Indian language — that could be read aloud (pair with
Cloud Text-to-Speech) or sent over SMS/WhatsApp in production.

Setup:
  1. Get a free API key from Google AI Studio: https://aistudio.google.com/apikey
  2. Set it as an environment variable: export GEMINI_API_KEY="your-key-here"

If no key is set, this module falls back to a template-based summary so
the rest of the app still runs and demos cleanly offline.
"""

import os

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

MODEL_NAME = "gemini-2.0-flash"


def _configure():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not GENAI_AVAILABLE:
        return None
    return genai.Client(api_key=api_key)


def _fallback_summary(risk_df, plan_df, language):
    """Template-based summary used when no Gemini API key is configured,
    so the demo still works without live credentials."""
    critical = risk_df[risk_df["risk_level"] == "Critical"]
    warning = risk_df[risk_df["risk_level"] == "Warning"]
    lines = [
        f"[Offline summary — set GEMINI_API_KEY for AI-generated reports in {language}]",
        f"{len(critical)} medicine stock lines are CRITICAL (stockout within {3} days) "
        f"across {critical['state'].nunique()} states.",
        f"{len(warning)} lines are at WARNING level (stockout within 7 days).",
        f"{len(plan_df)} redistribution transfers are recommended to cover the gaps.",
    ]
    if not critical.empty:
        top = critical.iloc[0]
        lines.append(
            f"Most urgent: {top['phc_name']} ({top['district']}, {top['state']}) "
            f"is out of {top['medicine']} in under a day."
        )
    return "\n".join(lines)


def generate_situation_report(risk_df, plan_df, language="English", top_n=8) -> str:
    """
    Build a natural-language situation report from the risk and
    redistribution tables. `language` can be any language name Gemini
    understands, e.g. "Hindi", "Bengali", "Tamil", "English" — this is
    how the platform meets the multilingual requirement without a
    separate translation pipeline.
    """
    client = _configure()
    if client is None:
        return _fallback_summary(risk_df, plan_df, language)

    critical_rows = risk_df[risk_df["risk_level"] == "Critical"].head(top_n)
    plan_rows = plan_df.head(top_n)

    prompt = f"""
You are a health-systems briefing assistant for Indian state health officials.
Write a short, clear situation report in {language} (plain text, no markdown),
suitable to be read aloud or sent by SMS to a district health officer.

Cover, in order:
1. How many PHCs/medicines are at Critical vs Warning stockout risk right now.
2. The single most urgent situation (name the PHC, district, medicine).
3. A one-line summary of the top 2-3 recommended stock transfers between PHCs.
4. A calm, action-oriented closing line.

Keep it under 150 words total.

CRITICAL RISK DATA:
{critical_rows[['phc_name','district','state','medicine','days_to_stockout']].to_string(index=False)}

RECOMMENDED TRANSFERS:
{plan_rows[['medicine','shortage_phc','donor_phc','distance_km','recommended_transfer_qty']].to_string(index=False)}
"""

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return response.text
    except Exception as e:
        return f"[Gemini API error, showing offline summary instead: {e}]\n\n" + \
               _fallback_summary(risk_df, plan_df, language)


if __name__ == "__main__":
    import pandas as pd
    from forecast import build_risk_table
    from redistribution import build_redistribution_plan

    stock_df = pd.read_csv("data/phc_stock_data.csv")
    risk_df = build_risk_table(stock_df)
    plan_df = build_redistribution_plan(risk_df)
    print(generate_situation_report(risk_df, plan_df, language="English"))

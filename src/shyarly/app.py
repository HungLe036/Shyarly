"""Minimal Streamlit view of the reports `shyarly update` writes. Read-only: no refit, no re-sim.

    uv run --group app streamlit run src/shyarly/app.py
"""

import streamlit as st

from shyarly import dashboard as d

st.set_page_config(page_title="Shyarly — Worlds 2026", layout="wide")
st.title("Shyarly — Worlds 2026 odds")
st.caption("Read from `reports/` and `data/manual/`. Run `uv run shyarly update` to refresh.")

pct = st.column_config.NumberColumn(format="percent")

st.header("Title odds")
odds = d.title_odds()
if odds.empty:
    st.info("No `reports/tournament_odds.csv` yet.")
else:
    st.dataframe(odds, hide_index=True, width="stretch",
                 column_config={c: pct for c in d.STAGES.values()})

st.header("Swiss record distribution")
swiss = d.swiss_records()
if swiss.empty:
    st.info("No `reports/swiss_odds.csv` yet.")
else:
    st.dataframe(swiss, hide_index=True, width="stretch",
                 column_config={c: pct for c in ["qualify", *d.RECORDS]})
    st.bar_chart(swiss.set_index("team")[d.RECORDS], horizontal=True)

st.header("Odds over time")
hist = d.odds_history()
if hist.empty:
    st.info("No daily snapshots in `reports/daily/` yet.")
else:
    team = st.selectbox("Team", sorted(hist["team"].unique()))
    sub = hist[hist["team"] == team].pivot(index="as_of", columns="stage", values="p")
    sub = sub[[s for s in d.STAGES.values() if s in sub.columns]]
    if len(sub) < 2:
        st.caption(f"Only one snapshot so far ({sub.index[0]:%Y-%m-%d}).")
    st.line_chart(sub)
    st.dataframe(sub, width="stretch", column_config={c: pct for c in sub.columns})

st.header("Predictions vs results")
preds = d.predictions_vs_results()
if preds.empty:
    st.info("No logged predictions in `reports/predictions_log.csv` yet.")
else:
    st.dataframe(preds, hide_index=True, width="stretch",
                 column_config={"p_series_a": pct, "p_game_a": pct})
md = d.live_scores()
if md is None:
    st.info("No `reports/live_scores.md` yet.")
else:
    st.markdown(md.split("\n", 1)[1] if md.startswith("# ") else md)

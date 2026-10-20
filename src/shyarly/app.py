"""Worlds 2026 odds for a general audience. Read-only view of what `shyarly update` writes:
no refit, no re-simulation.

    uv run --group app streamlit run src/shyarly/app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from shyarly import dashboard as d

NAVY, GOLD, INK, MUTED, GRID, BG = "#0a1428", "#c8aa6e", "#e8e6e1", "#a9a8a2", "#262c36", "#11151c"
FONT = "Barlow"

st.set_page_config(page_title="Who wins Worlds 2026?", page_icon="🏆", layout="centered")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500&family=Barlow+Condensed:wght@500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label {{ font-family: '{FONT}', sans-serif; }}
h1, h2, h3 {{ font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: 0.01em; color: {INK}; }}
h2 {{ font-weight: 600 !important; font-size: 2rem !important; margin-top: 2.5rem !important; }}
.hero {{ background: {NAVY}; color: #f0e6d2; padding: 2.25rem 2rem 1.75rem; border-radius: 4px; border: 1px solid #1e2d45;
         border-bottom: 3px solid {GOLD}; margin-bottom: 1.5rem; }}
.hero h1 {{ color: #f0e6d2 !important; font-size: 3.1rem !important; line-height: 1.02; margin: 0 0 .75rem; font-weight: 600; }}
.hero p {{ color: #cdbe91; font-size: 1.1rem; max-width: 34em; margin: 0 0 1rem; line-height: 1.5; }}
.hero .fav {{ color: #f0e6d2; font-size: 1.05rem; max-width: 34em; line-height: 1.5; }}
.hero .fav b {{ color: {GOLD}; font-weight: 500; }}
.hero small {{ display: block; margin-top: .9rem; color: #a09b8c; font-size: .85rem; }}
.lede {{ color: {MUTED}; font-size: 1.05rem; max-width: 36em; margin: -.25rem 0 1rem; line-height: 1.55; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 14px; margin: .25rem 0 .5rem; font-size: .92rem; color: {MUTED}; }}
.chips span {{ display: inline-flex; align-items: center; gap: 6px; }}
.chips i {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}
.note {{ font-size: .9rem; color: {MUTED}; }}
code {{ color: {GOLD} !important; background: #1b212b !important; }}
.lede b {{ color: {INK}; font-weight: 500; }}
</style>
""", unsafe_allow_html=True)


def pct(x: float) -> str:
    if x >= 0.995:
        return "99%+"
    if x < 0.001:
        return "<0.1%"
    return f"{100 * x:.1f}%" if x < 0.1 else f"{100 * x:.0f}%"


def theme(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(font=FONT, padding={"left": 12, "right": 12, "top": 4, "bottom": 4})
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor="#3a414d",
                            labelFontSize=13, titleFontSize=12, titleFontWeight="normal")
            .configure_view(strokeWidth=0)
            .configure_legend(disable=True))


region_scale = alt.Scale(domain=list(d.REGIONS), range=[c for _, c in d.REGIONS.values()])
odds = d.title_odds()
names = dict(zip(odds["team"], odds["name"])) if not odds.empty else {}
updated = d.last_updated()
runs = d.sim_runs()

# ------------------------------------------------------------------ hero
if odds.empty:
    st.markdown('<div class="hero"><h1>Who will win Worlds 2026?</h1>'
                '<p>No forecast yet. Run <code>uv run shyarly update</code> to create one.</p></div>',
                unsafe_allow_html=True)
    st.stop()

top = odds.iloc[0]
lead = d.region_share(odds).iloc[0]
times = f"{runs:,} times" if runs else "thousands of times"
st.markdown(f"""
<div class="hero">
  <h1>Who will win Worlds 2026?</h1>
  <p>We played the whole tournament {times} on a computer, using every team's results from the last
  two years, and counted how often each team won it.</p>
  <div class="fav">Favourite: <b>{top['name']}</b>, champions in {pct(top['Champion'])} of the simulated
  tournaments. Teams from {lead['region_name']} win {pct(lead['champion'])} of them between them.</div>
  <small>Forecast updated {updated:%B %d, %H:%M} UTC</small>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ title race
st.markdown("## The title race")
st.markdown('<p class="lede">How often each team lifted the trophy across all the simulated tournaments. '
            'Colour shows the team\'s home region.</p>', unsafe_allow_html=True)
st.markdown('<div class="chips">' + "".join(
    f'<span><i style="background:{c}"></i>{n}</span>' for n, c in d.REGIONS.values()) + "</div>",
    unsafe_allow_html=True)

race = odds.assign(label=odds["Champion"].map(pct),
                   region_name=odds["region"].map(lambda r: d.REGIONS.get(r, (r,))[0]))
order = race["name"].tolist()
bars = alt.Chart(race).mark_bar(cornerRadiusEnd=4, height=18).encode(
    y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=180, ticks=False, domain=False)),
    x=alt.X("Champion:Q", title=None, axis=alt.Axis(format="%", tickCount=5)),
    color=alt.Color("region:N", scale=region_scale),
    tooltip=[alt.Tooltip("name:N", title="Team"), alt.Tooltip("region_name:N", title="Region"),
             alt.Tooltip("Champion:Q", title="Wins the title", format=".1%"),
             alt.Tooltip("Final:Q", title="Reaches the final", format=".1%")],
)
labels = bars.mark_text(align="left", dx=6, fontSize=13).encode(text="label:N", color=alt.value(INK))
st.altair_chart(theme((bars + labels).properties(height=34 * len(race))), width="stretch")

# ------------------------------------------------------------------ how far
st.markdown("## How far each team gets")
st.markdown('<p class="lede">Eight teams reach the quarterfinals, four the semifinals, two the final. '
            'Brighter means more likely.</p>', unsafe_allow_html=True)
stages = list(d.STAGES.values())
heat = odds.melt(id_vars=["name"], value_vars=stages, var_name="stage", value_name="p")
heat["label"] = heat["p"].map(pct)
base = alt.Chart(heat).encode(
    x=alt.X("stage:N", sort=stages, title=None, axis=alt.Axis(orient="top", labelAngle=0, ticks=False, domain=False)),
    y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=180, ticks=False, domain=False)))
cells = base.mark_rect(cornerRadius=3, stroke=BG, strokeWidth=2).encode(
    color=alt.Color("p:Q", scale=alt.Scale(domain=[0, 0.3, 0.6, 0.9],
                                           range=["#18212d", "#1f4f8a", "#3987e5", "#9cc6f5"])),
    tooltip=[alt.Tooltip("name:N", title="Team"), alt.Tooltip("stage:N", title="Reaches"),
             alt.Tooltip("p:Q", title="Chance", format=".1%")])
text = base.mark_text(fontSize=13).encode(
    text="label:N", color=alt.condition("datum.p > 0.7", alt.value("#0b1220"), alt.value(INK)))
st.altair_chart(theme((cells + text).properties(height=30 * len(odds))), width="stretch")

# ------------------------------------------------------------------ swiss
swiss = d.swiss_records()
if not swiss.empty:
    st.markdown("## Getting out of the Swiss stage")
    st.markdown('<p class="lede">All 16 teams play up to five rounds. Three wins and a team goes through to the '
                'knockouts; three losses and it goes home. Blue is the chance of going through, and with which '
                'record; red is the chance of going out.</p>', unsafe_allow_html=True)
    sp = d.swiss_split(swiss)
    sorder = swiss["name"].tolist()
    rec_colors = ["#9cc6f5", "#3987e5", "#1f5a9e", "#8f3a3a", "#d04a4a", "#f09595"]
    rec_scale = alt.Scale(domain=d.RECORDS, range=rec_colors)
    sbars = alt.Chart(sp).mark_bar(height=18, stroke=BG, strokeWidth=1.5).encode(
        y=alt.Y("name:N", sort=sorder, title=None, axis=alt.Axis(labelLimit=180, ticks=False, domain=False)),
        x=alt.X("sum(signed):Q", title=None, stack="zero",
                axis=alt.Axis(values=[-1, -0.5, 0, 0.5, 1], labelExpr="format(abs(datum.value), '.0%')")),
        color=alt.Color("record:N", scale=rec_scale),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("name:N", title="Team"), alt.Tooltip("record:N", title="Final record"),
                 alt.Tooltip("p:Q", title="Chance", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED, strokeWidth=1).encode(x="x:Q")
    legend = ["3-0", "3-1", "3-2 (through)", "2-3", "1-3", "0-3 (out)"]
    st.markdown('<div class="chips">' + "".join(
        f'<span><i style="background:{c}"></i>{t}</span>' for t, c in zip(legend, rec_colors)) + "</div>",
        unsafe_allow_html=True)
    st.altair_chart(theme((sbars + rule).properties(height=34 * len(swiss))), width="stretch")

# ------------------------------------------------------------------ matches
st.markdown("## Match by match")
cards = d.match_cards(d.predictions_vs_results(), names)


@st.cache_data(ttl=3600, show_spinner="Working out the match odds…")
def upcoming(stamp: float) -> pd.DataFrame:
    """Current forecast for drawn but unplayed matches that have no logged prediction yet."""
    from shyarly import paths
    from shyarly.config import load_leagues, load_model_config, load_worlds
    from shyarly.data.manual import load_results, swiss_fixed
    from shyarly.models.predictor import Predictor, worlds_team_ref

    worlds, leagues, cfg = load_worlds(), load_leagues(), load_model_config()
    rows = [r for r in load_results(paths.MANUAL_RESULTS, worlds) if r.winner is None]
    if not rows or not paths.GAMES_PARQUET.exists():
        return pd.DataFrame()
    fx = swiss_fixed(load_results(paths.MANUAL_RESULTS, worlds), worlds, [t.code for t in worlds.teams])
    pr = Predictor.from_config(pd.read_parquet(paths.GAMES_PARQUET), pd.Timestamp.now(tz="UTC"), cfg)
    ref = lambda c: worlds_team_ref(c, worlds, leagues, 2026)  # noqa: E731
    out = []
    for r in rows:
        bo = 5 if r.stage == "knockout" else fx.best_of.get(r.match_id)
        if bo is None:
            continue
        out.append({"match_date_utc": r.date_utc.isoformat(), "stage": r.stage, "round": r.round,
                    "team_a": r.team_a, "team_b": r.team_b, "best_of": bo, "model": "bt", "mode": "live",
                    "p_series_a": pr.p_series(ref(r.team_a), ref(r.team_b), bo), "winner": pd.NA, "score": pd.NA})
    return pd.DataFrame(out)


if cards.empty:
    from shyarly import paths as _paths
    stamp = _paths.MANUAL_RESULTS.stat().st_mtime if _paths.MANUAL_RESULTS.exists() else 0.0
    cards = d.match_cards(upcoming(stamp), names)
if cards.empty:
    st.markdown('<p class="lede">No matches drawn yet. Once the draw is out, each match and its favourite '
                'appear here, then get checked against the result.</p>', unsafe_allow_html=True)
else:
    if cards["called_it"].isna().all():
        st.markdown('<p class="lede">The next matches and who the model expects to win. After each match the '
                    'result goes next to the forecast, so you can see how it is doing.</p>', unsafe_allow_html=True)
    played = cards.dropna(subset=["called_it"])
    if len(played):
        hits = int(played["called_it"].astype(bool).sum())
        st.markdown(f'<p class="lede">The favourite has won <b>{hits} of {len(played)}</b> matches so far.</p>',
                    unsafe_allow_html=True)
    show = cards.assign(
        chance=cards["chance"].round(2), best_of=cards["best_of"].map(lambda b: f"Bo{b}"),
        called_it=cards["called_it"].map(lambda v: "" if pd.isna(v) else ("Yes" if v else "No")))
    if cards["called_it"].isna().all():                 # nothing played yet: no result columns
        show = show.drop(columns=["result", "called_it"])
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "date": "Date", "round": "Round", "favourite": "Favourite", "underdog": "Underdog",
        "chance": st.column_config.ProgressColumn("Favourite's chance", format="percent", min_value=0, max_value=1),
        "best_of": "Format", "result": "Result", "called_it": "Favourite won?"})

# ------------------------------------------------------------------ over time
hist = d.odds_history()
st.markdown("## How the odds have moved")
if hist.empty or hist["as_of"].nunique() < 2:
    st.markdown('<p class="lede">There is one forecast so far. Once matches are played a new one is made each '
                'day, and the lines here will show how each team\'s title chances rise and fall.</p>',
                unsafe_allow_html=True)
else:
    champ = hist[hist["stage"] == "Champion"].assign(name=lambda x: x["team"].map(names).fillna(x["team"]))
    picks = st.multiselect("Teams", order, default=order[:4], max_selections=6)
    sub = champ[champ["name"].isin(picks)]
    lines = alt.Chart(sub).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2.5).encode(
        x=alt.X("as_of:T", title=None, axis=alt.Axis(format="%b %d", grid=False)),
        y=alt.Y("p:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("name:N", scale=alt.Scale(domain=picks, range=[c for _, c in d.REGIONS.values()][:len(picks)])),
        tooltip=[alt.Tooltip("name:N", title="Team"), alt.Tooltip("as_of:T", title="Date", format="%b %d"),
                 alt.Tooltip("p:Q", title="Title chance", format=".1%")])
    ends = alt.Chart(sub).transform_window(rank="rank()", sort=[alt.SortField("as_of", order="descending")],
                                           groupby=["name"]).transform_filter("datum.rank == 1").mark_text(
        align="left", dx=8, fontSize=13).encode(x="as_of:T", y="p:Q", text="name:N", color=alt.value(INK))
    st.altair_chart(theme((lines + ends).properties(height=320)), width="stretch")

# ------------------------------------------------------------------ explainer
st.markdown("## How this works")
with st.expander("Where do these numbers come from?"):
    st.markdown(f"""
- **Team strength.** Every professional game from the last two years goes into a rating for each
  team and each region. Recent games count more. International events are what tell us how the
  regions compare.
- **Single matches.** Two ratings give a chance for one game, and from that a chance for a
  best-of-three or best-of-five.
- **The whole tournament.** The computer then plays out Worlds {times}, following the official
  rules for the draw, the Swiss stage and the bracket, and counts how far each team gets.
  Because we're never sure exactly how strong a team is, each simulated tournament uses slightly
  different strengths.
- **As Worlds goes on,** real results replace simulated ones and the forecast is redone each day.
""")
with st.expander("How accurate is it?"):
    st.markdown("""
- On past Worlds and MSI tournaments it beat a coin flip, a region-only guess and the classic Elo
  rating, and its percentages are well calibrated: when it says 70%, the favourite wins about 70%
  of the time.
- Against bookmakers it's roughly level with the average of typical betting sites, and probably a
  little behind the sharpest one.
- It doesn't know about rosters, champion picks, the current patch or anything said on match day,
  so treat it as a well-informed starting point, not a certainty.
""")
st.markdown(f'<p class="note">Forecast files are in <code>reports/</code>. Last updated '
            f'{updated:%B %d %Y, %H:%M} UTC.</p>', unsafe_allow_html=True)

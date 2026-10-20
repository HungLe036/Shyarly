"""OddsPortal export: clean-row filter, date-aware team names, joining to series by match day."""

import pandas as pd
import pytest

from shyarly.data.market import fair_probs
from shyarly.data.oddsportal import load_oddsportal
from shyarly.eval.benchmark import join_quotes, make_resolver

COLS = ["id", "tournament", "season", "match_date", "home_team", "away_team", "home_score", "away_score", "winner",
        "odds_home", "odds_away", "match_url", "match_id", "odds_type", "odds_average_home", "odds_average_away",
        "odds_opening_home", "odds_opening_away", "bookmaker_count"]


def row(i, date, home, away, avg=(1.5, 2.6), url=None, match_id=None, enriched=True, **kw):
    a = avg if enriched else (None, None)
    return {"id": i, "tournament": kw.get("tournament", "World Championship"), "season": int(date[-4:]),
            "match_date": date, "home_team": home, "away_team": away, "home_score": 1, "away_score": 0,
            "winner": home, "odds_home": avg[0], "odds_away": avg[1],
            "match_url": url or f"https://x/{i}/#m{i}", "match_id": match_id or f"m{i}", "odds_type": "closing",
            "odds_average_home": a[0], "odds_average_away": a[1], "odds_opening_home": 1.6, "odds_opening_away": 2.4,
            "bookmaker_count": 6 if enriched else 0}


def write(tmp_path, rows):
    p = tmp_path / "op.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    return p


def test_loader_keeps_clean_enriched_rows_and_counts_drops(tmp_path):
    rows = [
        row(1, "01 Oct 2024", "A", "B"),
        row(2, "02 Oct 2024", "vs", "Odds, Predictions and H2H Results"),            # scraper junk
        row(3, "03 Oct 2024", "C", "D", enriched=False),                              # no per-bookmaker odds
        row(4, "04 Oct 2024", "E", "F", url="https://x/ef/#same", avg=(1.8, 2.0)),    # one link and price on
        row(5, "09 Oct 2024", "E", "F", url="https://x/ef/#same", avg=(1.8, 2.0)),    #   two dates: stale
        row(6, "05 Oct 2024", "G", "H", match_id="dup"),                              # duplicate match id
        row(7, "05 Oct 2024", "G", "H", match_id="dup"),
    ]
    q, dropped = load_oddsportal(write(tmp_path, rows))
    assert list(q["fixture_id"]) == ["m1"]
    assert dropped == {"junk": 1, "not_enriched": 1, "stale_shared_link": 2, "duplicate_match_id": 2}
    r = q.iloc[0]
    assert r["start"] == pd.Timestamp("2024-10-01", tz="UTC")
    assert r["p_mkt_series"] == pytest.approx(fair_probs(1.5, 2.6)[0])
    assert r["p_open_series"] == pytest.approx(fair_probs(1.6, 2.4)[0])
    assert (r["team_1"], r["team_2"], r["season"]) == ("A", "B", 2024)


def test_resolver_tries_current_name_then_past_name():
    resolve = make_resolver({"Team A", "Team B", "Old A"}, {"A": "Team A"}, {"A": {2022: "Old A"}})
    assert resolve("A", 2024) == ["Team A"]
    assert resolve("A", 2022) == ["Team A", "Old A"]
    assert resolve("team-b", 2022) == ["Team B"]
    assert resolve("Nobody", 2022) == []


def test_join_quotes_by_match_day_with_past_names():
    t = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    series = pd.DataFrame([
        {"series_id": "s1", "date_utc": t("2022-10-01T23:30"), "event": "WLDs 2022", "team_a": "Old A", "team_b": "Team B",
         "winner_team": "Team B", "inferred_best_of": 5, "wins_a": 1, "wins_b": 3},
    ])
    games = pd.DataFrame(columns=["series_id", "game_in_series", "blue_team", "red_team", "blue_win"])
    quotes = pd.DataFrame([{"fixture_id": "q1", "tournament": "World Championship", "season": 2022,
                            "start": t("2022-10-02"), "team_1": "A", "team_2": "Team B", "p_mkt_series": 0.4}])
    resolve = make_resolver({"Team A", "Team B", "Old A"}, {"A": "Team A"}, {"A": {2022: "Old A"}})
    j, unmatched = join_quotes(quotes, series, games, resolve, window=pd.Timedelta(days=1))
    assert unmatched == [] and j.loc[0, "team_1"] == "Old A" and j.loc[0, "y_series"] == 0
    assert j.loc[0, "best_of"] == 5 and j.loc[0, "p_mkt_series"] == 0.4

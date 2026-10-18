"""Schema, duplicate and point-in-time tests for the games table (data/processed/games.parquet)."""

import pandas as pd
import pytest

from shyarly import paths
from shyarly.config import load_leagues
from shyarly.data.games import GAMES_SCHEMA, build_games, build_series

LEAGUES = load_leagues()


def _game(gid, date, league, blue, red, blue_wins, game=1, patch="26.01"):
    common = dict(gameid=gid, date=date, league=league, game=game, patch=patch, playoffs=0, split=None)
    return [
        {**common, "side": "Blue", "teamname": blue, "result": int(blue_wins)},
        {**common, "side": "Red", "teamname": red, "result": int(not blue_wins)},
    ]


def _frame(*games):
    return pd.DataFrame([row for g in games for row in g])


def _synthetic():
    return _frame(
        # Domestic history: GEN in LCK, BLG in LPL, KC in LEC.
        _game("d1", "2026-02-01 10:00:00", "LCK", "Gen.G", "T1", True),
        _game("d2", "2026-02-01 10:00:00", "LPL", "Bilibili Gaming", "Top Esports", False),
        _game("d3", "2026-02-02 10:00:00", "LEC", "Karmine Corp", "G2 Esports", True),
        # A same-region Worlds qualifier before the event's first cross-region game.
        _game("q1", "2026-09-17 09:00:00", "WLDs", "Top Esports", "Bilibili Gaming", True),
        # Worlds main event: a Bo3 with three games, then a Bo1 the next day.
        _game("w1", "2026-10-15 18:00:00", "WLDs", "Gen.G", "Bilibili Gaming", True, game=1),
        _game("w2", "2026-10-15 19:00:00", "WLDs", "Bilibili Gaming", "Gen.G", True, game=2),
        _game("w3", "2026-10-15 20:00:00", "WLDs", "Gen.G", "Bilibili Gaming", True, game=3),
        _game("w4", "2026-10-16 18:00:00", "WLDs", "Karmine Corp", "Gen.G", False, game=1),
        # A remake-style row with no winner must be dropped.
        [
            dict(gameid="bad", date="2026-10-16 19:00:00", league="WLDs", game=1, patch="26.20",
                 playoffs=0, split=None, side="Blue", teamname="T1", result=0),
            dict(gameid="bad", date="2026-10-16 19:00:00", league="WLDs", game=1, patch="26.20",
                 playoffs=0, split=None, side="Red", teamname="Gen.G", result=0),
        ],
    )


@pytest.fixture(scope="module")
def games():
    return build_games(_synthetic(), LEAGUES)


def test_schema_columns_and_dtypes(games):
    assert list(games.columns) == list(GAMES_SCHEMA)
    for col, dtype in GAMES_SCHEMA.items():
        assert str(games[col].dtype) == dtype, col
    assert str(games["date_utc"].dt.tz) == "UTC"


def test_one_row_per_game_and_no_winnerless_games(games):
    assert games["game_id"].is_unique
    assert "bad" not in set(games["game_id"])
    assert set(games["winner"]) <= {"blue", "red"}
    assert (games["blue_win"] == (games["winner"] == "blue").astype("int8")).all()
    assert (games["blue_team"] != games["red_team"]).all()


def test_double_recorded_game_is_dropped():
    rows = pd.concat([
        _synthetic(),
        _frame(_game("d1-copy", "2026-02-01 10:00:30", "LCK", "Gen.G", "T1", True)),
    ])
    g = build_games(rows, LEAGUES)
    assert "d1" in set(g["game_id"]) and "d1-copy" not in set(g["game_id"])


def test_team_season_ids(games):
    row = games.set_index("game_id").loc["d1"]
    assert row["blue_team_season_id"] == "gen-g_2026"
    assert row["red_team_season_id"] == "t1_2026"


def test_series_grouping(games):
    g = games.set_index("game_id")
    assert g.loc["w1", "series_id"] == g.loc["w2", "series_id"] == g.loc["w3", "series_id"]
    assert list(g.loc[["w1", "w2", "w3"], "game_in_series"]) == [1, 2, 3]
    assert g.loc["w4", "series_id"] != g.loc["w1", "series_id"]
    series = build_series(games).set_index("series_id")
    s = series.loc[g.loc["w1", "series_id"]]
    assert (s["n_games"], s["winner_team"], s["inferred_best_of"]) == (3, "Gen.G", 3)


def test_domestic_region_comes_from_league(games):
    g = games.set_index("game_id")
    assert (g.loc["d1", "blue_region"], g.loc["d2", "blue_region"], g.loc["d3", "red_region"]) == ("LCK", "LPL", "LEC")
    assert not g.loc["d1", "cross_region"]


def test_international_region_is_as_of_earlier_games(games):
    g = games.set_index("game_id")
    assert (g.loc["w1", "blue_region"], g.loc["w1", "red_region"]) == ("LCK", "LPL")
    assert g.loc["w1", "cross_region"]


def test_qualifier_detected_and_not_scored(games):
    g = games.set_index("game_id")
    assert g.loc["q1", "stage"] == "qualifier" and not g.loc["q1", "scored"]
    assert g.loc["w1", "stage"] == "main" and g.loc["w1", "scored"]
    assert g.loc["w1", "event"] == "WLDs 2026"


def test_future_rows_do_not_change_past_regions():
    """Point-in-time: adding later games (here KC moving to the LCK) must not
    change any column of earlier games."""
    base = build_games(_synthetic(), LEAGUES).set_index("game_id")
    extra = _frame(
        _game("f1", "2026-11-01 10:00:00", "LCK", "Karmine Corp", "T1", True),
        _game("f2", "2026-11-02 10:00:00", "LCK", "Karmine Corp", "Gen.G", True),
        _game("f3", "2026-11-03 10:00:00", "WLDs", "Karmine Corp", "Gen.G", False),
    )
    more = build_games(pd.concat([_synthetic(), extra]), LEAGUES).set_index("game_id")
    pd.testing.assert_frame_equal(base, more.loc[base.index])


# ------------------------------------------------------- integration (real data)

real = pytest.mark.skipif(not paths.GAMES_PARQUET.exists(), reason="run `shyarly ingest` first")


@pytest.fixture(scope="module")
def real_games():
    return pd.read_parquet(paths.GAMES_PARQUET)


@real
def test_real_schema(real_games):
    assert list(real_games.columns) == list(GAMES_SCHEMA)
    for col, dtype in GAMES_SCHEMA.items():
        assert str(real_games[col].dtype) == dtype, col


@real
def test_real_no_duplicates(real_games):
    assert real_games["game_id"].is_unique
    assert not real_games.duplicated(["date_utc", "blue_team", "red_team"]).any()
    assert (real_games["blue_team"] != real_games["red_team"]).all()


@real
def test_real_dates_in_range(real_games):
    assert real_games["date_utc"].min() >= pd.Timestamp("2022-01-01", tz="UTC")
    assert real_games["date_utc"].max() < pd.Timestamp.now(tz="UTC")


@real
def test_real_series_numbering_is_contiguous(real_games):
    by = real_games.sort_values(["series_id", "game_in_series"]).groupby("series_id")["game_in_series"]
    assert (by.min() == 1).all()
    assert (by.max() == by.count()).all()
    assert by.count().max() <= 5


@real
def test_real_backtest_events_present_with_regions(real_games):
    s = real_games[real_games["scored"]]
    events = set(s["event"])
    for ev in ["WLDs 2022", "MSI 2023", "WLDs 2023", "MSI 2024", "WLDs 2024",
               "MSI 2025", "WLDs 2025", "MSI 2026"]:
        assert ev in events, ev
    both = s["blue_region"].notna() & s["red_region"].notna()
    assert both.mean() >= 0.99

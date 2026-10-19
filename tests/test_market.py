"""Compact, committed market-odds table built from OddsPapi historical odds."""

import pandas as pd
import pytest

from shyarly.data.market import MARKET_COLUMNS, append_market_rows, closing_quote, fair_probs

START = pd.Timestamp("2026-10-23T17:00:00Z")


def history(snaps1, snaps2, bookmaker="pinnacle"):
    """Shape documented by OddsPapi: bookmakers -> markets -> 181 -> outcomes -> 181/182 -> players -> 0 -> [snapshots]."""
    def snaps(xs):
        return [{"createdAt": t, "price": p} for t, p in xs]
    return {"bookmakers": {bookmaker: {"markets": {"181": {"outcomes": {
        "181": {"players": {"0": snaps(snaps1)}}, "182": {"players": {"0": snaps(snaps2)}}}}}}}}


def test_fair_probs_remove_the_margin():
    p1, p2, over = fair_probs(1.5, 2.6)
    assert p1 + p2 == pytest.approx(1.0)
    assert p1 == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 2.6))
    assert over == pytest.approx(1 / 1.5 + 1 / 2.6 - 1)


def test_closing_quote_is_last_price_before_start():
    h = history([("2026-10-22T10:00:00Z", 1.60), ("2026-10-23T16:55:00Z", 1.50), ("2026-10-23T17:20:00Z", 1.10)],
                [("2026-10-22T10:00:00Z", 2.40), ("2026-10-23T16:50:00Z", 2.60), ("2026-10-23T17:20:00Z", 7.00)])
    q = closing_quote(h, START, "pinnacle")
    assert (q["price_1"], q["price_2"]) == (1.50, 2.60)          # in-play prices after the start are ignored
    assert q["quote_time_utc"] == "2026-10-23T16:55:00+00:00"
    assert q["n_quotes"] == 6


def test_closing_quote_missing_market_or_no_pre_start_price():
    assert closing_quote({"bookmakers": {}}, START, "pinnacle") is None
    late = history([("2026-10-23T17:30:00Z", 1.5)], [("2026-10-23T17:30:00Z", 2.6)])
    assert closing_quote(late, START, "pinnacle") is None


def test_tournament_rows_skip_unfinished_and_fixtures_without_history():
    from shyarly.data.market import tournament_rows
    from shyarly.data.oddspapi import OddsPapiError

    fixtures = [
        {"fixtureId": "a", "statusId": 2, "startTime": "2026-10-23T17:00:00Z", "participant1Name": "X", "participant2Name": "Y"},
        {"fixtureId": "b", "statusId": 2, "startTime": "2026-10-23T18:00:00Z"},
        {"fixtureId": "c", "statusId": 0, "startTime": "2026-10-30T18:00:00Z"},
    ]

    class Fake:
        def get(self, endpoint, refresh=False, **params):
            if endpoint == "fixtures":
                return fixtures
            if params["fixtureId"] == "b":
                raise OddsPapiError("historical-odds returned HTTP 404", 404)
            assert params["fixtureId"] == "a"
            return history([("2026-10-23T16:00:00Z", 1.5)], [("2026-10-23T16:00:00Z", 2.6)])

    rows, missing = tournament_rows(Fake(), 1, "pinnacle", [181, 1847])
    assert missing == 1 and [r["market"] for r in rows] == [181]
    assert rows[0]["team_1"] == "X" and rows[0]["p1_fair"] == pytest.approx(fair_probs(1.5, 2.6)[0], abs=1e-6)


def test_append_never_overwrites(tmp_path):
    path = tmp_path / "market_odds.csv"
    row = {c: None for c in MARKET_COLUMNS} | {"fixture_id": "f1", "bookmaker": "pinnacle", "market": 181, "price_1": 1.5}
    assert append_market_rows(path, [row]) == 1
    assert append_market_rows(path, [row | {"price_1": 9.9}, row | {"fixture_id": "f2"}]) == 1
    df = pd.read_csv(path)
    assert list(df.columns) == MARKET_COLUMNS
    assert df.set_index("fixture_id").loc["f1", "price_1"] == 1.5 and len(df) == 2

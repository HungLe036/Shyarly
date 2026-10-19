"""OddsPapi client: caching, monthly budget guard, key hygiene. No network."""

from datetime import UTC, datetime

import pytest

from shyarly.data.oddspapi import BudgetExceeded, OddsPapi, OddsPapiError, load_key

CFG = {"base_url": "https://example.test/v4", "monthly_budget": 2, "min_interval_s": 0}
KEY = "secret-key-123"


def client(tmp_path, calls, status=200, now=datetime(2026, 10, 19, tzinfo=UTC)):
    def transport(url, params):
        calls.append((url, dict(params)))
        return (status, {"ok": True}) if status == 200 else (status, {"error": f"bad key {KEY}"})
    return OddsPapi(key=KEY, cache_dir=tmp_path, cfg=CFG, transport=transport, now=lambda: now)


def test_cached_calls_are_not_repeated(tmp_path):
    calls = []
    c = client(tmp_path, calls)
    c.get("tournaments", sportId=18)
    c.get("tournaments", sportId=18)
    assert len(calls) == 1 and c.billable_this_month() == 1
    assert calls[0][1]["apiKey"] == KEY


def test_budget_stops_billable_calls_but_not_free_ones(tmp_path):
    calls = []
    c = client(tmp_path, calls)
    c.get("fixtures", tournamentId=1)
    c.get("fixtures", tournamentId=2)
    with pytest.raises(BudgetExceeded):
        c.get("fixtures", tournamentId=3)
    c.get("historical-odds", fixtureId="x", bookmakers="pinnacle")   # free: still allowed
    assert len(calls) == 3 and c.billable_this_month() == 2


def test_budget_resets_next_month(tmp_path):
    calls = []
    client(tmp_path, calls).get("fixtures", tournamentId=1)
    client(tmp_path, calls).get("fixtures", tournamentId=2)
    assert client(tmp_path, calls, now=datetime(2026, 11, 1, tzinfo=UTC)).remaining() == 2


def test_key_is_never_written_to_disk(tmp_path):
    calls = []
    c = client(tmp_path, calls)
    c.get("tournaments", sportId=18)
    c.get("historical-odds", fixtureId="x", bookmakers="pinnacle")
    for f in tmp_path.rglob("*"):
        assert KEY not in str(f)
        if f.is_file():
            assert KEY not in f.read_text(encoding="utf-8")


def test_errors_are_counted_and_redact_the_key(tmp_path):
    calls = []
    c = client(tmp_path, calls, status=400)
    with pytest.raises(OddsPapiError) as e:
        c.get("fixtures", tournamentId=1)
    assert KEY not in str(e.value) and "***" in str(e.value)
    assert c.billable_this_month() == 1


def test_load_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDSPAPI_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('ODDSPAPI_KEY="abc"\n', encoding="utf-8")
    assert load_key(env) == "abc"
    with pytest.raises(OddsPapiError):
        load_key(tmp_path / "missing.env")

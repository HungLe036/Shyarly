import pandas as pd
import pytest

from shyarly import dashboard as d


def test_loaders_handle_missing_files(tmp_path):
    assert d.title_odds(tmp_path / "none.csv").empty
    assert d.swiss_records(tmp_path / "none.csv").empty
    assert d.odds_history(tmp_path / "daily").empty
    assert d.odds_history(tmp_path).empty                      # dir exists, no snapshot yet
    assert d.predictions_vs_results(tmp_path / "log.csv", tmp_path / "res.csv").empty
    assert d.live_scores(tmp_path / "none.md") is None


def test_region_share_and_swiss_split():
    odds = pd.DataFrame({"team": ["GEN", "T1", "BLG"], "name": ["Gen.G", "T1", "Bilibili Gaming"],
                         "region": ["LCK", "LCK", "LPL"], "Champion": [0.3, 0.2, 0.1]})
    rs = d.region_share(odds)
    assert rs["region"].tolist() == ["LCK", "LPL"] and rs.loc[0, "champion"] == 0.5
    assert rs.loc[0, "region_name"] == "Korea"
    swiss = pd.DataFrame({"team": ["GEN"], "name": ["Gen.G"], "qualify": [0.85],
                          **{r: [v] for r, v in zip(d.RECORDS, [0.3, 0.3, 0.25, 0.08, 0.05, 0.02])}})
    sp = d.swiss_split(swiss)
    assert len(sp) == 6
    assert sp.loc[sp["group"] == "Advances", "signed"].sum() == pytest.approx(0.85)
    assert sp.loc[sp["group"] == "Eliminated", "signed"].sum() == pytest.approx(-0.15)


def test_match_cards_name_favourite_and_result():
    preds = pd.DataFrame({"match_date_utc": ["2026-10-23T17:00:00+00:00"] * 2, "stage": ["swiss"] * 2,
                          "round": ["1", "1"], "team_a": ["IG", "LOS"], "team_b": ["GEN", "T1"], "best_of": [1, 1],
                          "model": ["bt", "bt"], "mode": ["live", "live"], "p_series_a": [0.36, 0.16],
                          "winner": ["GEN", pd.NA], "score": ["0-1", pd.NA]})
    c = d.match_cards(preds, {"GEN": "Gen.G"})
    assert c.loc[0, "favourite"] == "Gen.G" and c.loc[0, "chance"] == pytest.approx(0.64)
    assert c.loc[0, "result"] == "Gen.G won 0-1" and bool(c.loc[0, "called_it"])
    assert c.loc[1, "result"] == "Not played yet" and pd.isna(c.loc[1, "called_it"])
    assert c.loc[0, "round"] == "Swiss round 1"


def test_sim_runs_reads_the_report(tmp_path):
    md = tmp_path / "t.md"
    md.write_text("# x\n\n- **Simulation:** 200,000 runs, seed 1; each run ...\n", encoding="utf-8")
    assert d.sim_runs(md) == 200000
    assert d.sim_runs(tmp_path / "none.md") is None


def test_header_only_log_is_empty(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("match_id,stage,round,team_a,team_b,match_date_utc,best_of,mode\n")
    assert d.predictions_vs_results(log, tmp_path / "res.csv").empty


def test_history_and_predictions_join(tmp_path):
    daily = tmp_path / "daily"
    daily.mkdir()
    for day, p in [("2026-10-19", 0.2), ("2026-10-20", 0.3)]:
        pd.DataFrame({"team": ["GEN"], "qualify": [0.8], "semis": [0.6], "final": [0.4], "champion": [p],
                      "as_of": [day]}).to_csv(daily / f"{day}_odds.csv", index=False)
    hist = d.odds_history(daily)
    champ = hist[hist["stage"] == "Champion"]
    assert champ["p"].tolist() == [0.2, 0.3]

    log = tmp_path / "log.csv"
    pd.DataFrame({"match_id": ["swiss-1-GEN-IG"] * 2, "stage": ["swiss"] * 2, "round": ["1"] * 2,
                  "team_a": ["GEN"] * 2, "team_b": ["IG"] * 2, "match_date_utc": ["2026-10-23"] * 2,
                  "best_of": [1, 1], "mode": ["live", "live"], "p_game_a": [0.7, 0.9], "p_series_a": [0.7, 0.9],
                  "logged_at_utc": ["a", "b"]}).to_csv(log, index=False)
    res = tmp_path / "res.csv"
    res.write_text("stage,round,team_a,team_b,date_utc,winner,score,source\n"
                   "swiss,1,IG,GEN,2026-10-23T17:00:00Z,GEN,1-0,x\n")
    out = d.predictions_vs_results(log, res)
    assert len(out) == 1 and out.loc[0, "p_series_a"] == 0.7 and out.loc[0, "winner"] == "GEN"

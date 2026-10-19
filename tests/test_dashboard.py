import pandas as pd

from shyarly import dashboard as d


def test_loaders_handle_missing_files(tmp_path):
    assert d.title_odds(tmp_path / "none.csv").empty
    assert d.swiss_records(tmp_path / "none.csv").empty
    assert d.odds_history(tmp_path / "daily").empty
    assert d.odds_history(tmp_path).empty                      # dir exists, no snapshot yet
    assert d.predictions_vs_results(tmp_path / "log.csv", tmp_path / "res.csv").empty
    assert d.live_scores(tmp_path / "none.md") is None


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

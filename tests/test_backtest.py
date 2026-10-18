import pandas as pd

from shyarly.config import load_model_config
from shyarly.eval.backtest import make_queries, param_grid


def _games():
    t = pd.to_datetime(["2024-05-01 10:00", "2024-05-01 23:30", "2024-05-03 02:00", "2024-04-20 10:00"], utc=True)
    return pd.DataFrame({
        "game_id": ["a", "b", "c", "q"], "date_utc": t.astype("datetime64[us, UTC]"),
        "event": ["MSI 2024", "MSI 2024", "MSI 2024", "MSI 2024"],
        "scored": [True, True, True, False], "blue_win": [1, 0, 1, 1],
    })


def test_live_cutoff_is_game_day_start():
    q = make_queries(_games(), ["MSI 2024"], "live").set_index("game_id")
    assert q.loc["b", "cutoff"] == pd.Timestamp("2024-05-01", tz="UTC")
    assert q.loc["c", "cutoff"] == pd.Timestamp("2024-05-03", tz="UTC")
    assert (q["cutoff"] <= q["date_utc"]).all()
    assert "q" not in q.index                       # unscored (qualifier) games are not queried


def test_pre_event_cutoff_is_first_scored_day():
    q = make_queries(_games(), ["MSI 2024"], "pre_event")
    assert (q["cutoff"] == pd.Timestamp("2024-05-01", tz="UTC")).all()


def test_param_grid_covers_product():
    cfg = load_model_config()
    g = param_grid(cfg, "elo")
    sizes = [len(v) for v in cfg["elo"]["grid"].values()]
    assert len(g) == sizes[0] * sizes[1] * sizes[2]
    assert param_grid(cfg, "coinflip") == [{}]

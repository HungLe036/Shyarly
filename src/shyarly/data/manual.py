"""Hand-entered Worlds 2026 match results: data/manual/worlds2026_results.csv.

One row per match (series). A row with an empty `winner` is a drawn pairing that has not
been played yet; it fixes the pairing in the simulation and gets a logged prediction.

    stage     swiss | knockout
    round     1-5 for swiss; QF1, QF2, QF3, QF4, SF1, SF2, F for knockout
              (bracket halves: QF1/QF2 feed SF1, QF3/QF4 feed SF2; see sim/knockout.py)
    team_a, team_b  team codes from config/worlds2026.yaml (order is free)
    date_utc  scheduled start, ISO 8601 UTC, e.g. 2026-10-23T17:00:00Z
    winner    team code, or empty if not played
    score     "a-b" games won by team_a and team_b, e.g. 2-1; empty if not played
    source    where the row came from (URL or short name + access date)

`swiss_fixed` replays the Swiss rounds in order and checks them against the rules: Round 1
must match the official draw, later rounds pair teams with the same record and need the
previous round finished, and each score must fit the best-of for that record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from shyarly.config import WorldsConfig

COLUMNS = ["stage", "round", "team_a", "team_b", "date_utc", "winner", "score", "source"]
KO_ROUNDS = ["QF1", "QF2", "QF3", "QF4", "SF1", "SF2", "F"]
KO_FEEDS = {"SF1": ("QF1", "QF2"), "SF2": ("QF3", "QF4"), "F": ("SF1", "SF2")}
KO_BEST_OF = 5


class ResultsError(ValueError):
    """The results file breaks a format rule or contradicts the tournament config."""


@dataclass(frozen=True)
class MatchRow:
    stage: str
    round: str
    team_a: str
    team_b: str
    date_utc: pd.Timestamp
    winner: str | None
    score: tuple[int, int] | None
    source: str

    @property
    def match_id(self) -> str:
        a, b = sorted((self.team_a, self.team_b))
        return f"{self.stage}-{self.round}-{a}-{b}"


@dataclass
class SwissFixed:
    pairings: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    results: dict[tuple[int, int, int], int] = field(default_factory=dict)   # (round, lo, hi) -> winner
    best_of: dict[str, int] = field(default_factory=dict)                    # match_id -> 1 or 3
    complete_rounds: int = 0
    final_records: list[tuple[int, int]] = field(default_factory=list)       # after the known results


@dataclass
class KnockoutFixed:
    bracket: list[tuple[int, int]] | None = None    # QF1..QF4 pairs, team indices
    results: dict[str, int] = field(default_factory=dict)   # round label -> winner index


def load_results(path: Path, worlds: WorldsConfig) -> list[MatchRow]:
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ResultsError(f"{path.name}: missing columns {missing}")
    codes = {t.code for t in worlds.teams}
    rows: list[MatchRow] = []
    for n, rec in enumerate(df.to_dict("records"), start=2):   # line 1 is the header
        rows.append(_parse_row({k: str(v).strip() for k, v in rec.items()}, codes, f"{path.name} line {n}"))
    seen: set[str] = set()
    slots: set[tuple[str, str, str]] = set()
    for r in rows:
        if r.match_id in seen:
            raise ResultsError(f"duplicate match {r.match_id}")
        seen.add(r.match_id)
        for t in (r.team_a, r.team_b):
            key = (r.stage, r.round, t)
            if key in slots:
                raise ResultsError(f"{t} appears twice in {r.stage} round {r.round}")
            slots.add(key)
    return rows


def _parse_row(rec: dict[str, str], codes: set[str], where: str) -> MatchRow:
    stage, rnd = rec["stage"].lower(), rec["round"].upper()
    if stage not in ("swiss", "knockout"):
        raise ResultsError(f"{where}: stage must be swiss or knockout, got {rec['stage']!r}")
    if stage == "swiss" and rnd not in {"1", "2", "3", "4", "5"}:
        raise ResultsError(f"{where}: swiss round must be 1-5, got {rec['round']!r}")
    if stage == "knockout" and rnd not in KO_ROUNDS:
        raise ResultsError(f"{where}: knockout round must be one of {KO_ROUNDS}, got {rec['round']!r}")
    a, b = rec["team_a"], rec["team_b"]
    for t in (a, b):
        if t not in codes:
            raise ResultsError(f"{where}: unknown team {t!r} (use codes from config/worlds2026.yaml)")
    if a == b:
        raise ResultsError(f"{where}: a team cannot play itself")
    try:
        date = pd.Timestamp(rec["date_utc"])
    except ValueError as e:
        raise ResultsError(f"{where}: bad date_utc {rec['date_utc']!r}") from e
    date = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
    winner = rec["winner"] or None
    score = None
    if winner is not None and winner not in (a, b):
        raise ResultsError(f"{where}: winner {winner} is not in the match {a} vs {b}")
    if rec["score"]:
        try:
            x, y = (int(v) for v in rec["score"].split("-"))
        except ValueError as e:
            raise ResultsError(f"{where}: score must look like 2-1, got {rec['score']!r}") from e
        score = (x, y)
        if winner is None:
            raise ResultsError(f"{where}: score {rec['score']} given without a winner")
        if (x > y) != (winner == a) or x == y:
            raise ResultsError(f"{where}: score {rec['score']} does not match winner {winner}")
    elif winner is not None:
        raise ResultsError(f"{where}: score is required when a winner is set")
    return MatchRow(stage, rnd, a, b, date, winner, score, rec.get("source", ""))


def _check_score(r: MatchRow, best_of: int) -> None:
    if r.score is None:
        return
    need = (best_of + 1) // 2
    if max(r.score) != need or min(r.score) >= need:
        raise ResultsError(f"{r.match_id}: Bo{best_of} match cannot end {r.score[0]}-{r.score[1]}")


def swiss_fixed(rows: list[MatchRow], worlds: WorldsConfig, codes: list[str]) -> SwissFixed:
    """Replay Swiss rows into simulator inputs (fixed pairings and results by team index)."""
    idx = {c: i for i, c in enumerate(codes)}
    sw = worlds.swiss
    out = SwissFixed()
    official = [tuple(p) for p in sw["round1"].get("pairings") or []]
    if official:
        out.pairings[1] = [(idx[a], idx[b]) for a, b in official]
    by_round: dict[int, list[MatchRow]] = {}
    for r in rows:
        if r.stage == "swiss":
            by_round.setdefault(int(r.round), []).append(r)

    wins, losses = [0] * len(codes), [0] * len(codes)
    for rnd in range(1, sw["n_rounds"] + 1):
        rr = by_round.get(rnd, [])
        if not rr:
            if any(k > rnd for k in by_round):
                raise ResultsError(f"Round {rnd} has no rows but a later round does")
            break
        if out.complete_rounds < rnd - 1:
            raise ResultsError(f"Round {rnd - 1} has matches without results; enter them before Round {rnd} pairings")
        active = {i for i in range(len(codes)) if wins[i] < sw["wins_to_advance"] and losses[i] < sw["losses_to_eliminate"]}
        pairs = [(idx[r.team_a], idx[r.team_b]) for r in rr]
        if rnd == 1 and official:
            allowed = {frozenset(p) for p in out.pairings[1]}
            for r, p in zip(rr, pairs):
                if frozenset(p) not in allowed:
                    raise ResultsError(f"{r.match_id}: not a Round 1 pairing in the official draw (config swiss.round1.pairings)")
        else:
            if sorted(i for p in pairs for i in p) != sorted(active):
                raise ResultsError(f"Round {rnd} pairings must cover each active team exactly once")
            out.pairings[rnd] = pairs
        for r, (a, b) in zip(rr, pairs):
            if (wins[a], losses[a]) != (wins[b], losses[b]):
                raise ResultsError(f"{r.match_id}: {r.team_a} ({wins[a]}-{losses[a]}) and {r.team_b} "
                                   f"({wins[b]}-{losses[b]}) do not share a record")
            bo = sw["best_of_by_record"][f"{wins[a]}-{losses[a]}"]
            out.best_of[r.match_id] = bo
            _check_score(r, bo)
        for r, (a, b) in zip(rr, pairs):
            if r.winner is not None:
                w = idx[r.winner]
                out.results[(rnd, min(a, b), max(a, b))] = w
                wins[w] += 1
                losses[b if w == a else a] += 1
        if len(rr) == len(active) // 2 and all(r.winner for r in rr):
            out.complete_rounds = rnd
    out.final_records = list(zip(wins, losses))
    return out


def knockout_fixed(rows: list[MatchRow], swiss: SwissFixed, codes: list[str]) -> KnockoutFixed:
    """Knockout rows into a fixed bracket and results. Needs the full Swiss Stage."""
    ko = {r.round: r for r in rows if r.stage == "knockout"}
    if not ko:
        return KnockoutFixed()
    if swiss.complete_rounds < 5:
        raise ResultsError("Knockout rows need the full Swiss Stage results first")
    idx = {c: i for i, c in enumerate(codes)}
    rec = swiss.final_records
    qf = [ko.get(q) for q in KO_ROUNDS[:4]]
    if any(r is None for r in qf):
        raise ResultsError("Enter all four quarterfinals (QF1-QF4) once the bracket is drawn")
    bracket = [(idx[r.team_a], idx[r.team_b]) for r in qf]
    qualifiers = sorted(i for i, (w, _) in enumerate(rec) if w == 3)
    if sorted(i for p in bracket for i in p) != qualifiers:
        raise ResultsError("Quarterfinal teams must be exactly the eight Swiss qualifiers")
    with_three_oh = [k for k, (a, b) in enumerate(bracket) if (3, 0) in (rec[a], rec[b])]
    for k in with_three_oh:
        a, b = bracket[k]
        if sorted([rec[a], rec[b]]) != [(3, 0), (3, 2)]:
            raise ResultsError(f"{KO_ROUNDS[k]}: a 3-0 team must face a 3-2 team (RULESET 4.3.4.1)")
    if len(with_three_oh) != 2 or {k // 2 for k in with_three_oh} != {0, 1}:
        raise ResultsError("The two 3-0 vs 3-2 quarterfinals must be on opposite halves "
                           "(one in QF1/QF2, one in QF3/QF4; RULESET 4.3.4.2)")
    out = KnockoutFixed(bracket=bracket)
    for label in KO_ROUNDS:
        r = ko.get(label)
        if r is None:
            continue
        if label in KO_FEEDS:
            feeders = [out.results.get(f) for f in KO_FEEDS[label]]
            if None in feeders or sorted(feeders) != sorted((idx[r.team_a], idx[r.team_b])):
                raise ResultsError(f"{label}: teams must be the winners of {' and '.join(KO_FEEDS[label])}")
        _check_score(r, KO_BEST_OF)
        if r.winner is not None:
            out.results[label] = idx[r.winner]
    return out

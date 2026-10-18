"""Monte Carlo of the Worlds 2026 Swiss Stage (rules: config/worlds2026.yaml `swiss`).

Each run plays 5 rounds:
  * Round 1 pairs draw pools (1 v 4, 2 v 3) with no same-region match. Official pairings,
    when known, can be fixed instead.
  * Rounds 2–5 pair teams with the same record at random, with no rematches.
  * Each match is Bo1 or Bo3 by record; a series is won with the closed-form best-of
    probability of the side-averaged game probability.
  * A team stops at 3 wins (advance) or 3 losses (eliminated).

Draw approximation: the ruleset says a conflicting drawn team "will move to the next
available slot" without defining the slot order. Round 1 picks uniformly among all valid
pool pairings, and later rounds use randomized backtracking over valid pairings. If no
rematch-free pairing exists, the rematch rule is waived (ruleset 4.3.3.2), and the run
counts it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from shyarly.models.series import series_win_prob

RECORDS = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]


@dataclass(frozen=True)
class SwissRules:
    n_rounds: int
    wins_to_advance: int
    losses_to_eliminate: int
    best_of_by_record: dict[str, int]
    pool_matchups: tuple[tuple[int, int], ...]
    forbid_same_region_round1: bool
    forbid_rematch: bool

    @classmethod
    def from_config(cls, swiss: dict) -> SwissRules:
        return cls(
            n_rounds=swiss["n_rounds"],
            wins_to_advance=swiss["wins_to_advance"],
            losses_to_eliminate=swiss["losses_to_eliminate"],
            best_of_by_record={k: int(v) for k, v in swiss["best_of_by_record"].items()},
            pool_matchups=tuple(tuple(p) for p in swiss["round1"]["pool_matchups"]),
            forbid_same_region_round1=swiss["round1"]["forbid_same_region"],
            forbid_rematch=swiss["later_rounds"]["forbid_rematch"],
        )


@dataclass(frozen=True)
class Match:
    round: int
    a: int
    b: int
    record: str          # shared "W-L" before the match
    best_of: int
    winner: int


@dataclass
class RunLog:
    matches: list[Match] = field(default_factory=list)
    final: list[tuple[int, int]] = field(default_factory=list)   # (wins, losses) per team
    waived_rematches: int = 0


@dataclass
class SwissResult:
    teams: list[str]
    runs: int
    qualify: np.ndarray          # count per team
    records: np.ndarray          # [team, RECORDS] counts
    waived_rematches: int

    def table(self) -> dict[str, np.ndarray]:
        return {"qualify": self.qualify / self.runs, **{r: self.records[:, i] / self.runs for i, r in enumerate(RECORDS)}}


class SwissSimulator:
    def __init__(self, teams: list[str], regions: list[str], tiers: list[int], p_game: np.ndarray,
                 rules: SwissRules, fixed_pairings: dict[int, list[tuple[int, int]]] | None = None,
                 fixed_results: dict[tuple[int, int, int], int] | None = None):
        """p_game[i, j] = P(i beats j in one game). fixed_pairings: round -> [(a, b), ...].
        fixed_results: (round, a, b) with a < b -> winner index (for rounds already played)."""
        n = len(teams)
        if n != 16 or p_game.shape != (n, n):
            raise ValueError("The Swiss Stage needs exactly 16 teams and a 16x16 probability matrix")
        self.teams, self.regions, self.tiers, self.rules = teams, regions, tiers, rules
        self.fixed_pairings = fixed_pairings or {}
        self.fixed_results = fixed_results or {}
        # Nested lists: scalar lookups in the inner loop are much faster than numpy indexing.
        self.series_p = {bo: series_win_prob(p_game, bo).tolist() for bo in set(rules.best_of_by_record.values())}
        self.round1_options = [self._valid_pool_pairings(a, b) for a, b in rules.pool_matchups]
        for opts, (a, b) in zip(self.round1_options, rules.pool_matchups):
            if not opts:
                raise ValueError(f"No valid Round 1 pairing between pools {a} and {b}")

    # ------------------------------------------------------------- draws

    def _valid_pool_pairings(self, pool_a: int, pool_b: int) -> list[list[tuple[int, int]]]:
        a = [i for i, t in enumerate(self.tiers) if t == pool_a]
        b = [i for i, t in enumerate(self.tiers) if t == pool_b]
        out = []
        for perm in permutations(b):
            pairs = list(zip(a, perm))
            if not self.rules.forbid_same_region_round1 or all(self.regions[x] != self.regions[y] for x, y in pairs):
                out.append(pairs)
        return out

    @staticmethod
    def _random_matching(group: list[int], met: set[frozenset], rng: random.Random,
                         forbid: bool) -> list[tuple[int, int]] | None:
        def backtrack(rem: list[int]) -> list[tuple[int, int]] | None:
            if not rem:
                return []
            a, rest = rem[0], rem[1:]
            order = rest[:]
            rng.shuffle(order)
            for b in order:
                if forbid and frozenset((a, b)) in met:
                    continue
                sub = backtrack([x for x in rest if x != b])
                if sub is not None:
                    return [(a, b)] + sub
            return None

        teams = group[:]
        rng.shuffle(teams)
        return backtrack(teams)

    # --------------------------------------------------------------- run

    def run_once(self, rng: random.Random, series_p: dict[int, list] | None = None) -> RunLog:
        """One Swiss Stage. series_p: per-run series probabilities (default: the fixed matrix)."""
        series_p = series_p or self.series_p
        r = self.rules
        wins, losses = [0] * 16, [0] * 16
        met: set[frozenset] = set()
        log = RunLog()
        for rnd in range(1, r.n_rounds + 1):
            active = [i for i in range(16) if wins[i] < r.wins_to_advance and losses[i] < r.losses_to_eliminate]
            if not active:
                break
            if rnd in self.fixed_pairings:
                pairs = self._check_fixed(rnd, wins, losses, active)
            elif rnd == 1:
                pairs = [pair for opts in self.round1_options for pair in rng.choice(opts)]
            else:
                pairs = []
                groups: dict[tuple[int, int], list[int]] = {}
                for i in active:
                    groups.setdefault((wins[i], losses[i]), []).append(i)
                for key in sorted(groups):
                    m = self._random_matching(groups[key], met, rng, r.forbid_rematch)
                    if m is None:                       # ruleset 4.3.3.2: rule may be waived
                        m = self._random_matching(groups[key], met, rng, forbid=False)
                        log.waived_rematches += 1
                    pairs += m
            for a, b in pairs:
                record = f"{wins[a]}-{losses[a]}"
                bo = r.best_of_by_record[record]
                key = (rnd, min(a, b), max(a, b))
                if key in self.fixed_results:
                    winner = self.fixed_results[key]
                else:
                    winner = a if rng.random() < series_p[bo][a][b] else b
                loser = b if winner == a else a
                wins[winner] += 1
                losses[loser] += 1
                met.add(frozenset((a, b)))
                log.matches.append(Match(rnd, a, b, record, bo, winner))
        log.final = list(zip(wins, losses))
        return log

    def _check_fixed(self, rnd: int, wins, losses, active) -> list[tuple[int, int]]:
        pairs = self.fixed_pairings[rnd]
        seen = sorted(x for p in pairs for x in p)
        if seen != sorted(active):
            raise ValueError(f"Fixed Round {rnd} pairings must cover each active team exactly once")
        for a, b in pairs:
            if (wins[a], losses[a]) != (wins[b], losses[b]):
                raise ValueError(f"Fixed Round {rnd} pairs {self.teams[a]} and {self.teams[b]} with different records")
        return [tuple(p) for p in pairs]

    def _per_run_series(self, sampler: Callable[[int], np.ndarray] | None, runs: int,
                        chunk: int) -> Iterator[dict[int, list] | None]:
        if sampler is None:
            yield from (None for _ in range(runs))
            return
        bos = sorted(set(self.rules.best_of_by_record.values()))
        done = 0
        while done < runs:
            n = min(chunk, runs - done)
            mats = np.asarray(sampler(n))
            if mats.shape != (n, 16, 16):
                raise ValueError(f"sampler({n}) must return shape ({n}, 16, 16), got {mats.shape}")
            series = {bo: series_win_prob(mats, bo).tolist() for bo in bos}
            for k in range(n):
                yield {bo: series[bo][k] for bo in bos}
            done += n

    def simulate(self, runs: int, seed: int, sampler: Callable[[int], np.ndarray] | None = None,
                 chunk: int = 2000) -> SwissResult:
        """Play `runs` Swiss Stages. With `sampler(n)` -> (n, 16, 16) game-probability matrices,
        each run uses its own matrix (e.g. a posterior draw of team strengths), so strength
        uncertainty is shared across all matches of that run."""
        rng = random.Random(seed)
        qualify = np.zeros(16, dtype=np.int64)
        records = np.zeros((16, len(RECORDS)), dtype=np.int64)
        idx = {rec: k for k, rec in enumerate(RECORDS)}
        waived = 0
        for series_p in self._per_run_series(sampler, runs, chunk):
            log = self.run_once(rng, series_p)
            waived += log.waived_rematches
            for i, (w, l) in enumerate(log.final):
                records[i, idx[f"{w}-{l}"]] += 1
                qualify[i] += w >= self.rules.wins_to_advance
        return SwissResult(self.teams, runs, qualify, records, waived)

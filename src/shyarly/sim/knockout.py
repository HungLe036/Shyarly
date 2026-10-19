"""Worlds 2026 Knockout Stage (rules: config/worlds2026.yaml `knockout`, RULESET 4.1.3 and 4.3.4).

Eight Swiss qualifiers play a single-elimination Bo5 bracket.
  * QF draw (4.3.4.1): the two 3-0 teams are each paired with a random 3-2 team; the other
    four teams (the three 3-1 teams and the last 3-2 team) are paired at random.
  * Bracket (4.3.4.2): the two 3-0 vs 3-2 matches go on opposite halves; the other two
    matches are placed by draw order, modelled as a random half each. No restrictions (4.3.4.3).
Labels: QF1/QF2 feed SF1, QF3/QF4 feed SF2, SF1/SF2 feed F (same convention as the results file).
"""

from __future__ import annotations

import random

from shyarly.data.manual import KO_FEEDS, KO_ROUNDS

QF_LABELS = KO_ROUNDS[:4]
FEEDS = KO_FEEDS


def draw_bracket(final: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    """QF1..QF4 pairs (team indices) from the Swiss final records."""
    by: dict[tuple[int, int], list[int]] = {}
    for i, rec in enumerate(final):
        by.setdefault(rec, []).append(i)
    three_oh, three_one, three_two = by.get((3, 0), []), by.get((3, 1), []), by.get((3, 2), [])
    if (len(three_oh), len(three_one), len(three_two)) != (2, 3, 3):
        raise ValueError("Swiss standings must have two 3-0, three 3-1 and three 3-2 teams")
    t32 = three_two[:]
    rng.shuffle(t32)
    seeded = [(three_oh[0], t32[0]), (three_oh[1], t32[1])]
    rest = three_one + [t32[2]]
    rng.shuffle(rest)
    others = [(rest[0], rest[1]), (rest[2], rest[3])]
    rng.shuffle(seeded)
    rng.shuffle(others)
    # Each half holds one seeded (3-0 vs 3-2) match and one other match, in random order.
    halves = [[seeded[0], others[0]], [seeded[1], others[1]]]
    for h in halves:
        rng.shuffle(h)
    return halves[0] + halves[1]


def knockout_once(bracket: list[tuple[int, int]], p_bo5: list[list[float]], rng: random.Random,
                  fixed_results: dict[str, int] | None = None) -> dict[str, int]:
    """Play the bracket. p_bo5[i][j] = P(i beats j in a Bo5). Returns winner index per round label."""
    fixed = fixed_results or {}
    winners: dict[str, int] = {}

    def play(label: str, a: int, b: int) -> None:
        winners[label] = fixed[label] if label in fixed else (a if rng.random() < p_bo5[a][b] else b)

    for label, (a, b) in zip(QF_LABELS, bracket):
        play(label, a, b)
    for label, (x, y) in FEEDS.items():
        play(label, winners[x], winners[y])
    return winners

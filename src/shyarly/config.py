"""Typed loaders for the YAML files in config/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from shyarly import paths


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- sources.yaml


@dataclass(frozen=True)
class OracleElixirSource:
    download_url: str
    filename: str
    refresh_years: tuple[int, ...]
    file_ids: dict[int, str]

    def url(self, year: int) -> str:
        return self.download_url.format(file_id=self.file_ids[year])

    def local_name(self, year: int) -> str:
        return self.filename.format(year=year)


@dataclass(frozen=True)
class LeaguepediaSource:
    api_url: str
    min_interval_s: float
    max_retries: int
    backoff_s: float
    user_agent: str


@dataclass(frozen=True)
class Sources:
    oracles_elixir: OracleElixirSource
    leaguepedia: LeaguepediaSource
    rules: dict[str, str]


def load_sources(path: Path = paths.SOURCES_CONFIG) -> Sources:
    raw = _read_yaml(path)
    oe = raw["oracles_elixir"]
    return Sources(
        oracles_elixir=OracleElixirSource(
            download_url=oe["download_url"],
            filename=oe["filename"],
            refresh_years=tuple(oe["refresh_years"]),
            file_ids={int(y): str(i) for y, i in oe["file_ids"].items()},
        ),
        leaguepedia=LeaguepediaSource(**raw["leaguepedia"]),
        rules=dict(raw["rules"]),
    )


# ----------------------------------------------------------------- leagues.yaml


@dataclass(frozen=True)
class LeagueMap:
    scored_events: frozenset[str]
    other_events: frozenset[str]
    # (league code, year) -> (region, tier)
    _entries: tuple[tuple[str, int | None, int | None, str, int], ...]

    @property
    def international_events(self) -> frozenset[str]:
        return self.scored_events | self.other_events

    def lookup(self, league: str, year: int) -> tuple[str, int] | None:
        for code, first, last, region, tier in self._entries:
            if code == league and (first is None or first <= year <= last):
                return region, tier
        return None


def load_leagues(path: Path = paths.LEAGUES_CONFIG) -> LeagueMap:
    raw = _read_yaml(path)
    entries = []
    for e in raw["leagues"]:
        first, last = e.get("years", (None, None))
        entries.append((str(e["code"]), first, last, str(e["region"]), int(e["tier"])))
    ev = raw["international_events"]
    return LeagueMap(
        scored_events=frozenset(ev["scored"]),
        other_events=frozenset(ev["other"]),
        _entries=tuple(entries),
    )


# ------------------------------------------------------------------- model.yaml


def load_model_config(path: Path = paths.MODEL_CONFIG) -> dict[str, Any]:
    return _read_yaml(path)


# -------------------------------------------------------------- worlds2026.yaml


@dataclass(frozen=True)
class Team:
    code: str
    name: str
    league: str
    seed: int | None
    tier: int
    status: str  # "qualified" | "pending"
    candidates: tuple[str, ...] = ()
    oe_name: str | None = None


@dataclass(frozen=True)
class WorldsConfig:
    raw: dict[str, Any]
    regions: dict[str, str]
    teams: tuple[Team, ...]
    play_in_teams: tuple[Team, ...] = ()

    def candidate(self, code: str) -> Team:
        """A Play-In team placed into the pending Swiss slot (its tier)."""
        slot = next(t for t in self.teams if t.status == "pending")
        if code not in slot.candidates:
            raise KeyError(f"{code} is not a candidate for the pending slot {slot.code}")
        t = next(p for p in self.play_in_teams if p.code == code)
        return Team(code=t.code, name=t.name, league=t.league, seed=t.seed, tier=slot.tier,
                    status="qualified", oe_name=t.oe_name)

    @property
    def swiss(self) -> dict[str, Any]:
        return self.raw["swiss"]

    @property
    def knockout(self) -> dict[str, Any]:
        return self.raw["knockout"]

    def team(self, code: str) -> Team:
        for t in self.teams:
            if t.code == code:
                return t
        raise KeyError(code)


def load_worlds(path: Path = paths.WORLDS_CONFIG) -> WorldsConfig:
    raw = _read_yaml(path)
    teams = tuple(
        Team(
            code=t["code"],
            name=t["name"],
            league=t["league"],
            seed=t.get("seed"),
            tier=int(t["tier"]),
            status=t.get("status", "qualified"),
            candidates=tuple(t.get("candidates", ())),
            oe_name=t.get("oe_name"),
        )
        for t in raw["teams"]
    )
    play_in = tuple(
        Team(code=t["code"], name=t["name"], league=t["league"], seed=t.get("seed"), tier=0,
             status="play_in", oe_name=t.get("oe_name"))
        for t in raw["play_in"]["teams"]
    )
    return WorldsConfig(raw=raw, regions=dict(raw["regions"]), teams=teams, play_in_teams=play_in)

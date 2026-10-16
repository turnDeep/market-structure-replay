"""提案1: 主要高値圏と下落局面を、自分の注文・建玉から独立して保持する。

近接した高値を同じ固定価格帯に関連付ける。主要性は高値同士の差ではなく、
その高値からの下落幅と期間で判断する。将来の高値で過去の帯を動かさない。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from .common import Bar, Episode, HighArea, MarketSnapshot, Pivot, Rules, check_bar, positive


@dataclass
class AreaState:
    id: str
    anchor: Pivot
    low: float
    high: float
    known_at: datetime
    evidence_ids: list[str] = field(default_factory=list)
    qualified_index: int | None = None
    accepted_breakout: bool = False
    outside_closes: int = 0
    low_bar: Bar | None = None


class MarketEpisodes:
    def __init__(self, rules: Rules):
        self.rules = rules
        self.bars: list[Bar] = []
        self.areas: dict[str, AreaState] = {}
        self.seen_pivots: dict[str, Pivot] = {}
        self.episode: Episode | None = None
        self.reversal_reference: Pivot | None = None
        self.history: list[MarketSnapshot] = []
        self.events: list[dict] = []
        self.counter = 0

    def on_close(self, bar: Bar, pivots: tuple[Pivot, ...]) -> MarketSnapshot:
        check_bar(bar, self.bars[-1] if self.bars else None)
        for p in sorted(pivots, key=lambda p: (p.available_index, p.origin_index, p.id)):
            if p.available_index > bar.index or p.available_at > bar.at:
                raise ValueError("future pivot")
            if p.id in self.seen_pivots:
                if self.seen_pivots[p.id] != p:
                    raise ValueError("historical pivot rewritten")
                continue
            self.seen_pivots[p.id] = p
            if p.kind != "high" or p.radius < self.rules.pivot_radius or not positive(p.atr_at_origin):
                continue
            matches = [a for a in self.areas.values()
                       if not a.accepted_breakout and a.low <= p.price <= a.high
                       and bar.index - a.anchor.origin_index <= self.rules.area_max_age]
            if matches:
                a = min(matches, key=lambda a: (abs(a.anchor.price - p.price), a.id))
                a.evidence_ids.append(p.id)
                self.events.append(dict(at=bar.at, kind="high_joined_existing_area", area_id=a.id, pivot=p.id))
            else:
                tol = self.rules.area_tolerance_atr * p.atr_at_origin
                a = AreaState(f"area:{p.id}", p, p.price - tol, p.price + tol,
                              bar.at, [p.id])
                # 確認待ち期間の価格は、確認された現在からのみ解釈する。
                prior = [b for b in self.bars if b.index > p.origin_index]
                if prior:
                    a.low_bar = min(prior, key=lambda b: (b.low, b.index))
                for old in prior:
                    a.outside_closes = a.outside_closes + 1 if old.close > a.high else 0
                    if a.outside_closes >= self.rules.breakout_accept_closes:
                        a.accepted_breakout = True
                self.areas[a.id] = a
                self.events.append(dict(at=bar.at, kind="high_area_known", area_id=a.id))

        for a in self.areas.values():
            if a.low_bar is None or bar.low < a.low_bar.low:
                a.low_bar = bar
            if a.accepted_breakout:
                continue
            a.outside_closes = a.outside_closes + 1 if bar.close > a.high else 0
            if a.outside_closes >= self.rules.breakout_accept_closes:
                a.accepted_breakout = True
                self.events.append(dict(at=bar.at, kind="high_area_breakout_accepted", area_id=a.id))
                continue
            depth = (a.anchor.price - bar.close) / a.anchor.atr_at_origin
            age = bar.index - a.anchor.origin_index
            if (a.qualified_index is None and age >= self.rules.major_decline_bars
                    and depth >= self.rules.major_decline_atr):
                a.qualified_index = bar.index
                self.events.append(dict(at=bar.at, kind="major_decline_qualified", area_id=a.id))

        eligible = [a for a in self.areas.values() if a.qualified_index is not None
                    and not a.accepted_breakout
                    and bar.index - a.anchor.origin_index <= self.rules.area_max_age
                    and (a.anchor.price - bar.close) / a.anchor.atr_at_origin >= self.rules.major_decline_atr]
        # 進行中の下落では、局所的な戻り高値へアンカーを差し替えない。
        if self.episode is None:
            choice = max(eligible, key=lambda a: a.anchor.origin_index) if eligible else None
        elif self.episode.phase == "advancing":
            choices = [a for a in eligible
                       if a.anchor.origin_index > (self.episode.last_reversal_index or self.episode.started_index)]
            choice = max(choices, key=lambda a: a.anchor.origin_index) if choices else None
        else:
            choice = None
        if choice is not None:
            self.counter += 1
            low = choice.low_bar or bar
            self.episode = Episode(f"episode:{self.counter}", choice.id, choice.anchor,
                                   bar.index, low.low, low.index, low.at, low.atr_prior,
                                   "declining", None)
            self.reversal_reference = None
            self.events.append(dict(at=bar.at, kind="new_market_episode", episode_id=self.episode.id))

        ep = self.episode
        if ep is not None:
            if (ep.phase == "advancing" and self.reversal_reference is not None
                    and bar.close < self.reversal_reference.price):
                ep = replace(ep, phase="declining")
                self.events.append(dict(at=bar.at, kind="reversal_boundary_lost", episode_id=ep.id))
            if bar.low < ep.low:
                # 反転失敗でもepisode IDと試行予算を維持する。
                ep = replace(ep, low=bar.low, low_index=bar.index, low_at=bar.at,
                             low_atr=bar.atr_prior, phase="declining")
                self.events.append(dict(at=bar.at, kind="episode_new_low", episode_id=ep.id))
            self.episode = ep
        self.bars.append(bar)
        snapshot = MarketSnapshot(bar.at, tuple(
            HighArea(a.id, a.anchor, a.low, a.high, tuple(a.evidence_ids), a.known_at,
                     a.qualified_index, not a.accepted_breakout
                     and bar.index - a.anchor.origin_index <= self.rules.area_max_age)
            for a in sorted(self.areas.values(), key=lambda a: a.id)), self.episode)
        self.history.append(snapshot)
        return snapshot

    def confirm_upward_structure(self, episode_id: str, bar: Bar, reference: Pivot) -> None:
        if not self.bars or self.bars[-1] != bar:
            raise ValueError("confirmation must follow this close observation")
        if self.episode and self.episode.id == episode_id:
            self.episode = replace(self.episode, phase="advancing", last_reversal_index=bar.index)
            self.reversal_reference = reference
            self.events.append(dict(at=bar.at, kind="upward_structure_confirmed", episode_id=episode_id))

    def note_failed_reversal(self, bar: Bar, reference: Pivot) -> None:
        """主要な売り確認を市場へ通知。自分の損切り通知からは呼ばない。"""
        if not self.bars or self.bars[-1] != bar:
            raise ValueError("current close required")
        if self.episode and bar.close < reference.price:
            self.episode = replace(self.episode, phase="declining")
            self.events.append(dict(at=bar.at, kind="market_reversal_failed", episode_id=self.episode.id))

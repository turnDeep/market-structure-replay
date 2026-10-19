"""共通型。数値は未検証の設計値で、ICT原典の公式閾値ではない。

indexは観測日足の連番。ATRは前日まで、週月足は確定足のみ。
SELLはロングの決済。特定暦日、手動Pivot、LuxAlgoを条件に使わない。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import floor, isfinite


@dataclass(frozen=True)
class Rules:
    pivot_radius: int = 5
    buy_reference_radius: int = 2
    sell_support_radius: int = 2
    area_tolerance_atr: float = 0.5
    area_max_age: int = 504
    major_decline_atr: float = 3.0
    major_decline_bars: int = 15
    breakout_accept_closes: int = 3
    # 保有管理: 小さな調整では保護安値を引き上げない。
    management_pullback_bars: int = 20
    management_pullback_atr: float = 2.5
    structural_exit_closes: int = 2
    # 第2山への上昇を支えた安値の規模。
    top_min_separation: int = 20
    top_lifetime: int = 60
    sell_support_bars: int = 5
    sell_support_rise_atr: float = 2.0
    sell_departure_atr: float = 1.0
    sell_confirm_closes: int = 2
    sell_mode: str = "close_sequence"  # close_sequence / breakdown_retest
    # OB反応後に、下降を維持していた戻り高値の突破を待つ。
    reaction_lifetime: int = 60
    buy_reference_drop_atr: float = 1.5
    buy_reference_bars: int = 3
    buy_departure_atr: float = 1.0
    buy_confirm_closes: int = 2
    buy_mode: str = "close_sequence"  # close_sequence / breakout_retest
    structure_buffer_atr: float = 0.05
    retest_tolerance_atr: float = 0.1
    require_linked_fvg: bool = False
    emergency_buffer_atr: float = 0.1
    tick_size: float = 0.25
    # 同じ主要下落局面内で共通予算を使う。
    max_attempts_per_episode: int = 2
    max_loss_r_per_episode: float = 2.0
    retry_new_low_atr: float = 0.5
    retry_higher_low_atr: float = 0.5


@dataclass(frozen=True)
class Bar:
    index: int
    at: datetime
    open: float
    high: float
    low: float
    close: float
    atr_prior: float


@dataclass(frozen=True)
class Pivot:
    id: str
    kind: str
    origin_index: int
    origin_at: datetime
    available_index: int
    available_at: datetime
    price: float
    atr_at_origin: float
    radius: int = 5


@dataclass(frozen=True)
class Zone:
    id: str
    kind: str
    direction: str
    timeframe: str
    origin_index: int
    origin_at: datetime
    available_index: int
    available_at: datetime
    body_low: float
    body_high: float
    wick_low: float
    wick_high: float


@dataclass(frozen=True)
class ZoneView:
    zone: Zone
    observed_at: datetime
    state: str  # active / recovered / reclaim_wait / invalidated / expired / filled

    @property
    def observable(self) -> bool:
        return self.state in {"active", "recovered", "reclaim_wait"}

    @property
    def tradeable(self) -> bool:
        return self.state in {"active", "recovered"}


@dataclass(frozen=True)
class HigherTimeframe:
    at: datetime
    weekly_completed_at: datetime | None
    monthly_completed_at: datetime | None
    weekly_trend: str = "unknown"
    monthly_trend: str = "unknown"
    weekly_bias: str = "unknown"
    monthly_bias: str = "unknown"


@dataclass(frozen=True)
class HighArea:
    id: str
    anchor: Pivot
    low: float
    high: float
    evidence_ids: tuple[str, ...]
    known_at: datetime
    qualified_index: int | None
    active: bool


@dataclass(frozen=True)
class Episode:
    id: str
    area_id: str
    anchor: Pivot
    started_index: int
    low: float
    low_index: int
    low_at: datetime
    low_atr: float
    phase: str  # declining / advancing
    last_reversal_index: int | None


@dataclass(frozen=True)
class MarketSnapshot:
    at: datetime
    areas: tuple[HighArea, ...]
    episode: Episode | None


@dataclass(frozen=True)
class BuyEvidence:
    key: str
    episode_id: str
    at: datetime
    index: int
    zone: Zone
    touch_index: int
    low: float
    low_index: int
    low_atr: float
    recovery_index: int
    reference: Pivot
    breakout_index: int
    confirmation: str
    emergency_stop: float
    linked_fvg_ids: tuple[str, ...]
    htf_overlap_ids: tuple[str, ...]
    sweep_depth_atr: float
    htf: HigherTimeframe
    ltf_conditions: str = "unknown_data"
    # Reaction-level identity shared by early and confirmed BUY variants.
    # It is intentionally separate from ``key`` because the latter includes
    # the model-specific confirmation bar.
    setup_id: str | None = None


@dataclass(frozen=True)
class SellEvidence:
    key: str
    at: datetime
    index: int
    reason: str
    source_id: str
    reference: Pivot


@dataclass(frozen=True)
class ProtectedLow:
    pivot: Pivot
    supporting_high: Pivot
    breakout_index: int
    decided_index: int
    effective_from_index: int


@dataclass(frozen=True)
class Position:
    id: str
    candidate: BuyEvidence
    entry_index: int
    entry_at: datetime
    entry_price: float
    quantity: int
    point_value: float
    initial_risk_usd: float


@dataclass(frozen=True)
class OrderIntent:
    id: str
    side: str
    at: datetime
    index: int
    reason: str
    candidate: BuyEvidence | None = None
    position_id: str | None = None


def positive(value: float) -> bool:
    return isfinite(value) and value > 0


def down_tick(value: float, tick: float) -> float:
    if not positive(tick):
        raise ValueError("positive tick required")
    return floor(value / tick) * tick


def check_bar(bar: Bar, previous: Bar | None) -> None:
    if not all(isfinite(v) for v in (bar.open, bar.high, bar.low, bar.close)):
        raise ValueError("nonfinite OHLC")
    if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
        raise ValueError("invalid OHLC ordering")
    if previous and not (bar.index == previous.index + 1 and bar.at > previous.at):
        raise ValueError("feed each observed daily bar exactly once")


def check_inputs(bar: Bar, pivots: tuple[Pivot, ...], zones: tuple[ZoneView, ...],
                 htf: HigherTimeframe) -> None:
    if htf.at != bar.at:
        raise ValueError("higher-timeframe snapshot must be point-in-time")
    for completed in (htf.weekly_completed_at, htf.monthly_completed_at):
        if completed is not None and completed > bar.at:
            raise ValueError("incomplete higher-timeframe bar supplied")
    for p in pivots:
        if not (p.origin_index <= p.available_index <= bar.index
                and p.origin_at <= p.available_at <= bar.at):
            raise ValueError("future or inconsistent pivot")
    for view in zones:
        z = view.zone
        if view.observed_at != bar.at:
            raise ValueError("stale zone state")
        if not (z.origin_index <= z.available_index <= bar.index
                and z.origin_at <= z.available_at <= bar.at):
            raise ValueError("future or inconsistent zone")
        if not z.wick_low <= z.body_low <= z.body_high <= z.wick_high:
            raise ValueError("invalid zone geometry")

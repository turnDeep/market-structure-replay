"""Causal daily OB-retest and double-top trade replay; implementation only.

Consumes immutable formation fields from the existing ICT detector and dated
weekly/monthly contexts. It does NOT use complete_adapted, outcome labels,
terminal_state, first_touch, or the detector's full-sample contact counts.
All new contacts, reactions, orders and positions are reconstructed forward.

This is a daily adaptation, not an intraday ICT execution reproduction. Default
behaviour is long-only, close-time decisions and next-observed-session open
fills. No network, broker, parameter search or performance testing is invoked.
The program has deliberately not been executed as part of its initial delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "nq-ict-retest-trade-0.1-unvalidated"
ROOT = Path(__file__).resolve().parent
OBJECT_COLUMNS = (
    "object_id", "symbol", "timeframe", "object_type", "rule_id",
    "sensitivity", "direction", "origin_time", "available_at", "zone_low",
    "zone_high", "wick_zone_low", "wick_zone_high", "evidence_state",
    "source_rule_state", "rule_version", "definition_variant",
)
# Deliberately exclude precomputed zone-lifecycle and future outcome fields.
CONTEXT_FIELDS = (
    "structure_trend", "structure_integrity", "bias_direction", "bias_status",
    "bias_reason", "bias_target_price", "position", "context_bar_end",
    "range_low", "range_high", "protected_high", "protected_low",
)
TF_RANK = {"1D": 1, "1W": 2, "1M": 3}


@dataclass(frozen=True)
class Config:
    """All thresholds are provisional engineering choices, not fitted values."""

    sensitivity: str = "balanced"
    zone_timeframes: tuple[str, ...] = ("1D", "1W", "1M")
    ob_rule_ids: tuple[str, ...] = ()  # Empty accepts the supplied detector variant.
    atr_period: int = 14
    local_pivot_radius: int = 2
    major_pivot_radius: int = 5
    trailing_pivot_radius: int = 5
    local_reference_lookback: int = 40
    max_zone_age_daily_bars: dict[str, int] = field(
        default_factory=lambda: {"1D": 504, "1W": 1512, "1M": 3024}
    )
    episode_timeout_bars: int = 20
    departure_bars: int = 2
    departure_atr: float = 0.50
    reclaim_grace_bars: int = 5
    allow_reactivated_zones: bool = True
    zone_invalidation_boundary: str = "wick"  # wick or zone
    zone_invalidation_clock: str = "source_close"  # source_close or daily_close
    reaction_recovery_boundary: str = "near"  # near or distal
    confirmation_mode: str = "structure"  # structure or recovery
    displacement_bars: int = 3
    displacement_atr: float = 0.50
    displacement_efficiency: float = 0.50
    require_reaction_fvg: bool = False
    allowed_entry_contexts: tuple[str, ...] = (
        "aligned_continuation", "mixed_timeframes", "countertrend_turn"
    )
    nonaligned_requires_htf_zone: bool = True
    nonaligned_requires_structure: bool = True
    double_top_min_separation: int = 20
    double_top_max_separation: int = 252
    double_top_tolerance_atr: float = 0.50
    double_top_max_sweep_atr: float = 1.50
    double_top_min_depth_atr: float = 1.50
    double_top_approach_atr: float = 0.25
    double_top_timeout_bars: int = 40
    double_top_exit_mode: str = "local_structure"  # local_structure or neckline
    stop_buffer_atr: float = 0.10
    trailing_stop_enabled: bool = True
    trailing_buffer_atr: float = 0.10
    max_entry_gap_atr: float = 1.00
    max_holding_bars: int | None = None
    contracts: int = 1
    point_value_usd: float = 20.0
    tick_size: float = 0.25
    slippage_ticks: float = 1.0
    commission_per_contract_side_usd: float = 2.50

    def check(self) -> None:
        """Input contract enforced when a future user explicitly runs the file."""
        for name in (
            "atr_period", "local_pivot_radius", "major_pivot_radius",
            "trailing_pivot_radius", "local_reference_lookback", "departure_bars",
            "episode_timeout_bars", "reclaim_grace_bars", "displacement_bars",
            "double_top_min_separation", "double_top_max_separation",
            "double_top_timeout_bars", "contracts",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "departure_atr", "displacement_atr", "double_top_tolerance_atr",
            "double_top_max_sweep_atr", "double_top_min_depth_atr",
            "double_top_approach_atr", "stop_buffer_atr", "trailing_buffer_atr",
            "max_entry_gap_atr", "point_value_usd", "tick_size",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.displacement_efficiency <= 1:
            raise ValueError("displacement_efficiency must be in (0,1]")
        if self.double_top_max_separation < self.double_top_min_separation:
            raise ValueError("double-top separation bounds are reversed")
        if self.double_top_max_sweep_atr < self.double_top_tolerance_atr:
            raise ValueError("maximum sweep must cover the tolerance band")
        for name, choices in {
            "zone_invalidation_boundary": {"wick", "zone"},
            "zone_invalidation_clock": {"source_close", "daily_close"},
            "reaction_recovery_boundary": {"near", "distal"},
            "confirmation_mode": {"structure", "recovery"},
            "double_top_exit_mode": {"local_structure", "neckline"},
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"Invalid {name}: {getattr(self, name)}")
        if not set(self.zone_timeframes) <= set(TF_RANK):
            raise ValueError("Unsupported zone timeframe")
        for tf in self.zone_timeframes:
            n = self.max_zone_age_daily_bars.get(tf)
            if not isinstance(n, int) or n < 1:
                raise ValueError(f"Positive maximum zone age required for {tf}")
        valid_contexts = {"aligned_continuation", "mixed_timeframes",
                          "countertrend_turn", "context_unknown"}
        if not set(self.allowed_entry_contexts) <= valid_contexts:
            raise ValueError("Unknown entry context")
        if self.max_holding_bars is not None and self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be positive or null")
        for v in (self.slippage_ticks, self.commission_per_contract_side_usd):
            if not math.isfinite(v) or v < 0:
                raise ValueError("Costs must be nonnegative and finite")


def number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is pd.NaT or value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path | None) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8-sig")) if path else {}
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("zone_timeframes", "ob_rule_ids", "allowed_entry_contexts"):
        if key in raw:
            raw[key] = tuple(raw[key])
    cfg = Config(**raw)
    cfg.check()
    return cfg


def normalized_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    if dates.dt.tz is not None:
        raise ValueError("Use timezone-free trading-date labels; do not silently convert")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Expected non-null daily trading-date labels")
    return dates


def load_inputs(prices: Path, objects: Path, contexts: Path, cfg: Config,
                as_of: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(prices, usecols=["datetime", "open", "high", "low", "close", "volume"])
    daily["datetime"] = normalized_dates(daily.datetime)
    if daily.datetime.duplicated().any() or not daily.datetime.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing")
    if as_of:
        daily = daily[daily.datetime <= pd.Timestamp(as_of).normalize()].copy()
    if daily.empty:
        raise ValueError("No price bars in requested range")
    cols = ["open", "high", "low", "close", "volume"]
    daily[cols] = daily[cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(daily[cols].to_numpy(float)).all():
        raise ValueError("OHLCV contains nonfinite values; input repair is out of scope")
    if ((daily.low > daily[["open", "close", "high"]].min(axis=1)) |
            (daily.high < daily[["open", "close", "low"]].max(axis=1)) |
            (daily.volume < 0) | (daily.low <= 0)).any():
        raise ValueError("Invalid OHLCV values")
    daily = daily.reset_index(drop=True)
    tr = pd.concat([daily.high - daily.low,
                    (daily.high - daily.close.shift()).abs(),
                    (daily.low - daily.close.shift()).abs()], axis=1).max(axis=1)
    # Freeze volatility at the preceding close for all current-day comparisons.
    daily["atr_prior"] = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean().shift()

    obj = pd.read_csv(objects, usecols=lambda c: c in OBJECT_COLUMNS)
    required = {"object_id", "timeframe", "object_type", "rule_id", "sensitivity",
                "direction", "origin_time", "available_at", "zone_low", "zone_high",
                "evidence_state"}
    if not required <= set(obj.columns):
        raise ValueError(f"Missing object columns: {required - set(obj.columns)}")
    obj = obj[obj.object_type.isin(["ob_candidate", "fvg"]) &
              obj.sensitivity.eq(cfg.sensitivity) &
              obj.timeframe.isin(cfg.zone_timeframes)].copy()
    if cfg.ob_rule_ids:
        obj = obj[obj.object_type.ne("ob_candidate") | obj.rule_id.isin(cfg.ob_rule_ids)]
    obj["origin_time"] = normalized_dates(obj.origin_time)
    obj["source_available_at"] = normalized_dates(obj.available_at)
    # Weekly and monthly detector rows can close on a holiday or weekend. The
    # information becomes actionable on the first observed daily bar on/after
    # that close; retaining the source date prevents a false same-day decision.
    daily_axis = daily.datetime.to_numpy(dtype="datetime64[ns]")
    effective_idx = np.searchsorted(
        daily_axis, obj["source_available_at"].to_numpy(dtype="datetime64[ns]"), side="left"
    )
    obj["available_at"] = [daily.datetime.iloc[i] if i < len(daily) else pd.NaT
                            for i in effective_idx]
    obj = obj[obj.available_at.notna()].copy()
    if obj.object_id.duplicated().any():
        raise ValueError("Object IDs must be unique; do not mix detector outputs")
    if (obj.origin_time > obj.source_available_at).any():
        raise ValueError("An object is available before its origin")
    if "symbol" in obj and not obj.symbol.dropna().eq("NQ=F").all():
        raise ValueError("Unexpected symbol in detector objects")
    if not obj.direction.isin(["bullish", "bearish"]).all():
        raise ValueError("Unsupported object direction")
    if (obj.available_at < daily.datetime.iloc[0]).any():
        raise ValueError("Price input must include history since all supplied objects became available")
    if not obj.available_at.isin(daily.datetime).all():
        raise ValueError("Object availability dates must map to the daily trading-date axis")
    obj = obj.sort_values(["available_at", "object_id"]).reset_index(drop=True)

    allowed = {"datetime"} | {f"{tf}_{k}" for tf in ("1w", "1m") for k in CONTEXT_FIELDS}
    allowed |= {"1w_source_context_bar_end", "1m_source_context_bar_end"}
    ctx = pd.read_csv(contexts, usecols=lambda c: c in allowed)
    ctx["datetime"] = normalized_dates(ctx.datetime)
    if ctx.datetime.duplicated().any():
        raise ValueError("Duplicate context dates")
    ctx = ctx[ctx.datetime.isin(daily.datetime)].sort_values("datetime")
    if not set(daily.datetime) <= set(ctx.datetime):
        raise ValueError("An explicit context row is required for each daily bar")
    for tf in ("1w", "1m"):
        column = f"{tf}_context_bar_end"
        if column not in ctx:
            raise ValueError(f"Missing context availability evidence: {column}")
        end = pd.to_datetime(ctx[column], errors="raise")
        if (end > ctx.datetime).any():
            raise ValueError(f"Future completed bar in {tf} context")
        ctx[f"{tf}_source_context_bar_end"] = end
        effective_idx = np.searchsorted(
            daily_axis, end.fillna(pd.Timestamp("2262-04-11")).to_numpy(dtype="datetime64[ns]"), side="left"
        )
        if ((effective_idx >= len(daily)) & end.notna().to_numpy()).any():
            raise ValueError(f"Context completion occurs after the price data for {tf}")
        ctx[column] = [pd.NaT if pd.isna(source) else daily.datetime.iloc[i]
                       for source, i in zip(end, effective_idx)]
    return daily, obj, ctx.set_index("datetime")


@dataclass
class Pivot:
    id: str
    kind: str
    radius: int
    index: int
    origin_at: pd.Timestamp
    available_at: pd.Timestamp
    price: float
    atr: float


@dataclass
class Episode:
    id: str
    zone_id: str
    number: int
    start_index: int
    start_at: pd.Timestamp
    reference: Pivot | None
    extreme: float
    status: str = "observing"
    wick_swept: bool = False
    band_close_breached: bool = False
    recovered: bool = False
    recovery_at: pd.Timestamp | None = None
    reference_ready_at: pd.Timestamp | None = None
    signalled: bool = False
    departed_bars: int = 0
    context_at_touch: dict[str, Any] = field(default_factory=dict)


@dataclass
class Zone:
    id: str
    timeframe: str
    kind: str
    direction: str
    rule_id: str
    origin_at: pd.Timestamp
    source_available_at: pd.Timestamp
    available_at: pd.Timestamp
    low: float
    high: float
    wick_low: float
    wick_high: float
    source_state: str
    state: str = "active"
    touches: int = 0
    breach_index: int | None = None
    breach_count: int = 0
    recovery_count: int = 0
    episode: Episode | None = None
    armed: bool = True
    away_bars: int = 0


@dataclass
class DoubleTop:
    id: str
    first: Pivot
    start_index: int
    start_at: pd.Timestamp
    second_high: float
    second_high_at: pd.Timestamp
    neckline: float
    neckline_at: pd.Timestamp
    local_low: Pivot | None
    upper_limit: float
    second_pivot_at: pd.Timestamp | None = None
    local_confirmed: bool = False
    neckline_confirmed: bool = False
    signalled: bool = False
    active: bool = True


@dataclass
class Position:
    id: str
    entry_index: int
    entry_at: pd.Timestamp
    entry_price: float
    stop: float
    initial_stop: float
    source_signal: str
    source_episode: str
    source_zone: str
    entry_context: str
    contracts: int


class Engine:
    def __init__(self, daily: pd.DataFrame, objects: pd.DataFrame,
                 contexts: pd.DataFrame, cfg: Config):
        self.d, self.ctx, self.cfg = daily, contexts, cfg
        self.by_date = {t: g.to_dict("records") for t, g in objects.groupby("available_at")}
        self.zones: dict[str, Zone] = {}
        self.pivots: list[Pivot] = []
        self.tops: dict[str, DoubleTop] = {}
        self.used_first_tops: set[str] = set()
        self.position: Position | None = None
        self.pending: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.orders: list[dict[str, Any]] = []
        self.order_counter = 0
        self.trades: list[dict[str, Any]] = []
        self.states: list[dict[str, Any]] = []
        self.i = -1
        self.t = pd.NaT
        self.c: dict[str, Any] = {}
        self.stopped_today = False

    def log(self, event: str, subject: str, **detail: Any) -> None:
        self.events.append(dict(event_id=f"event:{len(self.events) + 1}",
                                available_at=self.t, event_type=event,
                                subject_id=subject, **detail))

    def context(self, direction: str) -> dict[str, Any]:
        wanted = "up" if direction == "bullish" else "down"
        observed = []
        out: dict[str, Any] = {}
        for tf in ("1m", "1w"):
            for name in CONTEXT_FIELDS:
                out[f"{tf}_{name}"] = self.c.get(f"{tf}_{name}", None)
            trend = str(self.c.get(f"{tf}_structure_trend", "unknown"))
            integrity = str(self.c.get(f"{tf}_structure_integrity", "unknown"))
            if trend in {"up", "down"} and integrity != "broken":
                observed.append(trend)
        if not observed:
            category = "context_unknown"
        elif all(x == wanted for x in observed):
            category = "aligned_continuation"
        elif wanted in observed:
            category = "mixed_timeframes"
        else:
            category = "countertrend_turn"
        out["context_class"] = category
        # Bias is recorded independently and never substituted for structural trend.
        return out

    def confirm_pivots(self) -> list[Pivot]:
        new = []
        radii = sorted({self.cfg.local_pivot_radius, self.cfg.major_pivot_radius,
                        self.cfg.trailing_pivot_radius})
        for radius in radii:
            origin = self.i - radius
            if origin < radius:
                continue
            window = self.d.iloc[origin - radius:self.i + 1]
            row = self.d.iloc[origin]
            for kind, col, compare in (("high", "high", max), ("low", "low", min)):
                values = window[col].to_numpy(float)
                value = float(row[col])
                # Strict unique extrema. Equal-high clusters belong to the top model.
                if value != compare(values) or np.count_nonzero(values == value) != 1:
                    continue
                p = Pivot(f"pivot:{radius}:{kind}:{row.datetime.date()}", kind, radius,
                          origin, row.datetime, self.t, value, number(row.atr_prior))
                self.pivots.append(p)
                new.append(p)
                self.log("pivot_confirmed", p.id, pivot=asdict(p))
        return new

    def local_reference(self, kind: str, minimum_index: int | None = None) -> Pivot | None:
        minimum_index = max(0, self.i - self.cfg.local_reference_lookback) if minimum_index is None else minimum_index
        eligible = [p for p in self.pivots if p.kind == kind and
                    p.radius == self.cfg.local_pivot_radius and
                    p.index >= minimum_index and p.available_at < self.t]
        return max(eligible, key=lambda p: (p.index, p.id)) if eligible else None

    def activate(self) -> None:
        for r in self.by_date.get(self.t, []):
            if r["object_type"] == "ob_candidate" and r["evidence_state"] != "confirmed":
                self.log("unconfirmed_ob_not_tradeable", str(r["object_id"]))
                continue
            lo, hi = number(r["zone_low"]), number(r["zone_high"])
            if not math.isfinite(lo + hi) or lo > hi:
                raise ValueError(f"Invalid zone {r['object_id']}")
            z = Zone(str(r["object_id"]), str(r["timeframe"]), str(r["object_type"]),
                     str(r["direction"]), str(r["rule_id"]), pd.Timestamp(r["origin_time"]),
                     pd.Timestamp(r["source_available_at"]), self.t, lo, hi,
                     number(r.get("wick_zone_low"), lo),
                     number(r.get("wick_zone_high"), hi), str(r.get("source_rule_state", "unknown_data")))
            if z.wick_low > z.low or z.wick_high < z.high:
                raise ValueError(f"Wick boundaries do not contain zone {z.id}")
            self.zones[z.id] = z
            self.log("zone_available", z.id, timeframe=z.timeframe, direction=z.direction,
                     origin_at=z.origin_at, source_available_at=z.source_available_at,
                     effective_available_at=z.available_at, low=z.low, high=z.high,
                     rule_id=z.rule_id,
                     source_rule_state=z.source_state)

    def source_close_today(self, z: Zone) -> bool:
        if z.timeframe == "1D" or self.cfg.zone_invalidation_clock == "daily_close":
            return True
        end = self.c.get(f"{z.timeframe.lower()}_context_bar_end")
        return pd.notna(end) and pd.Timestamp(end).normalize() == self.t

    def boundary(self, z: Zone) -> float:
        if self.cfg.zone_invalidation_boundary == "wick":
            return z.wick_low if z.direction == "bullish" else z.wick_high
        return z.low if z.direction == "bullish" else z.high

    def recovery_level(self, z: Zone) -> float:
        if self.cfg.reaction_recovery_boundary == "near":
            return z.high if z.direction == "bullish" else z.low
        return z.low if z.direction == "bullish" else z.high

    def favorable_close(self, z: Zone, close: float) -> bool:
        level = self.recovery_level(z)
        return close > level if z.direction == "bullish" else close < level

    def update_zone(self, z: Zone, bar: pd.Series, atr: float) -> None:
        if z.available_at >= self.t or z.state in {"invalidated", "expired", "filled"}:
            return
        born = int(self.d.datetime.searchsorted(z.available_at))
        if self.i - born > self.cfg.max_zone_age_daily_bars[z.timeframe]:
            z.state = "expired"
            if z.episode:
                z.episode.status = "zone_expired"
            self.log("zone_age_expired", z.id, age_daily_bars=self.i - born)
            return
        hit = bar.low <= z.high and bar.high >= z.low
        prev = float(self.d.close.iloc[self.i - 1]) if self.i else float(bar.open)
        gap = not hit and ((prev > z.high and bar.open < z.low) or
                           (prev < z.low and bar.open > z.high))
        if gap:
            self.log("zone_gap_through_order_unobserved", z.id)
        if z.kind == "fvg":
            filled = hit and (bar.low <= z.low if z.direction == "bullish" else bar.high >= z.high)
            if filled:
                z.state = "filled"
                self.log("fvg_fully_filled", z.id)
            return
        bound = self.boundary(z)
        outside = bar.close < bound if z.direction == "bullish" else bar.close > bound
        if z.state == "reclaim_wait":
            if not outside and self.cfg.allow_reactivated_zones:
                z.state = "recovered"
                z.recovery_count += 1
                self.log("ob_reactivated_after_close_breach", z.id,
                         waiting_bars=self.i - int(z.breach_index), boundary=bound)
                z.breach_index = None
            elif self.i - int(z.breach_index) >= self.cfg.reclaim_grace_bars:
                z.state = "invalidated"
                if z.episode:
                    z.episode.status = "zone_invalidated"
                self.log("ob_reclaim_window_expired", z.id, boundary=bound)
                return
        elif outside and self.source_close_today(z):
            z.breach_index = self.i
            z.breach_count += 1
            z.state = "reclaim_wait" if self.cfg.allow_reactivated_zones else "invalidated"
            self.log("ob_source_close_breach", z.id, state=z.state, boundary=bound,
                     source_timeframe=z.timeframe)
            if z.state == "invalidated":
                if z.episode:
                    z.episode.status = "zone_invalidated"
                return

        if hit and z.armed:
            if z.episode and z.episode.status == "observing":
                z.episode.status = "superseded_by_new_visit"
                self.log("visit_superseded_by_new_visit", z.episode.id)
            reference = self.local_reference("high" if z.direction == "bullish" else "low")
            z.touches += 1
            ep = Episode(f"{z.id}|visit:{z.touches}", z.id, z.touches, self.i, self.t,
                         reference, float(bar.low if z.direction == "bullish" else bar.high),
                         reference_ready_at=reference.available_at if reference else None,
                         context_at_touch=self.context(z.direction))
            z.episode, z.armed, z.away_bars = ep, False, 0
            self.log("ob_visit_started", ep.id, zone_id=z.id, visit_number=z.touches,
                     reference=asdict(reference) if reference else None,
                     context_at_touch=ep.context_at_touch)
        ep = z.episode
        if ep is not None and ep.status in {"observing", "signalled"}:
            ep.extreme = min(ep.extreme, float(bar.low)) if z.direction == "bullish" else max(ep.extreme, float(bar.high))
            wick_breach = bar.low < z.low if z.direction == "bullish" else bar.high > z.high
            if wick_breach and not ep.wick_swept:
                ep.wick_swept = True
                self.log("visit_band_wick_breach", ep.id, boundary=z.low if z.direction == "bullish" else z.high)
            band_close = bar.close < z.low if z.direction == "bullish" else bar.close > z.high
            ep.band_close_breached |= bool(band_close)
            favorable = self.favorable_close(z, float(bar.close))
            if favorable and not ep.recovered:
                ep.recovered, ep.recovery_at = True, self.t
                self.log("visit_recovery_confirmed" if ep.wick_swept else "visit_rejection_candidate",
                         ep.id, recovery_level=self.recovery_level(z),
                         same_bar=ep.start_index == self.i, daily_close_observation=True)
            elif not favorable:
                # Current evidence can fail later; historical recovery events remain.
                ep.recovered = False
            if ep.reference is None:
                ref = self.local_reference("high" if z.direction == "bullish" else "low", ep.start_index)
                if ref:
                    ep.reference, ep.reference_ready_at = ref, self.t
                    self.log("visit_local_reference_set", ep.id, reference=asdict(ref))
            if self.i - ep.start_index > self.cfg.episode_timeout_bars:
                ep.status = "timed_out"
                self.log("visit_confirmation_timeout", ep.id)

        away = (bar.low > z.high + self.cfg.departure_atr * atr) if z.direction == "bullish" else (bar.high < z.low - self.cfg.departure_atr * atr)
        z.away_bars = z.away_bars + 1 if away else 0
        if not z.armed and z.away_bars >= self.cfg.departure_bars:
            # Reaction evaluation occurs before retirement at the end of this bar.
            if ep:
                ep.departed_bars = z.away_bars

    def displacement(self, direction: str, start_index: int, atr: float) -> dict[str, Any]:
        start = max(start_index, self.i - self.cfg.displacement_bars + 1)
        path = self.d.iloc[start:self.i + 1]
        sign = 1 if direction == "bullish" else -1
        movement = sign * (float(path.close.iloc[-1]) - float(path.open.iloc[0]))
        values = np.r_[float(path.open.iloc[0]), path.close.to_numpy(float)]
        travel = float(np.abs(np.diff(values)).sum())
        efficiency = movement / travel if travel > 0 else 0.0
        return dict(passed=movement >= self.cfg.displacement_atr * atr and
                    efficiency >= self.cfg.displacement_efficiency,
                    movement_atr=movement / atr, efficiency=efficiency,
                    from_at=path.datetime.iloc[0], through_at=self.t)

    def linked_fvgs(self, z: Zone, ep: Episode) -> list[str]:
        leg = self.d.iloc[ep.start_index:self.i + 1]
        lo, hi = float(leg.low.min()), float(leg.high.max())
        return sorted(f.id for f in self.zones.values() if f.kind == "fvg" and
                      f.timeframe == "1D" and f.direction == z.direction and
                      ep.start_at <= f.origin_at <= f.available_at <= self.t and
                      f.low <= hi and f.high >= lo)

    def htf_support(self, z: Zone) -> list[str]:
        ids = [x.id for x in self.zones.values() if x.kind == "ob_candidate" and
               x.timeframe in {"1W", "1M"} and x.direction == z.direction and
               x.available_at < self.t and x.state in {"active", "recovered"} and
               x.low <= z.high and x.high >= z.low]
        return sorted(ids)

    def make_signal(self, direction: str, source: str, source_id: str,
                    **details: Any) -> dict[str, Any]:
        s = dict(signal_id=f"signal:{len(self.signals) + 1}", available_at=self.t,
                 direction=direction, source=source, source_id=source_id,
                 source_original_ltf_conditions="unknown_data",
                 definition_scope="daily_adaptation_unvalidated",
                 decision="not_yet_routed", **details)
        self.signals.append(s)
        return s

    def zone_signal(self, z: Zone, bar: pd.Series, atr: float) -> dict[str, Any] | None:
        ep = z.episode
        if z.kind != "ob_candidate" or z.state not in {"active", "recovered"} or ep is None:
            return None
        if ep.status != "observing" or ep.signalled or not ep.recovered:
            return None
        disp = self.displacement(z.direction, ep.start_index, atr)
        prev = float(self.d.close.iloc[self.i - 1])
        ref = ep.reference
        structural = False
        if ref and ref.available_at < self.t:
            structural = (prev <= ref.price < bar.close) if z.direction == "bullish" else (prev >= ref.price > bar.close)
        structural = bool(structural and disp["passed"])
        recovery = bool(ep.recovery_at == self.t and
                        ((bar.close > bar.open) if z.direction == "bullish" else (bar.close < bar.open)))
        if not (structural if self.cfg.confirmation_mode == "structure" else recovery):
            return None
        fvgs = self.linked_fvgs(z, ep)
        context = self.context(z.direction)
        support = self.htf_support(z)
        blockers = []
        if self.cfg.require_reaction_fvg and not fvgs:
            blockers.append("reaction_fvg_not_observed")
        if z.direction == "bullish":
            category = context["context_class"]
            if category not in self.cfg.allowed_entry_contexts:
                blockers.append("entry_context_not_enabled")
            if category != "aligned_continuation":
                if self.cfg.nonaligned_requires_htf_zone and not support:
                    blockers.append("nonaligned_entry_requires_htf_ob")
                if self.cfg.nonaligned_requires_structure and not structural:
                    blockers.append("nonaligned_entry_requires_structure")
        s = self.make_signal(z.direction, "ob_reaction", ep.id, zone_id=z.id,
                             zone_timeframe=z.timeframe, zone_rule_id=z.rule_id,
                             zone_origin_at=z.origin_at, zone_available_at=z.available_at,
                             zone_low=z.low, zone_high=z.high, visit_number=ep.number,
                             visit_at=ep.start_at, confirmation_mode=self.cfg.confirmation_mode,
                             reaction_extreme=ep.extreme, reference=asdict(ref) if ref else None,
                             structure_confirmed=structural, displacement=disp,
                             linked_fvg_ids=fvgs, htf_ob_ids=support, context=context,
                             blockers=blockers, wick_swept=ep.wick_swept,
                             zone_recovery_count=z.recovery_count)
        ep.signalled, ep.status = True, "signalled"
        return s

    def retire_visits(self) -> None:
        for z in self.zones.values():
            ep = z.episode
            if ep and ep.departed_bars >= self.cfg.departure_bars:
                if not z.armed:
                    self.log("visit_departed_rearmed", ep.id, current_status=ep.status)
                # Keep the current reaction observable until its timeout. A new
                # contact supersedes it; merely leaving the band is not failure.
                z.armed = True

    def double_top_signals(self, bar: pd.Series, atr: float,
                           new_pivots: list[Pivot]) -> list[dict[str, Any]]:
        candidates = [p for p in self.pivots if p.kind == "high" and
                      p.radius == self.cfg.major_pivot_radius and p.available_at < self.t and
                      self.cfg.double_top_min_separation <= self.i - p.index <= self.cfg.double_top_max_separation and
                      math.isfinite(p.atr) and p.id not in self.used_first_tops]
        for first in candidates:
            tolerance = self.cfg.double_top_tolerance_atr * first.atr
            prior = self.d.iloc[first.index + 1:self.i]
            if prior.empty:
                continue
            # A sustained close beyond this first top cancels its old resistance role.
            since_known = prior[prior.datetime > first.available_at]
            if len(since_known) and (since_known.close > first.price + tolerance).any():
                continue
            trough_index = int(prior.low.idxmin())
            neck = float(self.d.low.iloc[trough_index])
            if first.price - neck < self.cfg.double_top_min_depth_atr * first.atr:
                continue
            approach = float(self.d.close.iloc[self.i - 1]) <= first.price - self.cfg.double_top_approach_atr * first.atr
            match = (bar.high >= first.price - tolerance and bar.low <= first.price + tolerance and
                     bar.high <= first.price + self.cfg.double_top_max_sweep_atr * first.atr and
                     bar.close <= first.price + tolerance)
            if not (approach and match):
                continue
            top = DoubleTop(f"double_top:{first.id}:{self.t.date()}", first, self.i, self.t,
                            float(bar.high), self.t, neck, self.d.datetime.iloc[trough_index],
                            self.local_reference("low"), first.price + tolerance)
            self.tops[top.id] = top
            self.used_first_tops.add(first.id)
            self.log("double_top_second_peak_candidate", top.id, first=asdict(first),
                     second_high=top.second_high, neckline=neck, neckline_at=top.neckline_at,
                     local_low=asdict(top.local_low) if top.local_low else None,
                     second_peak_not_yet_confirmed=True)
        signals = []
        for top in self.tops.values():
            if not top.active:
                continue
            overshoot = bar.high > top.first.price + self.cfg.double_top_max_sweep_atr * top.first.atr
            price_invalid = bar.close > top.upper_limit or overshoot
            if price_invalid or self.i - top.start_index > self.cfg.double_top_timeout_bars:
                top.active = False
                self.log("double_top_invalidated" if price_invalid else "double_top_expired",
                         top.id, close_above_band=bool(bar.close > top.upper_limit),
                         excessive_high_overshoot=bool(overshoot))
                continue
            if bar.high > top.second_high:
                top.second_high, top.second_high_at = float(bar.high), self.t
                top.second_pivot_at = None
                self.log("double_top_second_peak_updated", top.id,
                         second_high=top.second_high, second_high_at=self.t,
                         second_peak_not_yet_confirmed=True)
            for p in new_pivots:
                if (p.kind == "high" and p.radius == self.cfg.major_pivot_radius and
                        p.origin_at == top.second_high_at and p.price == top.second_high):
                    top.second_pivot_at = self.t
                    self.log("double_top_second_pivot_confirmed", top.id, pivot=asdict(p))
            if top.local_low is None:
                top.local_low = self.local_reference("low", top.start_index)
            prev = float(self.d.close.iloc[self.i - 1])
            disp = self.displacement("bearish", top.start_index, atr)
            ref = top.local_low
            local = bool(ref and ref.available_at < self.t and prev >= ref.price > bar.close and disp["passed"])
            neckline = bool(prev >= top.neckline > bar.close)
            if local and not top.local_confirmed:
                top.local_confirmed = True
                self.log("double_top_local_reversal_confirmed", top.id, local_low=asdict(ref), displacement=disp)
            if neckline and not top.neckline_confirmed:
                top.neckline_confirmed = True
                self.log("double_top_neckline_confirmed", top.id, neckline=top.neckline,
                         second_pivot_available=top.second_pivot_at)
            chosen = local if self.cfg.double_top_exit_mode == "local_structure" else neckline
            if chosen and not top.signalled:
                top.signalled = True
                signals.append(self.make_signal(
                    "bearish", "double_top", top.id, blockers=[],
                    first_top=asdict(top.first), second_high=top.second_high,
                    second_high_at=top.second_high_at, second_pivot_at=top.second_pivot_at,
                    neckline=top.neckline, local_structure_confirmed=top.local_confirmed,
                    neckline_confirmed=top.neckline_confirmed,
                    confirmation_mode=self.cfg.double_top_exit_mode,
                    context=self.context("bearish")))
        return signals

    def slip(self) -> float:
        return self.cfg.tick_size * self.cfg.slippage_ticks

    def round_stop(self, value: float) -> float:
        return math.floor(value / self.cfg.tick_size) * self.cfg.tick_size

    def exit_position(self, raw_price: float, reason: str, order_id: str | None = None,
                      at_open: bool = False) -> None:
        p = self.position
        if p is None:
            return
        price = raw_price - self.slip()
        costs = 2 * self.cfg.commission_per_contract_side_usd * p.contracts
        gross = (price - p.entry_price) * self.cfg.point_value_usd * p.contracts
        self.trades.append(dict(position_id=p.id, entry_at=p.entry_at, entry_price=p.entry_price,
                                exit_at=self.t, exit_price=price, exit_at_open=at_open,
                                exit_reason=reason, source_signal=p.source_signal,
                                source_episode=p.source_episode, source_zone=p.source_zone,
                                entry_context=p.entry_context, contracts=p.contracts,
                                initial_stop=p.initial_stop, stop_at_exit=p.stop,
                                gross_pnl_after_slippage_usd=gross,
                                round_trip_commission_usd=costs, net_pnl_usd=gross - costs,
                                observed_holding_bars=self.i - p.entry_index,
                                sell_order_id=order_id))
        self.log("position_closed", p.id, reason=reason, fill_price=price,
                 at_open=at_open, sell_order_id=order_id)
        self.position = None

    def process_open(self, bar: pd.Series) -> None:
        self.stopped_today = False
        order, self.pending = self.pending, None
        if self.position and bar.open <= self.position.stop:
            if order:
                self.order_status(order, "cancelled_gap_stop_precedes_order")
            self.exit_position(float(bar.open), "hard_stop_gap", at_open=True)
            self.stopped_today = True
            return
        if order is None:
            return
        if order["side"] == "SELL":
            if self.position is None:
                self.order_status(order, "cancelled_no_long_position")
            else:
                self.order_status(order, "filled", fill_at=self.t, fill_price=float(bar.open) - self.slip())
                self.exit_position(float(bar.open), order["reason"], order["order_id"], at_open=True)
        else:
            if self.position is not None:
                self.order_status(order, "cancelled_position_already_open")
            elif bar.open <= order["stop"]:
                self.order_status(order, "cancelled_open_at_or_below_predeclared_stop")
            elif bar.open - order["decision_close"] > self.cfg.max_entry_gap_atr * order["decision_atr"]:
                self.order_status(order, "cancelled_upward_open_gap_too_large")
            else:
                fill = float(bar.open) + self.slip()
                self.order_status(order, "filled", fill_at=self.t, fill_price=fill)
                p = Position(f"position:{order['order_id']}", self.i, self.t, fill,
                             order["stop"], order["stop"], order["signal_id"],
                             order["episode_id"], order["zone_id"],
                             order["context_class"], self.cfg.contracts)
                self.position = p
                self.log("position_opened", p.id, position=asdict(p))

    def intrabar_stop(self, bar: pd.Series) -> None:
        if self.position and bar.low <= self.position.stop:
            self.exit_position(self.position.stop, "hard_stop_intrabar")
            self.stopped_today = True

    def order_status(self, order: dict[str, Any], status: str, **detail: Any) -> None:
        order.update(status=status, status_available_at=self.t, **detail)
        self.orders.append(order.copy())
        self.log("order_status_changed", order["order_id"], order=order.copy())

    def next_open_order(self, side: str, signal: dict[str, Any], bar: pd.Series,
                        atr: float, reason: str) -> None:
        self.order_counter += 1
        order = dict(order_id=f"order:{self.order_counter}", side=side,
                     submitted_at=self.t, execution="next_observed_daily_open",
                     status="pending", status_available_at=self.t,
                     signal_id=signal["signal_id"], reason=reason,
                     decision_close=float(bar.close), decision_atr=atr)
        if side == "BUY":
            z = self.zones[signal["zone_id"]]
            extreme = min(z.wick_low, float(signal["reaction_extreme"]))
            order.update(stop=self.round_stop(extreme - self.cfg.stop_buffer_atr * atr),
                         episode_id=signal["source_id"], zone_id=z.id,
                         context_class=signal["context"]["context_class"])
        # Save append-only versions, so a later fill never rewrites an old row.
        self.orders.append(order.copy())
        self.pending = order
        signal["decision"] = f"{side}_next_open"
        self.log("order_submitted", order["order_id"], order=order.copy())

    def route_signals(self, signals: list[dict[str, Any]], bar: pd.Series, atr: float) -> None:
        eligible = []
        for s in signals:
            if s["blockers"]:
                s["decision"] = "blocked_by_configuration"
            else:
                eligible.append(s)
                s["decision"] = "observed_no_order"
        sells = [s for s in eligible if s["direction"] == "bearish"]
        buys = [s for s in eligible if s["direction"] == "bullish"]
        if self.position:
            for s in buys:
                s["decision"] = "already_long_no_pyramiding"
            if sells:
                # Deterministic routing priority, not a probability or quality score.
                sells.sort(key=lambda s: (s["source"] != "double_top",
                                          -TF_RANK.get(s.get("zone_timeframe", ""), 0), s["source_id"]))
                self.next_open_order("SELL", sells[0], bar, atr, sells[0]["source"])
                for s in sells[1:]:
                    s["decision"] = "same_close_exit_coalesced"
            elif self.cfg.max_holding_bars is not None and self.i - self.position.entry_index >= self.cfg.max_holding_bars:
                signal = self.make_signal("bearish", "holding_timeout", self.position.id, blockers=[])
                self.next_open_order("SELL", signal, bar, atr, "holding_timeout")
            return
        for s in sells:
            s["decision"] = "bearish_watch_no_short_position"
        if sells:
            for s in buys:
                s["decision"] = "conflicting_same_close_signals_wait"
        elif self.stopped_today:
            for s in buys:
                s["decision"] = "same_day_stop_reentry_suppressed"
        elif buys:
            buys.sort(key=lambda s: (-TF_RANK[s["zone_timeframe"]], s["source_id"]))
            self.next_open_order("BUY", buys[0], bar, atr, "ob_reaction")
            for s in buys[1:]:
                s["decision"] = "same_close_entry_coalesced"

    def trail(self, new_pivots: list[Pivot], bar: pd.Series, atr: float) -> None:
        if not self.position or not self.cfg.trailing_stop_enabled or self.pending:
            return
        for p in new_pivots:
            if p.kind != "low" or p.radius != self.cfg.trailing_pivot_radius or p.index < self.position.entry_index:
                continue
            proposal = self.round_stop(p.price - self.cfg.trailing_buffer_atr * atr)
            if self.position.stop < proposal < bar.close:
                old = self.position.stop
                self.position.stop = proposal
                self.log("trailing_stop_revised_at_close", self.position.id, old_stop=old,
                         new_stop=proposal, pivot=asdict(p), effective="next_observed_bar")

    def run(self) -> None:
        for self.i in range(len(self.d)):
            bar = self.d.iloc[self.i]
            self.t = pd.Timestamp(bar.datetime)
            self.c = self.ctx.loc[self.t].to_dict()
            self.process_open(bar)
            self.intrabar_stop(bar)
            atr = number(bar.atr_prior)
            self.activate()
            # Finish all zone states before computing same-close cross-zone context.
            # Even during ATR warmup, update contact/close-breach observations.
            for z in list(self.zones.values()):
                self.update_zone(z, bar, atr)
            new_pivots = self.confirm_pivots()
            signals = []
            if math.isfinite(atr) and atr > 0:
                for z in self.zones.values():
                    s = self.zone_signal(z, bar, atr)
                    if s:
                        signals.append(s)
                signals.extend(self.double_top_signals(bar, atr, new_pivots))
                self.route_signals(signals, bar, atr)
                self.trail(new_pivots, bar, atr)
            self.retire_visits()
            p = self.position
            self.states.append(dict(datetime=self.t, position_id=p.id if p else None,
                                    position="long" if p else "flat", stop=p.stop if p else None,
                                    pending_order_id=self.pending["order_id"] if self.pending else None,
                                    active_ob_count=sum(z.kind == "ob_candidate" and z.state in {"active", "recovered"} for z in self.zones.values()),
                                    observation_context=self.context("bullish")))
        # Terminal snapshots are separate from historical logs, preserving prefixes.
        # Do not force-liquidate, fill a last-bar order or mark an open trade failed.

    def final_snapshot(self) -> dict[str, Any]:
        return dict(as_of=self.t, open_position=asdict(self.position) if self.position else None,
                    pending_order=self.pending, right_censored=True,
                    zones=[dict(zone_id=z.id, state=z.state, touches=z.touches,
                                breach_count=z.breach_count, recovery_count=z.recovery_count,
                                current_episode=asdict(z.episode) if z.episode else None)
                           for z in self.zones.values()],
                    active_double_tops=[asdict(t) for t in self.tops.values() if t.active])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=ROOT / "nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv")
    parser.add_argument("--objects", type=Path, default=ROOT / "nq_ict_detection_20260912_v09/objects_v09.csv")
    parser.add_argument("--contexts", type=Path, default=ROOT / "nq_ict_evaluation_20260912_v09/contexts_v09.csv")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--as-of", help="Inclusive trading-date cutoff, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    cfg = load_config(args.config)
    daily, obj, ctx = load_inputs(args.prices, args.objects, args.contexts, cfg, args.as_of)
    engine = Engine(daily, obj, ctx, cfg)
    engine.run()
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("events.jsonl", engine.events), ("signals.jsonl", engine.signals),
                           ("orders.jsonl", engine.orders), ("trades.jsonl", engine.trades),
                           ("daily_states.jsonl", engine.states)):
        write_jsonl(args.output / filename, rows)
    snapshot = clean(engine.final_snapshot())
    (args.output / "final_state.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    manifest = dict(version=VERSION, result_kind="deterministic_replay_not_strategy_validation",
                    validation_status="not_evaluated", config=asdict(cfg),
                    inputs={name: dict(path=str(path.resolve()), sha256=digest(path)) for name, path in
                            (("prices", args.prices), ("objects", args.objects), ("contexts", args.contexts))},
                    script_sha256=digest(Path(__file__)),
                    data_start=daily.datetime.iloc[0], data_end=daily.datetime.iloc[-1],
                    daily_rows=len(daily), objects_loaded=len(obj),
                    outputs={"events": len(engine.events), "signals": len(engine.signals),
                             "orders": engine.order_counter,
                             "order_state_rows": len(engine.orders),
                             "closed_trades": len(engine.trades)},
                    timing={"decision": "daily close", "fill": "next observed daily open",
                            "atr": "rolling true range mean, lagged one bar",
                            "pivot": "right-side bars required before availability",
                            "weekly_monthly": "source completion dates mapped to first observed daily bar",
                            "upper_timeframe_mapping": "source_available_at/context_bar_end retained; effective date is first daily date on or after source close"},
                    limitations=["No original lower-timeframe confirmation",
                                 "Upstream detector/context correctness remains to be tested",
                                 "Input dates are provider trading-date labels; missing sessions are not imputed",
                                 "Continuous-futures roll adjustment is inherited from input",
                                 "One long position; no shorts, pyramiding or portfolio-margin simulation",
                                 "Point-value cash PnL does not model roll execution or financing",
                                 "No fitted probabilities or performance claims",
                                 "All design thresholds are provisional and have not been validated"])
    (args.output / "manifest.json").write_text(json.dumps(clean(manifest), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Replay artifacts written to {args.output.resolve()}; strategy validation not performed.")


if __name__ == "__main__":
    main()

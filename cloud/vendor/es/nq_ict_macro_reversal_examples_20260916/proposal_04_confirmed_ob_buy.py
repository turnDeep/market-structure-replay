"""提案4: 主要下落中の事前OB反応から、戻り高値突破の確認まで追跡する。

OB時間足だけで採否を決めない。Sweepは連続値として記録し必須にしない。
FVGも記録するが標準設定では必須ではない。入力は旧BUYシグナルではなく
各日の生の足、当時の帯状態、確認済みPivot。
"""
from __future__ import annotations

from dataclasses import dataclass

from .common import (Bar, BuyEvidence, Episode, HigherTimeframe, MarketSnapshot,
                     Pivot, Rules, Zone, ZoneView, check_bar, down_tick, positive)


@dataclass
class Reaction:
    id: str
    episode_id: str
    zone: Zone
    touch_index: int
    low_bar: Bar
    reference: Pivot | None
    recovery_index: int | None = None
    breakout_index: int | None = None
    above_count: int = 0
    revision: int = 0
    emitted: bool = False
    early_emitted: bool = False


class ConfirmedObBuy:
    def __init__(self, rules: Rules):
        if rules.buy_mode not in {"close_sequence", "breakout_retest"}:
            raise ValueError("unsupported buy_mode")
        self.rules = rules
        self.previous: Bar | None = None
        self.active: dict[str, Reaction] = {}
        self.zone_versions: dict[str, Zone] = {}
        self.counter = 0
        self.trace: list[dict] = []
        # The runner reads this after each on_close call.  It is a separate
        # observation stream from the confirmed BUY result so that an early
        # recovery model can share exactly the same reaction population.
        self.early_result: tuple[BuyEvidence, ...] = ()

    def select_lower_high(self, ep: Episode, low: Bar, bar: Bar,
                          pivots: tuple[Pivot, ...]) -> Pivot | None:
        highs = sorted((p for p in pivots if p.kind == "high"
                        and p.radius >= self.rules.buy_reference_radius
                        and p.available_index < bar.index and p.available_at < bar.at
                        and ep.anchor.origin_index < p.origin_index < low.index),
                       key=lambda p: (p.origin_index, p.id))
        eligible: list[Pivot] = []
        for p in highs:
            if not positive(p.atr_at_origin):
                continue
            prior = [q for q in highs if q.origin_index < p.origin_index]
            preceding_price = prior[-1].price if prior else ep.anchor.price
            if (p.price < preceding_price
                    and low.index - p.origin_index >= self.rules.buy_reference_bars
                    and p.price - low.low >= self.rules.buy_reference_drop_atr * p.atr_at_origin):
                eligible.append(p)
        return max(eligible, key=lambda p: (p.origin_index, p.id)) if eligible else None

    @staticmethod
    def clear_confirmation(r: Reaction) -> None:
        r.breakout_index, r.above_count, r.emitted = None, 0, False
        r.early_emitted = False
        r.revision += 1

    def on_close(self, bar: Bar, market: MarketSnapshot, pivots: tuple[Pivot, ...],
                 zones: tuple[ZoneView, ...], htf: HigherTimeframe) -> tuple[BuyEvidence, ...]:
        check_bar(bar, self.previous)
        self.early_result = ()
        if market.at != bar.at or htf.at != bar.at:
            raise ValueError("current snapshots required")
        ep = market.episode
        live = {v.zone.id: v for v in zones if v.observable
                and v.zone.kind == "ob_candidate" and v.zone.direction == "bullish"}
        for zid in list(self.active):
            r = self.active[zid]
            if ep is None or r.episode_id != ep.id or zid not in live:
                self.trace.append(dict(at=bar.at, reaction=r.id, reason="episode_or_source_zone_unavailable"))
                del self.active[zid]
        if ep is None or not positive(bar.atr_prior):
            self.previous = bar
            self.trace.append(dict(at=bar.at, reaction=None, reason="major_decline_or_atr_not_available"))
            return ()

        result: list[BuyEvidence] = []
        early_result: list[BuyEvidence] = []
        for zid, view in sorted(live.items()):
            z = view.zone
            if zid in self.zone_versions and self.zone_versions[zid] != z:
                raise ValueError("historical zone geometry rewritten")
            self.zone_versions[zid] = z
            touch = bar.low <= z.wick_high and bar.high >= z.wick_low
            if zid not in self.active:
                if (ep.phase != "declining" or not touch
                        or z.available_index >= bar.index or z.available_at >= bar.at):
                    continue
                self.counter += 1
                self.active[zid] = Reaction(f"reaction:{self.counter}", ep.id, z,
                                            bar.index, bar,
                                            self.select_lower_high(ep, bar, bar, pivots))
            r = self.active[zid]
            if bar.index - r.touch_index > self.rules.reaction_lifetime:
                del self.active[zid]
                self.trace.append(dict(at=bar.at, reaction=r.id, reason="reaction_expired"))
                continue
            if bar.low < r.low_bar.low:
                r.low_bar = bar
                r.reference = self.select_lower_high(ep, bar, bar, pivots)
                r.recovery_index = None
                self.clear_confirmation(r)
            elif r.reference is None:
                # 遅れて確認された戻り高値は、その確認後からのみ使用する。
                r.reference = self.select_lower_high(ep, r.low_bar, bar, pivots)
            if bar.close < z.body_high:
                if r.recovery_index is not None:
                    self.clear_confirmation(r)
                r.recovery_index = None
            elif r.recovery_index is None and bar.close > bar.open:
                r.recovery_index = bar.index

            # Early model: use the same reaction instance and the same OB
            # recovery as the confirmed model, but do not wait for the lower-
            # high breakout.  A confirmed reference must already be available
            # so the candidate cohort remains point-in-time and comparable to
            # the later confirmed candidate.
            if (r.recovery_index is not None and not r.early_emitted
                    and r.reference is not None
                    and r.reference.available_index < bar.index
                    and r.reference.available_at < bar.at
                    and view.tradeable and ep.phase == "declining"):
                early_linked = tuple(sorted(v.zone.id for v in zones
                    if v.tradeable and v.zone.kind == "fvg" and v.zone.direction == "bullish"
                    and r.recovery_index <= v.zone.origin_index <= v.zone.available_index <= bar.index
                    and v.zone.wick_low <= bar.close and v.zone.wick_high >= r.low_bar.low))
                early_overlap = tuple(sorted(v.zone.id for v in zones
                    if v.tradeable and v.zone.kind == "ob_candidate" and v.zone.direction == "bullish"
                    and v.zone.timeframe in {"1W", "1M"}
                    and v.zone.wick_low <= z.wick_high and v.zone.wick_high >= z.wick_low))
                early_stop = down_tick(min(z.wick_low, r.low_bar.low)
                    - self.rules.emergency_buffer_atr * bar.atr_prior, self.rules.tick_size)
                early_ready = (0 < early_stop < bar.close
                    and (not self.rules.require_linked_fvg or bool(early_linked)))
                if early_ready:
                    setup_id = f"{ep.id}:{r.id}:v{r.revision}"
                    early = BuyEvidence(
                        f"{setup_id}:early:{bar.index}", ep.id, bar.at, bar.index, z,
                        r.touch_index, r.low_bar.low, r.low_bar.index, r.low_bar.atr_prior,
                        r.recovery_index, r.reference, r.recovery_index, "early_recovery",
                        early_stop, early_linked, early_overlap,
                        max(0.0, z.wick_low - r.low_bar.low) / r.low_bar.atr_prior,
                        htf, setup_id=setup_id)
                    early_result.append(early)
                    r.early_emitted = True

            reason = "await_ob_recovery"
            ref = r.reference
            if r.recovery_index is not None:
                reason = "await_preconfirmed_lower_high"
                if ref is not None:
                    boundary = ref.price + self.rules.structure_buffer_atr * ref.atr_at_origin
                    above = bar.close > boundary
                    if not above:
                        if r.breakout_index is not None:
                            self.clear_confirmation(r)
                    elif r.breakout_index is not None:
                        r.above_count += 1
                    elif (self.previous is not None and self.previous.close <= boundary
                          and ref.available_index < bar.index and ref.available_at < bar.at):
                        r.breakout_index, r.above_count = bar.index, 1
                    reason = "await_lower_high_break_confirmation"
                    departure = bar.close - r.low_bar.low >= self.rules.buy_departure_atr * r.low_bar.atr_prior
                    if self.rules.buy_mode == "close_sequence":
                        confirmed = r.above_count >= self.rules.buy_confirm_closes
                    else:
                        confirmed = (r.breakout_index is not None and bar.index > r.breakout_index
                                     and above and bar.close > bar.open
                                     and bar.low <= ref.price + self.rules.retest_tolerance_atr * ref.atr_at_origin
                                     and bar.low > r.low_bar.low)
                    linked = tuple(sorted(v.zone.id for v in zones
                        if v.tradeable and v.zone.kind == "fvg" and v.zone.direction == "bullish"
                        and r.recovery_index <= v.zone.origin_index <= v.zone.available_index <= bar.index
                        and v.zone.wick_low <= bar.close and v.zone.wick_high >= r.low_bar.low))
                    overlap = tuple(sorted(v.zone.id for v in zones
                        if v.tradeable and v.zone.kind == "ob_candidate" and v.zone.direction == "bullish"
                        and v.zone.timeframe in {"1W", "1M"}
                        and v.zone.wick_low <= z.wick_high and v.zone.wick_high >= z.wick_low))
                    stop = down_tick(min(z.wick_low, r.low_bar.low)
                                     - self.rules.emergency_buffer_atr * bar.atr_prior,
                                     self.rules.tick_size)
                    ready = (confirmed and departure and view.tradeable and not r.emitted
                             and ep.phase == "declining" and 0 < stop < bar.close
                             and (not self.rules.require_linked_fvg or bool(linked)))
                    if ready:
                        key = f"{ep.id}:{r.id}:v{r.revision}:{bar.index}"
                        result.append(BuyEvidence(key, ep.id, bar.at, bar.index, z, r.touch_index,
                            r.low_bar.low, r.low_bar.index, r.low_bar.atr_prior,
                            r.recovery_index, ref, r.breakout_index, self.rules.buy_mode,
                            stop, linked, overlap,
                            max(0.0, z.wick_low - r.low_bar.low) / r.low_bar.atr_prior, htf,
                            setup_id=f"{ep.id}:{r.id}:v{r.revision}"))
                        r.emitted = True
                        reason = "confirmed_ob_structure_buy"
                    elif confirmed and not view.tradeable:
                        reason = "source_zone_not_tradeable"
                    elif confirmed and ep.phase != "declining":
                        reason = "episode_already_advancing_no_new_main_entry"
                    elif confirmed and self.rules.require_linked_fvg and not linked:
                        reason = "linked_fvg_missing"
            self.trace.append(dict(at=bar.at, episode=ep.id, reaction=r.id, reason=reason,
                source_zone=z.id, source_timeframe=z.timeframe, source_state=view.state,
                low=r.low_bar.low, low_index=r.low_bar.index, recovery_index=r.recovery_index,
                reference=ref.id if ref else None,
                reference_available_at=ref.available_at if ref else None,
                breakout_index=r.breakout_index, above_count=r.above_count))
        self.early_result = tuple(early_result)
        self.previous = bar
        return tuple(result)

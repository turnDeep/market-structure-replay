"""Revision replay for early OB reactions and early double-top reversals.

This file keeps the original causal engine as a dependency and changes only
the explicitly documented revision points.  It is a separate experiment; the
baseline replay artifacts are not overwritten.  No LuxAlgo detector is used.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nq_ict_retest_trade_state_machine_20260915 import (
    CONTEXT_FIELDS,
    TF_RANK,
    Config as BaseConfig,
    DoubleTop,
    Engine as BaseEngine,
    Episode,
    Pivot,
    Zone,
    clean,
    digest,
    load_inputs,
    number,
    write_jsonl,
)


VERSION = "nq-ict-retest-trade-0.2-revision-unvalidated"
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RevisionConfig(BaseConfig):
    """Fixed revision rules; values are not fitted to the requested dates."""

    reaction_contact_boundary: str = "wick"
    append_only_recovery_events: bool = True
    double_top_reversal_body_atr: float = 0.10
    double_top_reversal_close_atr: float = 0.25
    double_top_exit_mode: str = "reversal"
    nonaligned_requires_htf_zone: bool = False
    nonaligned_requires_structure: bool = False

    def check(self) -> None:
        # BaseConfig validates every inherited field.  It does not know the
        # additional revision exit mode, so validate it through a compatible
        # temporary mode and then restore the declared mode.
        mode = self.double_top_exit_mode
        try:
            object.__setattr__(self, "double_top_exit_mode", "local_structure")
            super().check()
        finally:
            object.__setattr__(self, "double_top_exit_mode", mode)
        if mode not in {"local_structure", "neckline", "reversal"}:
            raise ValueError(f"Invalid double_top_exit_mode: {mode}")
        if self.reaction_contact_boundary not in {"body", "wick"}:
            raise ValueError("reaction_contact_boundary must be body or wick")
        if not 0 < self.double_top_reversal_body_atr:
            raise ValueError("double_top_reversal_body_atr must be positive")
        if not 0 < self.double_top_reversal_close_atr:
            raise ValueError("double_top_reversal_close_atr must be positive")


def load_revision_config(path: Path | None) -> RevisionConfig:
    raw = json.loads(path.read_text(encoding="utf-8-sig")) if path else {}
    allowed = {f.name for f in fields(RevisionConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("zone_timeframes", "ob_rule_ids", "allowed_entry_contexts"):
        if key in raw:
            raw[key] = tuple(raw[key])
    cfg = RevisionConfig(**raw)
    cfg.check()
    return cfg


class RevisionEngine(BaseEngine):
    """Base engine with explicit contact/recovery/reversal state changes."""

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

        body_hit = bar.low <= z.high and bar.high >= z.low
        wick_hit = bar.low <= z.wick_high and bar.high >= z.wick_low
        contact_low = z.wick_low if self.cfg.reaction_contact_boundary == "wick" else z.low
        contact_high = z.wick_high if self.cfg.reaction_contact_boundary == "wick" else z.high
        hit = bar.low <= contact_high and bar.high >= contact_low
        prev = float(self.d.close.iloc[self.i - 1]) if self.i else float(bar.open)
        gap = not hit and ((prev > contact_high and bar.open < contact_low) or
                           (prev < contact_low and bar.open > contact_high))
        if gap:
            self.log("zone_gap_through_order_unobserved", z.id,
                     contact_boundary=self.cfg.reaction_contact_boundary)

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
            # Dynamic fields are deliberately also written into events.  The
            # imported dataclass remains backward compatible with baseline QA.
            ep.body_contacted = bool(body_hit)
            ep.wick_contacted = bool(wick_hit)
            ep.first_recovery_at = None
            ep.last_recovery_at = None
            ep.recovery_cycles = 0
            ep.recovery_lost_at = None
            z.episode, z.armed, z.away_bars = ep, False, 0
            self.log("ob_visit_started", ep.id, zone_id=z.id, visit_number=z.touches,
                     reference=asdict(reference) if reference else None,
                     context_at_touch=ep.context_at_touch,
                     contact_boundary=self.cfg.reaction_contact_boundary,
                     body_contacted=bool(body_hit), wick_contacted=bool(wick_hit),
                     body_low=z.low, body_high=z.high,
                     wick_low=z.wick_low, wick_high=z.wick_high)

        ep = z.episode
        if ep is not None and ep.status in {"observing", "signalled"}:
            ep.extreme = min(ep.extreme, float(bar.low)) if z.direction == "bullish" else max(ep.extreme, float(bar.high))
            wick_breach = bar.low < z.low if z.direction == "bullish" else bar.high > z.high
            if wick_breach and not ep.wick_swept:
                ep.wick_swept = True
                self.log("visit_band_wick_breach", ep.id,
                         boundary=z.low if z.direction == "bullish" else z.high)
            band_close = bar.close < z.low if z.direction == "bullish" else bar.close > z.high
            ep.band_close_breached |= bool(band_close)
            favorable = self.favorable_close(z, float(bar.close))
            if favorable and not ep.recovered:
                ep.recovered = True
                ep.recovery_cycles = int(getattr(ep, "recovery_cycles", 0)) + 1
                if getattr(ep, "first_recovery_at", None) is None:
                    ep.first_recovery_at = self.t
                    ep.recovery_at = self.t
                    event_name = "visit_recovery_confirmed" if ep.wick_swept else "visit_rejection_candidate"
                else:
                    ep.last_recovery_at = self.t
                    event_name = "visit_recovery_reconfirmed" if self.cfg.append_only_recovery_events else "visit_recovery_confirmed"
                ep.last_recovery_at = self.t
                self.log(event_name, ep.id, recovery_level=self.recovery_level(z),
                         same_bar=ep.start_index == self.i, daily_close_observation=True,
                         recovery_cycle=ep.recovery_cycles,
                         first_recovery_at=getattr(ep, "first_recovery_at", None),
                         body_contacted=getattr(ep, "body_contacted", False),
                         wick_contacted=getattr(ep, "wick_contacted", False))
            elif not favorable and ep.recovered:
                ep.recovered = False
                ep.recovery_lost_at = self.t
                if self.cfg.append_only_recovery_events:
                    self.log("visit_recovery_lost", ep.id,
                             recovery_cycle=getattr(ep, "recovery_cycles", 0),
                             recovery_level=self.recovery_level(z))
            if ep.reference is None:
                ref = self.local_reference("high" if z.direction == "bullish" else "low", ep.start_index)
                if ref:
                    ep.reference, ep.reference_ready_at = ref, self.t
                    self.log("visit_local_reference_set", ep.id, reference=asdict(ref))
            if self.i - ep.start_index > self.cfg.episode_timeout_bars:
                ep.status = "timed_out"
                self.log("visit_confirmation_timeout", ep.id)

        away = ((bar.low > contact_high + self.cfg.departure_atr * atr)
                if z.direction == "bullish"
                else (bar.high < contact_low - self.cfg.departure_atr * atr))
        z.away_bars = z.away_bars + 1 if away else 0
        if not z.armed and z.away_bars >= self.cfg.departure_bars and ep:
            ep.departed_bars = z.away_bars

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
            structural = ((prev <= ref.price < bar.close) if z.direction == "bullish"
                          else (prev >= ref.price > bar.close))
        structural = bool(structural and disp["passed"])
        latest_recovery = getattr(ep, "last_recovery_at", None) or ep.recovery_at
        recovery = bool(latest_recovery == self.t and
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
        s = self.make_signal(z.direction, "ob_reaction", ep.id,
                             zone_id=z.id, zone_timeframe=z.timeframe,
                             zone_rule_id=z.rule_id, zone_origin_at=z.origin_at,
                             zone_available_at=z.available_at, zone_low=z.low,
                             zone_high=z.high, zone_wick_low=z.wick_low,
                             zone_wick_high=z.wick_high, visit_number=ep.number,
                             visit_at=ep.start_at, confirmation_mode=self.cfg.confirmation_mode,
                             reaction_extreme=ep.extreme,
                             reference=asdict(ref) if ref else None,
                             structure_confirmed=structural, displacement=disp,
                             linked_fvg_ids=fvgs, htf_ob_ids=support,
                             context=context, blockers=blockers,
                             wick_swept=ep.wick_swept,
                             contact_boundary=self.cfg.reaction_contact_boundary,
                             body_contacted=getattr(ep, "body_contacted", False),
                             wick_contacted=getattr(ep, "wick_contacted", False),
                             zone_recovery_count=z.recovery_count)
        ep.signalled, ep.status = True, "signalled"
        return s

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
            reversal = bool(
                self.i > top.start_index and
                (float(bar.open) - float(bar.close)) >= self.cfg.double_top_reversal_body_atr * top.first.atr and
                float(bar.close) < prev and
                float(bar.close) <= top.second_high - self.cfg.double_top_reversal_close_atr * top.first.atr
            )
            if local and not top.local_confirmed:
                top.local_confirmed = True
                self.log("double_top_local_reversal_confirmed", top.id, local_low=asdict(ref), displacement=disp)
            if neckline and not top.neckline_confirmed:
                top.neckline_confirmed = True
                self.log("double_top_neckline_confirmed", top.id, neckline=top.neckline,
                         second_pivot_available=top.second_pivot_at)
            if reversal:
                self.log("double_top_early_reversal_confirmed", top.id,
                         second_high=top.second_high, reversal_at=self.t,
                         reversal_body_atr=(float(bar.open) - float(bar.close)) / top.first.atr,
                         close_distance_atr=(top.second_high - float(bar.close)) / top.first.atr)
            chosen = {"local_structure": local, "neckline": neckline,
                      "reversal": reversal}[self.cfg.double_top_exit_mode]
            if chosen and not top.signalled:
                top.signalled = True
                signals.append(self.make_signal(
                    "bearish", "double_top", top.id, blockers=[],
                    first_top=asdict(top.first), second_high=top.second_high,
                    second_high_at=top.second_high_at, second_pivot_at=top.second_pivot_at,
                    neckline=top.neckline, local_structure_confirmed=top.local_confirmed,
                    neckline_confirmed=top.neckline_confirmed,
                    early_reversal_confirmed=reversal,
                    confirmation_mode=self.cfg.double_top_exit_mode,
                    context=self.context("bearish")))
        return signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=ROOT / "nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv")
    parser.add_argument("--objects", type=Path, default=ROOT / "nq_ict_detection_20260912_v09/objects_v09.csv")
    parser.add_argument("--contexts", type=Path, default=ROOT / "nq_ict_evaluation_20260912_v09/contexts_v09.csv")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--as-of")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    cfg = load_revision_config(args.config)
    daily, obj, ctx = load_inputs(args.prices, args.objects, args.contexts, cfg, args.as_of)
    engine = RevisionEngine(daily, obj, ctx, cfg)
    engine.run()
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("events.jsonl", engine.events), ("signals.jsonl", engine.signals),
                           ("orders.jsonl", engine.orders), ("trades.jsonl", engine.trades),
                           ("daily_states.jsonl", engine.states)):
        write_jsonl(args.output / filename, rows)
    (args.output / "final_state.json").write_text(
        json.dumps(clean(engine.final_snapshot()), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = dict(
        version=VERSION,
        result_kind="deterministic_revision_replay_not_strategy_validation",
        validation_status="not_evaluated",
        revision_rules={
            "contact_boundary": cfg.reaction_contact_boundary,
            "append_only_recovery_events": cfg.append_only_recovery_events,
            "confirmation_mode": cfg.confirmation_mode,
            "nonaligned_requires_htf_zone": cfg.nonaligned_requires_htf_zone,
            "nonaligned_requires_structure": cfg.nonaligned_requires_structure,
            "double_top_exit_mode": cfg.double_top_exit_mode,
            "double_top_reversal_body_atr": cfg.double_top_reversal_body_atr,
            "double_top_reversal_close_atr": cfg.double_top_reversal_close_atr,
        },
        config=asdict(cfg),
        inputs={name: dict(path=str(path.resolve()), sha256=digest(path)) for name, path in
                (("prices", args.prices), ("objects", args.objects), ("contexts", args.contexts))},
        script_sha256=digest(Path(__file__)),
        data_start=daily.datetime.iloc[0], data_end=daily.datetime.iloc[-1],
        daily_rows=len(daily), objects_loaded=len(obj),
        outputs={"events": len(engine.events), "signals": len(engine.signals),
                 "orders": engine.order_counter, "order_state_rows": len(engine.orders),
                 "closed_trades": len(engine.trades)},
        timing={"decision": "daily close", "fill": "next observed daily open",
                "pivot": "right-side bars required before availability",
                "weekly_monthly": "source completion dates mapped to first observed daily bar",
                "lower_timeframe_conditions": "unknown_data"},
        limitations=["No original lower-timeframe confirmation",
                     "This is an unvalidated revision, not a performance claim",
                     "Continuous-futures roll adjustment is inherited from input",
                     "One long position; no shorts, pyramiding or portfolio-margin simulation",
                     "Thresholds are fixed engineering choices and were not fitted to target dates"],
    )
    (args.output / "manifest.json").write_text(
        json.dumps(clean(manifest), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Revision replay artifacts written to {args.output.resolve()}; strategy validation not performed.")


if __name__ == "__main__":
    main()

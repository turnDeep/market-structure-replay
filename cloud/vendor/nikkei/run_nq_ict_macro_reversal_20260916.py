"""Run the macro-reversal examples through the frozen NKD=F daily detector.

This is a separate experiment from the prior revision replay.  It uses the
existing causal zone/pivot/context loader only as an input adapter and routes
BUY/SELL decisions through ``nq_ict_macro_reversal_examples_20260916``.

Clock:
    prior close -> next observed daily open execution -> current daily close

The program does not use user dates as trading conditions, outcome labels,
LuxAlgo, the prior route_signals/trail path, or complete-sample zone states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nq_ict_retest_trade_state_machine_20260915 import load_inputs, number
from nq_ict_retest_trade_state_machine_revision_20260915 import RevisionEngine, load_revision_config
from nq_ict_macro_reversal_examples_20260916.common import (
    Bar, HigherTimeframe, Pivot, Position, Rules, Zone, ZoneView, down_tick,
)
from nq_ict_macro_reversal_examples_20260916.integration_example import (
    CloseInputs, MacroReversalPolicy,
)


ROOT = Path(__file__).resolve().parent
VERSION = "nq-ict-macro-reversal-0.1-executed"
TERMINAL_ZONE_STATES = {"invalidated", "expired", "filled"}


def clean(value: Any) -> Any:
    if is_dataclass(value):
        return clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else None
    return value


def write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def timestamp(value: Any) -> pd.Timestamp | None:
    x = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(x) else pd.Timestamp(x).normalize()


def text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    value = str(value)
    return default if value.lower() in {"nan", "nat", "none"} else value


class CausalPivotAdapter:
    """Convert the existing engine pivots into the new package's clocked type."""

    def __init__(self, daily: pd.DataFrame):
        self.daily = daily
        self.seen: dict[str, Pivot] = {}

    def convert(self, pivots: list[Any]) -> tuple[Pivot, ...]:
        for raw in pivots:
            if raw.id in self.seen:
                continue
            available_index = int(self.daily.datetime.searchsorted(raw.available_at))
            self.seen[raw.id] = Pivot(
                id=str(raw.id), kind=str(raw.kind), origin_index=int(raw.index),
                origin_at=pd.Timestamp(raw.origin_at), available_index=available_index,
                available_at=pd.Timestamp(raw.available_at), price=float(raw.price),
                atr_at_origin=number(raw.atr), radius=int(raw.radius),
            )
        return tuple(sorted(self.seen.values(), key=lambda p: (
            p.available_index, p.origin_index, p.radius, p.kind, p.id)))


class MacroReplayEngine(RevisionEngine):
    def __init__(self, daily: pd.DataFrame, objects: pd.DataFrame,
                 contexts: pd.DataFrame, cfg: Any, rules: Rules):
        super().__init__(daily, objects, contexts, cfg)
        self.rules = rules
        self.policy = MacroReversalPolicy(rules)
        self.pivot_adapter = CausalPivotAdapter(daily)
        self.known_pivots: tuple[Pivot, ...] = ()
        self.previous_bar: Bar | None = None
        self.pending_structural_stop: tuple[str, float] | None = None
        self.live_stop: float | None = None
        self.execution_rows: list[dict[str, Any]] = []
        self.plan_rows: list[dict[str, Any]] = []
        self.state_rows: list[dict[str, Any]] = []
        self.zone_rows: list[dict[str, Any]] = []
        self.pivot_rows: list[dict[str, Any]] = []
        self.buy_rows: list[dict[str, Any]] = []
        self.sell_rows: list[dict[str, Any]] = []

    def _bar(self, row: pd.Series) -> Bar:
        return Bar(int(self.i), self.t, float(row.open), float(row.high),
                   float(row.low), float(row.close), number(row.atr_prior))

    def _zone_views(self) -> tuple[ZoneView, ...]:
        rows: list[ZoneView] = []
        for old in sorted(self.zones.values(), key=lambda z: z.id):
            if old.available_at > self.t:
                continue
            available_index = int(self.d.datetime.searchsorted(old.available_at))
            origin_index = int(self.d.datetime.searchsorted(old.origin_at, side="left"))
            state = str(old.state)
            zone = Zone(
                id=old.id, kind=old.kind, direction=old.direction,
                timeframe=old.timeframe, origin_index=origin_index,
                origin_at=pd.Timestamp(old.origin_at), available_index=available_index,
                available_at=pd.Timestamp(old.available_at), body_low=float(old.low),
                body_high=float(old.high), wick_low=float(old.wick_low),
                wick_high=float(old.wick_high),
            )
            rows.append(ZoneView(zone, self.t, state))
        return tuple(rows)

    def _higher_timeframe(self) -> HigherTimeframe:
        return HigherTimeframe(
            at=self.t,
            weekly_completed_at=timestamp(self.c.get("1w_source_context_bar_end")),
            monthly_completed_at=timestamp(self.c.get("1m_source_context_bar_end")),
            weekly_trend=text(self.c.get("1w_structure_trend")),
            monthly_trend=text(self.c.get("1m_structure_trend")),
            weekly_bias=text(self.c.get("1w_bias_direction")),
            monthly_bias=text(self.c.get("1m_bias_direction")),
        )

    def _roundtrip_cost(self, quantity: int) -> float:
        return 2.0 * float(self.cfg.commission_per_contract_side_usd) * quantity

    def _apply_open(self, bar: Bar) -> None:
        """Execute prior intent and declared stops before today's close."""
        if self.pending_structural_stop is not None:
            position_id, stop = self.pending_structural_stop
            if self.policy.position is not None and self.policy.position.id == position_id:
                self.live_stop = max(self.live_stop or -math.inf, stop)
                self.execution_rows.append(dict(at=bar.at, stage="structural_stop_effective",
                                                position_id=position_id, stop=self.live_stop))
            self.pending_structural_stop = None

        slip = float(self.cfg.tick_size) * float(self.cfg.slippage_ticks)
        if self.policy.position is not None and self.live_stop is not None and bar.open <= self.live_stop:
            self.policy.on_emergency_stop_fill(
                bar.at, bar.index, bar.open - slip,
                self._roundtrip_cost(self.policy.position.quantity),
            )
            self.execution_rows.append(dict(at=bar.at, stage="gap_emergency_stop", price=bar.open - slip))
            self.live_stop = None
            return

        intent = self.policy.pending
        if intent is not None:
            if intent.side == "BUY":
                candidate = intent.candidate
                if candidate is None:
                    raise ValueError("BUY intent has no candidate")
                fill = bar.open + slip
                if fill <= candidate.emergency_stop:
                    self.policy.cancel_pending(bar.at, bar.index, "open_at_or_below_initial_stop")
                    self.execution_rows.append(dict(at=bar.at, stage="buy_cancel", reason="initial_stop"))
                else:
                    self.policy.on_next_open_fill(
                        intent.id, bar.at, bar.index, fill, self.cfg.contracts,
                        self.cfg.point_value_usd, self._roundtrip_cost(self.cfg.contracts),
                    )
                    self.live_stop = candidate.emergency_stop
                    self.execution_rows.append(dict(at=bar.at, stage="buy_fill", price=fill,
                                                    order_id=intent.id, stop=self.live_stop))
            elif intent.side == "SELL":
                if self.policy.position is None:
                    self.policy.cancel_pending(bar.at, bar.index, "no_position_at_open")
                    self.execution_rows.append(dict(at=bar.at, stage="sell_cancel", reason="no_position"))
                else:
                    fill = bar.open - slip
                    self.policy.on_next_open_fill(
                        intent.id, bar.at, bar.index, fill, self.cfg.contracts,
                        self.cfg.point_value_usd, self._roundtrip_cost(self.cfg.contracts),
                    )
                    self.live_stop = None
                    self.execution_rows.append(dict(at=bar.at, stage="sell_fill", price=fill,
                                                    order_id=intent.id))
            else:
                raise ValueError(f"unknown order side {intent.side}")

        if self.policy.position is not None and self.live_stop is not None and bar.low <= self.live_stop:
            self.policy.on_emergency_stop_fill(
                bar.at, bar.index, self.live_stop - slip,
                self._roundtrip_cost(self.policy.position.quantity),
            )
            self.execution_rows.append(dict(at=bar.at, stage="intrabar_emergency_stop",
                                            price=self.live_stop - slip))
            self.live_stop = None

    def _record_state(self, bar: Bar, views: tuple[ZoneView, ...], htf: HigherTimeframe) -> None:
        p = self.policy.position
        pending = self.policy.pending
        ep = self.policy.market.episode
        self.state_rows.append(dict(
            datetime=bar.at, position_id=p.id if p else None,
            position="long" if p else "flat", live_stop=self.live_stop,
            pending_order_id=pending.id if pending else None,
            pending_side=pending.side if pending else None,
            active_ob_count=sum(v.zone.kind == "ob_candidate" and v.observable for v in views),
            bullish_ob_count=sum(v.zone.kind == "ob_candidate" and v.zone.direction == "bullish" and v.observable for v in views),
            bearish_ob_count=sum(v.zone.kind == "ob_candidate" and v.zone.direction == "bearish" and v.observable for v in views),
            fvg_count=sum(v.zone.kind == "fvg" and v.observable for v in views),
            episode_id=ep.id if ep else None, episode_phase=ep.phase if ep else None,
            episode_low=ep.low if ep else None,
            weekly_trend=htf.weekly_trend, monthly_trend=htf.monthly_trend,
            weekly_bias=htf.weekly_bias, monthly_bias=htf.monthly_bias,
        ))

    def run(self) -> None:
        for self.i in range(len(self.d)):
            row = self.d.iloc[self.i]
            self.t = pd.Timestamp(row.datetime)
            self.c = self.ctx.loc[self.t].to_dict()
            bar = self._bar(row)
            self._apply_open(bar)

            self.activate()
            for zone in list(self.zones.values()):
                self.update_zone(zone, row, number(row.atr_prior))
            self.confirm_pivots()
            self.known_pivots = self.pivot_adapter.convert(self.pivots)
            views = self._zone_views()
            htf = self._higher_timeframe()
            for pivot in self.known_pivots:
                if pivot.available_index == bar.index:
                    self.pivot_rows.append(pivot)
            for view in views:
                if view.zone.available_index == bar.index:
                    self.zone_rows.append(view)

            plan = self.policy.on_close(CloseInputs(bar, self.known_pivots, views, htf))

            for candidate in plan.buys:
                self.buy_rows.append(candidate)
            for sell in plan.sells:
                self.sell_rows.append(sell)

            if plan.management.proposed_protected_low is not None and self.policy.position is not None:
                low = plan.management.proposed_protected_low.pivot
                stop = down_tick(low.price - self.rules.emergency_buffer_atr * bar.atr_prior,
                                 self.rules.tick_size)
                self.pending_structural_stop = (self.policy.position.id, stop)

            self.plan_rows.append(dict(
                datetime=bar.at, reason=plan.reason,
                intent=plan.intent, buy_count=len(plan.buys), sell_count=len(plan.sells),
                buy_evidence=plan.buys, sell_evidence=plan.sells,
                management=plan.management, gate_reasons=plan.gate_reasons,
                episode_phase=plan.resulting_episode_phase,
            ))
            self._record_state(bar, views, htf)
            self.previous_bar = bar

    def final_state(self) -> dict[str, Any]:
        return dict(
            as_of=self.t, open_position=self.policy.position,
            pending_order=self.policy.pending, live_stop=self.live_stop,
            right_censored=True, episode=self.policy.market.episode,
            episode_budgets=self.policy.gate.budgets,
            active_high_areas=self.policy.market.history[-1].areas if self.policy.market.history else (),
        )


def make_rules() -> Rules:
    return Rules(
        pivot_radius=5,
        buy_reference_radius=2,
        sell_support_radius=2,
        area_tolerance_atr=0.50,
        area_max_age=504,
        major_decline_atr=3.0,
        major_decline_bars=15,
        breakout_accept_closes=3,
        management_pullback_bars=20,
        management_pullback_atr=2.5,
        structural_exit_closes=2,
        top_min_separation=20,
        top_lifetime=60,
        sell_support_bars=5,
        sell_support_rise_atr=2.0,
        sell_departure_atr=1.0,
        sell_confirm_closes=2,
        sell_mode="close_sequence",
        reaction_lifetime=60,
        buy_reference_drop_atr=1.5,
        buy_reference_bars=3,
        buy_departure_atr=1.0,
        buy_confirm_closes=2,
        buy_mode="close_sequence",
        structure_buffer_atr=0.05,
        retest_tolerance_atr=0.10,
        require_linked_fvg=False,
        emergency_buffer_atr=0.10,
        tick_size=0.25,
        max_attempts_per_episode=2,
        max_loss_r_per_episode=2.0,
        retry_new_low_atr=0.50,
        retry_higher_low_atr=0.50,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=ROOT / "nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv")
    parser.add_argument("--objects", type=Path, default=ROOT / "nq_ict_detection_20260912_v09/objects_v09.csv")
    parser.add_argument("--contexts", type=Path, default=ROOT / "nq_ict_evaluation_20260912_v09/contexts_v09.csv")
    parser.add_argument("--base-config", type=Path, default=ROOT / "nq_ict_retest_trade_revision_config_20260915.json")
    parser.add_argument("--as-of")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")

    cfg = load_revision_config(args.base_config)
    daily, objects, contexts = load_inputs(args.prices, args.objects, args.contexts, cfg, args.as_of)
    rules = make_rules()
    engine = MacroReplayEngine(daily, objects, contexts, cfg, rules)
    engine.run()

    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "close_plans.jsonl", engine.plan_rows)
    write_jsonl(args.output / "daily_states.jsonl", engine.state_rows)
    write_jsonl(args.output / "execution_events.jsonl", engine.execution_rows)
    write_jsonl(args.output / "zone_available_rows.jsonl", engine.zone_rows)
    write_jsonl(args.output / "pivot_available_rows.jsonl", engine.pivot_rows)
    write_jsonl(args.output / "buy_evidence.jsonl", engine.buy_rows)
    write_jsonl(args.output / "sell_evidence.jsonl", engine.sell_rows)
    write_jsonl(args.output / "market_events.jsonl", engine.policy.market.events)
    write_jsonl(args.output / "buyer_trace.jsonl", engine.policy.buyer.trace)
    write_jsonl(args.output / "seller_trace.jsonl", engine.policy.seller.trace)
    write_jsonl(args.output / "manager_events.jsonl", engine.policy.manager.events)
    write_jsonl(args.output / "gate_events.jsonl", engine.policy.gate.events)
    write_jsonl(args.output / "order_events.jsonl", engine.policy.execution_log)
    write_jsonl(args.output / "trades.jsonl", engine.policy.trades)
    (args.output / "final_state.json").write_text(
        json.dumps(clean(engine.final_state()), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = dict(
        version=VERSION,
        result_kind="executed_causal_macro_reversal_replay_not_final_strategy_validation",
        validation_status="pending_external_validator",
        input_scope=dict(data_start=daily.datetime.iloc[0], data_end=daily.datetime.iloc[-1],
                         daily_rows=len(daily), objects_loaded=len(objects)),
        inputs={name: dict(path=str(path.resolve()), sha256=digest(path)) for name, path in (
            ("prices", args.prices), ("objects", args.objects), ("contexts", args.contexts),
            ("base_config", args.base_config),
        )},
        script_sha256=digest(Path(__file__)),
        rules=clean(asdict(rules)),
        outputs=dict(
            close_plans=len(engine.plan_rows), daily_states=len(engine.state_rows),
            execution_events=len(engine.execution_rows), zones_available=len(engine.zone_rows),
            pivots_available=len(engine.pivot_rows), buy_evidence=len(engine.buy_rows),
            sell_evidence=len(engine.sell_rows), market_events=len(engine.policy.market.events),
            trades=len(engine.policy.trades),
        ),
        timing=dict(decision="daily close", fill="next observed daily open",
                    emergency_stop="gap/open and current daily low after declared stop",
                    structural_stop="next observed open after proposal",
                    pivot="right-side bars required before available_index",
                    weekly_monthly="only completed source bars supplied by causal loader",
                    lower_timeframe_conditions="unknown_data"),
        restrictions=[
            "Existing detector and context inputs are reconstructed point-in-time by the loader",
            "No user target date is a trading condition",
            "No old route_signals, zone_signal, trail or LuxAlgo path is used",
            "One long position; no short position and no pyramiding",
            "Daily OHLC cannot resolve same-bar high/low order",
            "Thresholds are fixed engineering choices and were not fitted in this run",
            "The output is a causal replay and requires independent validation for performance claims",
        ],
    )
    (args.output / "manifest.json").write_text(
        json.dumps(clean(manifest), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Executed macro reversal replay: {args.output.resolve()}")


if __name__ == "__main__":
    main()

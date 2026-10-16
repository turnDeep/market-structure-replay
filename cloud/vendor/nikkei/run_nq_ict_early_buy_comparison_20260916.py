"""Compare early-recovery BUY with confirmed-structure BUY.

The two variants consume one shared, causal candidate stream.  The stream is
generated once from the same daily OHLC, point-in-time zones, confirmed
pivots, and completed weekly/monthly context.  The only entry difference is:

* early_buy: enter after the bullish OB body is recovered, with an available
  lower-high reference;
* confirmed_buy: use the same reaction cohort, but wait for the lower-high
  close-sequence confirmation and departure rule.

Both variants use the same exit labels, execution clock, slippage,
commission, initial stop, macro protected-low manager, and episode gate.
The target dates mentioned in prior discussion are not trading conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nq_ict_retest_trade_state_machine_20260915 import load_inputs, number
from nq_ict_retest_trade_state_machine_revision_20260915 import RevisionEngine, load_revision_config
from nq_ict_macro_reversal_examples_20260916.common import (
    Bar, BuyEvidence, HigherTimeframe, MarketSnapshot, OrderIntent, Pivot,
    Position, Rules, SellEvidence, Zone, ZoneView, down_tick,
)
from nq_ict_macro_reversal_examples_20260916.proposal_01_market_episodes import MarketEpisodes
from nq_ict_macro_reversal_examples_20260916.proposal_02_macro_position_management import MacroPositionManager
from nq_ict_macro_reversal_examples_20260916.proposal_03_confirmed_top_sell import ConfirmedTopSell
from nq_ict_macro_reversal_examples_20260916.proposal_04_confirmed_ob_buy import ConfirmedObBuy
from nq_ict_macro_reversal_examples_20260916.proposal_05_episode_attempts import EpisodeAttemptGate
from run_nq_ict_macro_reversal_20260916 import CausalPivotAdapter, clean, digest, text, timestamp


ROOT = Path(__file__).resolve().parent
VERSION = "nq-ict-early-buy-comparison-0.1-executed"


@dataclass(frozen=True)
class SharedSignal:
    bar: Bar
    pivots: tuple[Pivot, ...]
    market: MarketSnapshot
    htf: HigherTimeframe
    early_buys: tuple[BuyEvidence, ...]
    confirmed_buys: tuple[BuyEvidence, ...]
    sells: tuple[SellEvidence, ...]


def write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(trades: list[dict[str, Any]], start: pd.Timestamp | None = None,
            end: pd.Timestamp | None = None) -> dict[str, Any]:
    rows = []
    for trade in trades:
        entry = pd.Timestamp(trade["entry_at"]).normalize()
        if start is not None and entry < start:
            continue
        if end is not None and entry > end:
            continue
        rows.append((pd.Timestamp(trade["exit_at"]).normalize(), trade))
    rows.sort(key=lambda item: (item[0], str(item[1]["position_id"])))
    pnl = [float(t["net_pnl_usd"]) for _, t in rows]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    cumulative = np.cumsum(pnl) if pnl else np.array([], dtype=float)
    high_water = np.maximum.accumulate(cumulative) if pnl else np.array([], dtype=float)
    dd = cumulative - high_water if pnl else np.array([], dtype=float)
    gross_loss = abs(sum(losses))
    holding = [(pd.Timestamp(t["exit_at"]) - pd.Timestamp(t["entry_at"])).days for _, t in rows]
    return dict(
        completed_trades=len(rows), wins=len(wins), losses=len(losses),
        win_rate_pct=round(len(wins) / len(rows) * 100, 4) if rows else None,
        net_pnl_usd=round(sum(pnl), 2), gross_profit_usd=round(sum(wins), 2),
        gross_loss_usd=round(-gross_loss, 2),
        profit_factor=round(sum(wins) / gross_loss, 6) if gross_loss else None,
        max_drawdown_usd=round(float(dd.min()), 2) if len(dd) else 0.0,
        median_holding_days=float(np.median(holding)) if holding else None,
        mean_holding_days=float(np.mean(holding)) if holding else None,
        exit_reasons=dict(Counter(str(t["reason"]) for _, t in rows)),
    )


class SharedCandidateReplay(RevisionEngine):
    """Generate the candidate stream once, before either execution ledger."""

    def __init__(self, daily: pd.DataFrame, objects: pd.DataFrame,
                 contexts: pd.DataFrame, cfg: Any, rules: Rules):
        super().__init__(daily, objects, contexts, cfg)
        self.rules = rules
        self.market = MarketEpisodes(rules)
        self.seller = ConfirmedTopSell(rules)
        self.buyer = ConfirmedObBuy(rules)
        self.adapter = CausalPivotAdapter(daily)
        self.known_pivots: tuple[Pivot, ...] = ()
        self.signals: list[SharedSignal] = []
        self.early_rows: list[BuyEvidence] = []
        self.confirmed_rows: list[BuyEvidence] = []
        self.sell_rows: list[SellEvidence] = []

    def _bar(self, row: pd.Series) -> Bar:
        return Bar(int(self.i), self.t, float(row.open), float(row.high), float(row.low),
                   float(row.close), number(row.atr_prior))

    def _zones(self) -> tuple[ZoneView, ...]:
        result: list[ZoneView] = []
        for old in sorted(self.zones.values(), key=lambda z: z.id):
            if old.available_at > self.t:
                continue
            result.append(ZoneView(
                Zone(old.id, old.kind, old.direction, old.timeframe,
                     int(self.d.datetime.searchsorted(old.origin_at, side="left")),
                     pd.Timestamp(old.origin_at),
                     int(self.d.datetime.searchsorted(old.available_at)),
                     pd.Timestamp(old.available_at), float(old.low), float(old.high),
                     float(old.wick_low), float(old.wick_high)),
                self.t, str(old.state)))
        return tuple(result)

    def _htf(self) -> HigherTimeframe:
        return HigherTimeframe(
            self.t,
            timestamp(self.c.get("1w_source_context_bar_end")),
            timestamp(self.c.get("1m_source_context_bar_end")),
            text(self.c.get("1w_structure_trend")),
            text(self.c.get("1m_structure_trend")),
            text(self.c.get("1w_bias_direction")),
            text(self.c.get("1m_bias_direction")),
        )

    def run(self) -> None:
        for self.i in range(len(self.d)):
            row = self.d.iloc[self.i]
            self.t = pd.Timestamp(row.datetime)
            self.c = self.ctx.loc[self.t].to_dict()
            bar = self._bar(row)
            self.activate()
            for zone in list(self.zones.values()):
                self.update_zone(zone, row, number(row.atr_prior))
            self.confirm_pivots()
            self.known_pivots = self.adapter.convert(self.pivots)
            zones = self._zones()
            htf = self._htf()
            market = self.market.on_close(bar, self.known_pivots)
            sells = self.seller.on_close(bar, market, self.known_pivots)
            confirmed = self.buyer.on_close(bar, market, self.known_pivots, zones, htf)
            early = self.buyer.early_result
            # This is the common regime clock.  It is deliberately independent
            # of either ledger's position and gate outcome.
            if sells:
                for sell in sells:
                    self.market.note_failed_reversal(bar, sell.reference)
            elif confirmed:
                representative = max(
                    confirmed,
                    key=lambda c: ((c.reference.origin_index if c.reference else -1), c.key),
                )
                self.market.confirm_upward_structure(
                    representative.episode_id, bar, representative.reference)
            self.signals.append(SharedSignal(
                bar, self.known_pivots, market, htf, tuple(early), tuple(confirmed), tuple(sells)))
            self.early_rows.extend(early)
            self.confirmed_rows.extend(confirmed)
            self.sell_rows.extend(sells)


class VariantLedger:
    """One execution ledger consuming the immutable shared signal stream."""

    def __init__(self, name: str, rules: Rules, cfg: Any,
                 matched_setup_ids: set[str]):
        self.name = name
        self.rules = rules
        self.cfg = cfg
        self.matched_setup_ids = set(matched_setup_ids)
        self.gate = EpisodeAttemptGate(rules)
        self.manager = MacroPositionManager(rules)
        self.position: Position | None = None
        self.pending: OrderIntent | None = None
        self.live_stop: float | None = None
        self.pending_structural_stop: tuple[str, float] | None = None
        self.counter = 0
        self.eligible_setup_ids: set[str] = set()
        self.trades: list[dict[str, Any]] = []
        self.execution_rows: list[dict[str, Any]] = []
        self.plan_rows: list[dict[str, Any]] = []
        self.state_rows: list[dict[str, Any]] = []
        self.previous_at: pd.Timestamp | None = None

    def _new_id(self) -> str:
        self.counter += 1
        return f"{self.name}:order:{self.counter}"

    def _roundtrip_cost(self) -> float:
        return 2.0 * float(self.cfg.commission_per_contract_side_usd) * int(self.cfg.contracts)

    def _cancel(self, at: pd.Timestamp, index: int, reason: str) -> None:
        if self.pending is None:
            return
        order = self.pending
        if order.side == "BUY":
            self.gate.cancel(order.id)
        self.execution_rows.append(dict(at=at, index=index, action="cancel",
                                        order_id=order.id, reason=reason))
        self.pending = None

    def _close(self, at: pd.Timestamp, index: int, price: float, reason: str) -> None:
        if self.position is None:
            raise ValueError("cannot close flat ledger")
        position = self.position
        cost = self._roundtrip_cost()
        net = (price - position.entry_price) * position.quantity * position.point_value - cost
        self.gate.closed(position.id, index, net, position.initial_risk_usd)
        trade = dict(position_id=position.id, episode_id=position.candidate.episode_id,
                     entry_at=position.entry_at, entry_price=position.entry_price,
                     exit_at=at, exit_price=price, reason=reason, net_pnl_usd=net,
                     entry_setup_id=position.candidate.setup_id,
                     entry_confirmation=position.candidate.confirmation)
        self.trades.append(trade)
        self.execution_rows.append(dict(at=at, index=index, action="position_closed", **trade))
        self.position = None
        self.live_stop = None

    def on_open(self, bar: Bar) -> None:
        if self.pending_structural_stop is not None:
            pid, stop = self.pending_structural_stop
            if self.position is not None and self.position.id == pid:
                self.live_stop = max(self.live_stop or -math.inf, stop)
                self.execution_rows.append(dict(at=bar.at, index=bar.index,
                                                action="structural_stop_effective",
                                                position_id=pid, stop=self.live_stop))
            self.pending_structural_stop = None

        slip = float(self.cfg.tick_size) * float(self.cfg.slippage_ticks)
        if self.position is not None and self.live_stop is not None and bar.open <= self.live_stop:
            self._close(bar.at, bar.index, bar.open - slip, "emergency_stop")
            self.execution_rows.append(dict(at=bar.at, index=bar.index,
                                            action="gap_emergency_stop", price=bar.open - slip))
            self._cancel(bar.at, bar.index, "emergency_stop_precedes_pending_exit")
            return

        if self.pending is not None:
            order = self.pending
            if order.side == "BUY":
                candidate = order.candidate
                if candidate is None:
                    raise ValueError("BUY intent without candidate")
                fill = bar.open + slip
                if fill <= candidate.emergency_stop:
                    self._cancel(bar.at, bar.index, "open_at_or_below_initial_stop")
                else:
                    risk = (fill - candidate.emergency_stop) * int(self.cfg.contracts) * float(self.cfg.point_value_usd)
                    self.position = Position(order.id, candidate, bar.index, bar.at, fill,
                                             int(self.cfg.contracts), float(self.cfg.point_value_usd), risk)
                    self.gate.filled(order.id, bar.index)
                    self.pending = None
                    self.live_stop = candidate.emergency_stop
                    self.execution_rows.append(dict(at=bar.at, index=bar.index,
                                                    action="buy_fill", order_id=order.id,
                                                    price=fill, stop=self.live_stop))
            elif order.side == "SELL":
                if self.position is None or self.position.id != order.position_id:
                    self._cancel(bar.at, bar.index, "no_matching_position")
                else:
                    fill = bar.open - slip
                    self.pending = None
                    self._close(bar.at, bar.index, fill, order.reason)
                    self.execution_rows.append(dict(at=bar.at, index=bar.index,
                                                    action="sell_fill", order_id=order.id,
                                                    price=fill))
            else:
                raise ValueError(f"unknown side: {order.side}")

        if self.position is not None and self.live_stop is not None and bar.low <= self.live_stop:
            stop_price = self.live_stop - slip
            self._close(bar.at, bar.index, stop_price, "emergency_stop")
            self.execution_rows.append(dict(at=bar.at, index=bar.index,
                                            action="intrabar_emergency_stop",
                                            price=stop_price))

    def _candidate_pool(self, signal: SharedSignal) -> tuple[BuyEvidence, ...]:
        for candidate in signal.early_buys:
            if candidate.setup_id:
                self.eligible_setup_ids.add(candidate.setup_id)
        if self.name == "early_buy":
            return tuple(c for c in signal.early_buys
                         if c.setup_id is not None and c.setup_id in self.matched_setup_ids)
        return tuple(c for c in signal.confirmed_buys
                     if c.setup_id is not None and c.setup_id in self.matched_setup_ids
                     and c.setup_id in self.eligible_setup_ids)

    def on_close(self, signal: SharedSignal) -> None:
        bar = signal.bar
        management = self.manager.on_close(bar, signal.pivots, self.position)
        all_sells = tuple(signal.sells) + ((management.exit_signal,) if management.exit_signal else ())
        intent: OrderIntent | None = None
        reason = "hold_major_wave"
        chosen_sells: list[SellEvidence] = []
        if self.position is not None:
            chosen_sells = [s for s in all_sells if s.index >= self.position.entry_index]
            if chosen_sells:
                chosen = management.exit_signal or chosen_sells[0]
                intent = OrderIntent(self._new_id(), "SELL", bar.at, bar.index,
                                     chosen.reason, position_id=self.position.id)
                reason = "confirmed_exit_for_next_open"
        elif all_sells:
            reason = "bearish_structure_observed_flat_no_short_order"
        else:
            candidates = self._candidate_pool(signal)
            for candidate in sorted(
                candidates,
                key=lambda c: ((c.reference.origin_index if c.reference else -1), c.key),
                reverse=True,
            ):
                allowed, why = self.gate.assess(candidate, bar, signal.pivots, False, False)
                if allowed:
                    intent = OrderIntent(self._new_id(), "BUY", bar.at, bar.index, why, candidate)
                    self.gate.reserve(intent.id, candidate)
                    reason = "early_entry_for_next_open" if self.name == "early_buy" else "confirmed_entry_for_next_open"
                    break
        if management.proposed_protected_low is not None and self.position is not None:
            low = management.proposed_protected_low.pivot
            stop = down_tick(low.price - self.rules.emergency_buffer_atr * bar.atr_prior,
                             self.rules.tick_size)
            self.pending_structural_stop = (self.position.id, stop)
        self.pending = intent
        self.plan_rows.append(dict(
            datetime=bar.at, variant=self.name, reason=reason,
            intent=intent, shared_early_count=len(signal.early_buys),
            shared_confirmed_count=len(signal.confirmed_buys),
            cohort_confirmed_count=len(self._candidate_pool(signal)) if not intent else None,
            shared_sell_count=len(signal.sells), management=management,
            selected_sell_count=len(chosen_sells),
        ))
        self.state_rows.append(dict(
            datetime=bar.at, variant=self.name,
            position_id=self.position.id if self.position else None,
            position="long" if self.position else "flat",
            pending_order_id=self.pending.id if self.pending else None,
            pending_side=self.pending.side if self.pending else None,
            live_stop=self.live_stop,
            episode_id=signal.market.episode.id if signal.market.episode else None,
            episode_phase=signal.market.episode.phase if signal.market.episode else None,
            weekly_trend=signal.htf.weekly_trend, monthly_trend=signal.htf.monthly_trend,
            weekly_bias=signal.htf.weekly_bias, monthly_bias=signal.htf.monthly_bias,
        ))
        self.previous_at = bar.at

    def final_state(self) -> dict[str, Any]:
        return dict(
            variant=self.name, open_position=self.position, pending_order=self.pending,
            live_stop=self.live_stop, right_censored=True,
            eligible_setup_count=len(self.eligible_setup_ids),
            episode_budgets=self.gate.budgets,
        )


def make_rules() -> Rules:
    return Rules(
        pivot_radius=5, buy_reference_radius=2, sell_support_radius=2,
        area_tolerance_atr=0.50, area_max_age=504,
        major_decline_atr=3.0, major_decline_bars=15, breakout_accept_closes=3,
        management_pullback_bars=20, management_pullback_atr=2.5,
        structural_exit_closes=2, top_min_separation=20, top_lifetime=60,
        sell_support_bars=5, sell_support_rise_atr=2.0, sell_departure_atr=1.0,
        sell_confirm_closes=2, sell_mode="close_sequence", reaction_lifetime=60,
        buy_reference_drop_atr=1.5, buy_reference_bars=3, buy_departure_atr=1.0,
        buy_confirm_closes=2, buy_mode="close_sequence", structure_buffer_atr=0.05,
        retest_tolerance_atr=0.10, require_linked_fvg=False,
        emergency_buffer_atr=0.10, tick_size=0.25,
        max_attempts_per_episode=2, max_loss_r_per_episode=2.0,
        retry_new_low_atr=0.50, retry_higher_low_atr=0.50,
    )


def write_variant(out: Path, ledger: VariantLedger) -> None:
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "trades.jsonl", ledger.trades)
    write_jsonl(out / "execution_events.jsonl", ledger.execution_rows)
    write_jsonl(out / "close_plans.jsonl", ledger.plan_rows)
    write_jsonl(out / "daily_states.jsonl", ledger.state_rows)
    write_jsonl(out / "gate_events.jsonl", ledger.gate.events)
    write_jsonl(out / "manager_events.jsonl", ledger.manager.events)
    (out / "final_state.json").write_text(
        json.dumps(clean(ledger.final_state()), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
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
        parser.error("output directory already exists")
    cfg = load_revision_config(args.base_config)
    daily, objects, contexts = load_inputs(args.prices, args.objects, args.contexts, cfg, args.as_of)
    rules = make_rules()
    common = SharedCandidateReplay(daily, objects, contexts, cfg, rules)
    common.run()
    early_ids = {x.setup_id for x in common.early_rows if x.setup_id}
    confirmed_ids = {x.setup_id for x in common.confirmed_rows if x.setup_id}
    cohort_ids = early_ids & confirmed_ids
    ledgers = {
        "early_buy": VariantLedger("early_buy", rules, cfg, cohort_ids),
        "confirmed_buy": VariantLedger("confirmed_buy", rules, cfg, cohort_ids),
    }
    for signal in common.signals:
        for ledger in ledgers.values():
            ledger.on_open(signal.bar)
        for ledger in ledgers.values():
            ledger.on_close(signal)

    comparison = {
        "input_scope": dict(data_start=daily.datetime.iloc[0], data_end=daily.datetime.iloc[-1], daily_rows=len(daily)),
        "candidate_cohort": dict(
            early_setup_ids=len(early_ids), confirmed_setup_ids=len(confirmed_ids),
            shared_setup_ids=len(cohort_ids), early_only=len(early_ids - confirmed_ids),
            confirmed_only=len(confirmed_ids - early_ids),
            early_rows=len(common.early_rows), confirmed_rows=len(common.confirmed_rows),
            sell_rows=len(common.sell_rows),
            execution_cohort_is_shared=True,
        ),
        "execution_clock": dict(decision="daily close", fill="next observed daily open",
                                 same_exit_labels=True, same_slippage=True, same_commission=True,
                                 same_initial_stop=True, same_episode_gate=True),
        "models": {
            name: dict(
                all=metrics(ledger.trades),
                recent_3y_entry_window=metrics(ledger.trades, pd.Timestamp("2023-08-27"), pd.Timestamp("2026-08-27")),
                buy_fills=sum(x.get("action") == "buy_fill" for x in ledger.execution_rows),
                sell_fills=sum(x.get("action") == "sell_fill" for x in ledger.execution_rows),
                final_open_position=ledger.final_state().get("open_position") is not None,
            ) for name, ledger in ledgers.items()
        },
        "limitations": [
            "This is a matched candidate comparison, not a claim of predictive superiority.",
            "The shared candidate stream uses the confirmed detector's causal market-state clock; neither ledger changes it.",
            "Only reactions with a lower-high reference available at early recovery and later confirmed by the other model enter the matched cohort.",
            "The matched cohort is selected after observing both evidence streams; early-only setups are reported but excluded from the matched performance table.",
            "Daily OHLC cannot resolve the intraday order of the high and low.",
            "The input is a Yahoo continuous NKD=F series with roll/adjustment limitations.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "shared_early_candidates.jsonl", common.early_rows)
    write_jsonl(args.output / "shared_confirmed_candidates.jsonl", common.confirmed_rows)
    write_jsonl(args.output / "shared_sell_evidence.jsonl", common.sell_rows)
    write_jsonl(args.output / "shared_daily_signals.jsonl", [
        dict(datetime=s.bar.at, index=s.bar.index,
             early_count=len(s.early_buys), confirmed_count=len(s.confirmed_buys),
             sell_count=len(s.sells), episode_id=s.market.episode.id if s.market.episode else None,
             episode_phase=s.market.episode.phase if s.market.episode else None)
        for s in common.signals
    ])
    for name, ledger in ledgers.items():
        write_variant(args.output / name, ledger)
    (args.output / "comparison.json").write_text(
        json.dumps(clean(comparison), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = dict(
        version=VERSION,
        result_kind="matched_candidate_early_vs_confirmed_buy_comparison",
        validation_status="pending_external_validator",
        inputs={name: dict(path=str(path.resolve()), sha256=sha256(path)) for name, path in (
            ("prices", args.prices), ("objects", args.objects), ("contexts", args.contexts),
            ("base_config", args.base_config),
        )},
        script_sha256=sha256(Path(__file__)), rules=clean(asdict(rules)),
        outputs=dict(shared_signals=len(common.signals), early_candidates=len(common.early_rows),
                     confirmed_candidates=len(common.confirmed_rows), sell_evidence=len(common.sell_rows)),
        comparison=comparison,
        restrictions=[
            "No target date is a trading condition",
            "No LuxAlgo or old route_signals/zone_signal/trail path is used",
            "One long position per variant; no short and no pyramiding",
            "The two variants consume the same causal shared signal stream",
            "Candidate cohort is restricted to setups with early recovery reference available",
            "Parameters are fixed engineering choices and were not fitted in this run",
        ],
    )
    (args.output / "manifest.json").write_text(
        json.dumps(clean(manifest), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# NKD=F 早期BUY対確認済みBUY 比較",
        "",
        f"- 対象: {comparison['input_scope']['data_start']} ～ {comparison['input_scope']['data_end']}",
        f"- 日足: {comparison['input_scope']['daily_rows']:,}本",
        "- 共通時計: 日足終値で判断、次の観測日足始値で約定",
        "- 共通ラベル: 同じSELL検知、構造ストップ、緊急ストップ、手数料、スリッページ、局面予算",
        "",
        "## 候補母集団",
        "",
        f"早期BUY候補のsetup ID: {len(early_ids)}、確認済みBUY候補のsetup ID: {len(confirmed_ids)}、両者の共通setup ID: {len(cohort_ids)}です。両モデルとも共通setup IDだけで実行しました。早期専用setup {len(early_ids - confirmed_ids)}件はカバレッジとして記録し、性能表から除外しています。",
        "",
        "| モデル | 完了取引 | 勝率 | 純損益USD | PF | 最大DD USD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in comparison["models"].items():
        m = result["all"]
        lines.append(f"| {name} | {m['completed_trades']} | {m['win_rate_pct']}% | {m['net_pnl_usd']:,.2f} | {m['profit_factor']} | {m['max_drawdown_usd']:,.2f} |")
    lines += [
        "",
        "## 直近約3年のエントリー窓",
        "",
        "| モデル | 完了取引 | 勝率 | 純損益USD | PF |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in comparison["models"].items():
        m = result["recent_3y_entry_window"]
        lines.append(f"| {name} | {m['completed_trades']} | {m['win_rate_pct']}% | {m['net_pnl_usd']:,.2f} | {m['profit_factor']} |")
    lines += [
        "",
        "## 解釈上の注意",
        "",
        "この比較で変えたのは、同じ反応候補に対してBUY意図を出す時点だけです。早期BUYが利益を増やしても、同じ候補・同じ出口・同じ時計を保った比較結果であり、ICTやOBの因果性を証明するものではありません。最終未決済ポジションは完了取引から除外します。",
        "",
    ]
    lines.extend(f"- {x}" for x in comparison["limitations"])
    (args.output / "report_ja.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Executed matched early/confirmed BUY comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()

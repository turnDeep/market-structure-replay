"""5案の接続例。価格取得/ループ再生/バックテスト/実発注は自動実行しない。

on_closeは観察と注文意図の作成。約定は次バー始値の執行層が通知する。
緊急stopのギャップ/場中判定は執行層が行い、on_emergency_stop_fillで通知。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from .common import (Bar, BuyEvidence, HigherTimeframe, MarketSnapshot, OrderIntent,
                     Pivot, Position, Rules, SellEvidence, ZoneView, check_bar, check_inputs)
from .proposal_01_market_episodes import MarketEpisodes
from .proposal_02_macro_position_management import MacroPositionManager, ManagementResult
from .proposal_03_confirmed_top_sell import ConfirmedTopSell
from .proposal_04_confirmed_ob_buy import ConfirmedObBuy
from .proposal_05_episode_attempts import EpisodeAttemptGate


@dataclass(frozen=True)
class CloseInputs:
    bar: Bar
    pivots: tuple[Pivot, ...]
    zones: tuple[ZoneView, ...]  # 日足/確定週足/確定月足のOBとFVG
    htf: HigherTimeframe


@dataclass(frozen=True)
class ClosePlan:
    at: datetime
    index: int
    market_before_signal: MarketSnapshot
    buys: tuple[BuyEvidence, ...]
    sells: tuple[SellEvidence, ...]
    management: ManagementResult
    intent: OrderIntent | None
    reason: str
    gate_reasons: tuple[tuple[str, str], ...]
    resulting_episode_phase: str | None


class MacroReversalPolicy:
    def __init__(self, rules: Rules | None = None):
        self.rules = rules or Rules()
        if min(self.rules.buy_confirm_closes, self.rules.sell_confirm_closes,
               self.rules.structural_exit_closes, self.rules.breakout_accept_closes) < 1:
            raise ValueError("close counts must be positive")
        self.market = MarketEpisodes(self.rules)
        self.manager = MacroPositionManager(self.rules)
        self.seller = ConfirmedTopSell(self.rules)
        self.buyer = ConfirmedObBuy(self.rules)
        self.gate = EpisodeAttemptGate(self.rules)
        self.position: Position | None = None
        self.pending: OrderIntent | None = None
        self.last_close: Bar | None = None
        self.history: list[ClosePlan] = []
        self.execution_log: list[dict] = []
        self.trades: list[dict] = []
        self.counter = 0

    def _new_id(self) -> str:
        self.counter += 1
        return f"order:{self.counter}"

    def on_close(self, inputs: CloseInputs) -> ClosePlan:
        b = inputs.bar
        check_bar(b, self.last_close)
        check_inputs(b, inputs.pivots, inputs.zones, inputs.htf)
        if self.pending is not None:
            raise ValueError("resolve previous close order at next open before feeding another close")
        market = self.market.on_close(b, inputs.pivots)
        sells = self.seller.on_close(b, market, inputs.pivots)
        buys = self.buyer.on_close(b, market, inputs.pivots, inputs.zones, inputs.htf)
        management = self.manager.on_close(b, inputs.pivots, self.position)
        all_sells = sells + ((management.exit_signal,) if management.exit_signal else ())

        # 市場の新しい構造確認は保有の有無に関係なく更新。
        # 同日反対方向が成立した場合は新規買いを保留し、判定理由を保存。
        if all_sells:
            for s in all_sells:
                self.market.note_failed_reversal(b, s.reference)
        elif buys:
            # 最新の戻り高値を使った確認を代表値にする。確率順位ではない。
            representative = max(buys, key=lambda c: (c.reference.origin_index, c.key))
            self.market.confirm_upward_structure(representative.episode_id, b, representative.reference)

        intent = None
        gate_reasons: list[tuple[str, str]] = []
        if self.position is not None:
            current_sells = [s for s in all_sells if s.index >= self.position.entry_index]
            if current_sells:
                # 主構造の失敗を先に扱い、同日の主要トップは同じ計画に全件保存。
                chosen = management.exit_signal or current_sells[0]
                intent = OrderIntent(self._new_id(), "SELL", b.at, b.index,
                                     chosen.reason, position_id=self.position.id)
                reason = "confirmed_exit_for_next_open"
            else:
                reason = "hold_major_wave"
        elif all_sells:
            reason = "bearish_structure_observed_flat_no_short_order"
        else:
            # 順位は最新参照高値→IDの固定順。勝率推定や7要素加点ではない。
            for c in sorted(buys, key=lambda c: (-c.reference.origin_index, c.key)):
                allowed, why = self.gate.assess(c, b, inputs.pivots, False, False)
                gate_reasons.append((c.key, why))
                if allowed:
                    intent = OrderIntent(self._new_id(), "BUY", b.at, b.index, why, c)
                    self.gate.reserve(intent.id, c)
                    break
            reason = "confirmed_entry_for_next_open" if intent else "wait_for_major_reversal"
        self.pending = intent
        phase = self.market.episode.phase if self.market.episode else None
        plan = ClosePlan(b.at, b.index, market, buys, all_sells, management,
                         intent, reason, tuple(gate_reasons), phase)
        self.history.append(plan)
        self.last_close = b
        return plan

    def cancel_pending(self, at: datetime, index: int, reason: str) -> None:
        order = self.pending
        if order is None or index < order.index or at < order.at:
            raise ValueError("no valid pending order")
        if order.side == "BUY":
            self.gate.cancel(order.id)
        self.execution_log.append(dict(at=at, index=index, action="cancel", order_id=order.id, reason=reason))
        self.pending = None

    def on_next_open_fill(self, intent_id: str, at: datetime, index: int,
                          price_after_slippage: float, quantity: int, point_value: float,
                          roundtrip_cost: float = 0.0) -> None:
        """実約定/明示した約定モデルからの通知。呼んだだけで実注文は出ない。

SELL時のroundtrip_costに往復手数料等を渡す。価格はスリッページ反映後。
BUYの枚数・証拠金・1取引リスク上限は執行層で先に決める。
"""
        order = self.pending
        if (order is None or order.id != intent_id or index != order.index + 1
                or at <= order.at):
            raise ValueError("fill must match the pending order at the next observed open")
        if not (isfinite(price_after_slippage) and price_after_slippage > 0
                and quantity > 0 and isfinite(point_value) and point_value > 0
                and isfinite(roundtrip_cost) and roundtrip_cost >= 0):
            raise ValueError("invalid execution values")
        if order.side == "BUY":
            c = order.candidate
            if c is None or self.position is not None:
                raise ValueError("invalid BUY state")
            if price_after_slippage <= c.emergency_stop:
                self.cancel_pending(at, index, "open_at_or_below_initial_stop")
                return
            risk = (price_after_slippage - c.emergency_stop) * quantity * point_value
            self.position = Position(order.id, c, index, at, price_after_slippage,
                                     quantity, point_value, risk)
            self.gate.filled(order.id, index)
            self.execution_log.append(dict(at=at, index=index, action="buy_filled",
                                           position_id=order.id, price=price_after_slippage))
        else:
            p = self.position
            if (p is None or p.id != order.position_id or p.quantity != quantity
                    or p.point_value != point_value):
                raise ValueError("SELL must close the actual matching position")
            self._close(at, index, price_after_slippage, roundtrip_cost, order.reason)
        self.pending = None

    def on_emergency_stop_fill(self, at: datetime, index: int,
                               price_after_slippage: float, roundtrip_cost: float) -> None:
        # 執行層がstop到達を確認して通知。日足の高安順序はここで推定しない。
        if (self.position is None or self.last_close is None
                or index != self.last_close.index + 1 or at <= self.last_close.at):
            raise ValueError("stop callback must precede the next daily close")
        if self.pending:
            self.cancel_pending(at, index, "emergency_stop_precedes_pending_exit")
        self._close(at, index, price_after_slippage, roundtrip_cost, "emergency_stop")

    def _close(self, at: datetime, index: int, price: float, costs: float, reason: str) -> None:
        p = self.position
        if (p is None or index < p.entry_index or at < p.entry_at
                or not isfinite(price) or price <= 0 or not isfinite(costs) or costs < 0):
            raise ValueError("invalid close notification")
        net = (price - p.entry_price) * p.quantity * p.point_value - costs
        self.gate.closed(p.id, index, net, p.initial_risk_usd)
        trade = dict(position_id=p.id, episode_id=p.candidate.episode_id,
                     entry_at=p.entry_at, entry_price=p.entry_price,
                     exit_at=at, exit_price=price, reason=reason, net_pnl_usd=net)
        self.trades.append(trade)
        self.execution_log.append(dict(at=at, index=index, action="position_closed", **trade))
        self.position = None


# 実行順:
# 1. 前終値の計画を受け、次バー始値にギャップstopを優先して執行層で判定。
# 2. 前日のintentを次の観測始値で約定/取消し、上記callbackを通知。
# 3. 日中の事前emergency_stop到達を執行層で処理。
# 4. 終値確定後、当時利用可能な足・Pivot・帯・確定週月足をon_closeへ。
# 5. 終端の建玉/保留注文はそのまま保存。終端で強制決済しない。

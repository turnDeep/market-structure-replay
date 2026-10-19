"""旧検知器と新しい参加ルールを接続する型。全日時は旧観測日足時計。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from nq_ict_macro_reversal_examples_20260916.common import BuyEvidence


Route = Literal["early", "confirmed"]


@dataclass(frozen=True)
class PriorityRules:
    # 未検証の配分仮説。旧Rulesの総回数上限・累積損失制限も別途適用。
    max_early_fills_per_episode: int = 1
    max_confirmed_fills_per_episode: int = 1
    track_after_early_rejection: bool = True
    enable_confirmed_entry: bool = True


@dataclass(frozen=True)
class ReactionCatalog:
    """当日の検知器更新完了後に有効な全reaction version。

    新安値・回復失敗でrevisionが変わるため、古いversionを監視から外せる。
    全期間の最終生存集合や、将来confirmedされる集合を渡してはいけない。
    """
    index: int
    at: datetime
    live_setup_ids: frozenset[str]


@dataclass(frozen=True)
class EntryOffer:
    route: Route
    kind: str  # original_early / deferred_local_break / observed_confirmation
    source_early: BuyEvidence  # 登録時の候補をそのまま保存
    decision: BuyEvidence     # 今回の注文判断日。遅延参加ではsourceとは異なる


@dataclass
class Watch:
    source: BuyEvidence
    local_trigger_emitted: bool = False
    confirmed_trigger_emitted: bool = False

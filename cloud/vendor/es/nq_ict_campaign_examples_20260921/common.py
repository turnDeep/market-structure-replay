"""Separate a campaign's lifecycle from daily detector evidence.

All indices are observed daily bars, not calendar days. No dates are signals.
Values below are provisional design choices, not optimized/ICT official values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nq_ict_macro_reversal_examples_20260916.common import (
    BuyEvidence, HighArea, Pivot, SellEvidence,
)
from nq_ict_priority_entry_examples_20260916.integration_example import Packet


@dataclass(frozen=True)
class CampaignConfig:
    # Candidate detection is retained; the new budget scope is separate.
    confirmed_entry_mode: str = "structure_or_retest"
    enable_post_structure_retest: bool = True
    # Sell support must be a major pivot, not an arbitrary local low.
    top_support_radius: int = 5
    top_exit_mode: str = "close_sequence"
    top_support_bars: int = 5
    top_support_rise_atr: float = 2.0
    # Hard initial stop stays. Protected lows normally use CLOSE exits only.
    enable_protected_intrabar_stop: bool = False
    reentry_mode: str = "fresh_support_same_episode"
    support_join_bars: int = 3
    # Revision 0.7: counts and historical campaign losses are descriptive.
    # The old policies remain explicit comparison modes, never implicit caps.
    attempt_allocation: str = "unlimited"
    campaign_loss_limit_mode: str = "observe_only"
    reentry_evidence_mode: str = "fresh_price_reaction"
    # Price structure is an observation feed shared with recovery BUY. Turning
    # off ordinary continuation orders must not turn off that feed.
    observe_continuation_structure: bool = True
    enable_continuation_buy: bool = False
    continuation_accept_closes: int = 3
    support_candidate_mode: str = "independent_ob_reactions"
    support_selection: str = "structural_or_leg"
    support_min_prior_departure_atr: float = 2.0
    post_sell_mode: str = "two_paths"
    post_sell_new_low_atr: float = 0.5
    failed_top_reclaim_closes: int = 2
    failed_top_reclaim_level: str = "peak"
    sell_detector: str = "double_top_early"
    double_top_qualification: str = "major_leg"
    top_leg_lookback: int = 252
    top_leg_min_bars: int = 20
    top_leg_min_atr: float = 4.0
    top_pullback_min_fraction: float = 0.15
    top_cluster_bars: int = 10
    top_cluster_atr: float = 0.5
    double_top_min_separation: int = 20
    double_top_max_separation: int = 252
    double_top_tolerance_atr: float = 0.5
    double_top_max_sweep_atr: float = 1.5
    double_top_min_depth_atr: float = 2.0
    double_top_reaction_window: int = 60
    double_top_require_confirmed_h2: bool = False
    # Supplementary daily detectors, not new mandatory early BUY filters.
    reference_lookback: int = 20
    sweep_reclaim_wait: int = 5
    mss_min_displacement_atr: float = 0.5
    fvg_min_ticks: int = 1
    fvg_min_atr: float = 0.1
    tactical_zone_lifetime: int = 30
    evidence_margin_atr: float = 0.5
    require_early_support_sweep: bool = False
    require_early_htf_ob_overlap: bool = False
    # Research-only directional gate. "up" means the latest completed HTF
    # state is bullish; unknown/mixed states do not pass these variants.
    buy_htf_filter: str = "none"
    # Revision 0.4: a historical structural role must be known before contact.
    structure_min_leg_bars: int = 20
    structure_min_leg_atr: float = 4.0
    structure_origin_window: int = 5
    structure_zone_tolerance_atr: float = 0.5
    reaction_filter: str = "observe_only"
    reaction_min_rebound_atr: float = 1.0
    reaction_min_close_location: float = 0.65
    # Revision 0.11: v0.10 recovery-only control, with ordinary continuation
    # orders disabled by default. Failure memory remains an opt-in comparison.
    management_mode: str = "entry_scale"
    reaction_failure_closes: int = 2
    wave_pivot_radius: int = 5
    wave_min_pullback_atr: float = 2.5
    wave_exit_closes: int = 2
    exit_reentry_policy: str = "episode_scoped"
    failed_reaction_policy: str = "off"
    exit_reclaim_closes: int = 2
    enable_recovery_buy: bool = True
    recovery_structure_policy: str = "post_exit_low"
    continuation_context_mode: str = "legacy_local"
    continuation_risk_mode: str = "legacy_structure"
    continuation_max_risk_atr: float = 3.0
    # Revision 0.6: reuse the qualified parent top as the risk-scope boundary.
    cycle_mode: str = "major_top_boundary"
    cycle_reclaim_closes: int = 3
    # Revision 0.5: bounded formation attribution, not lifetime best departure.
    leg_ob_window_daily: int = 10
    leg_ob_window_weekly: int = 30
    leg_ob_window_monthly: int = 60
    leg_ob_departure_atr: float = 1.0
    leg_ob_accept_closes: int = 3
    sell_representation: str = "parent_and_trigger"

    def __post_init__(self) -> None:
        for name, allowed in {
            "attempt_allocation": {"unlimited", "shared_total", "legacy_per_route"},
            "campaign_loss_limit_mode": {"observe_only", "enforce"},
            "reentry_evidence_mode": {"fresh_price_reaction", "legacy_material_support"},
            "support_candidate_mode": {"independent_ob_reactions", "legacy_watches"},
            "support_selection": {"structural_or_leg", "structural_role", "major_reaction", "legacy"},
            "post_sell_mode": {"two_paths", "legacy"},
            "failed_top_reclaim_level": {"peak", "reference"},
            "double_top_qualification": {"major_leg", "legacy"},
            "reaction_filter": {"observe_only", "fixed_thresholds"},
            "management_mode": {"thesis_wave", "entry_scale", "legacy"},
            "exit_reentry_policy": {"episode_scoped", "selective", "exit_reason", "off"},
            "failed_reaction_policy": {"episode_scoped", "reclaim_then_new_visit", "off"},
            "recovery_structure_policy": {"post_exit_low", "preserved_support_after_local_stop"},
            "continuation_context_mode": {"wave_or_reclaim", "legacy_local"},
            "continuation_risk_mode": {"reaction", "legacy_structure"},
            "cycle_mode": {"major_top_boundary", "independent_advance", "structure_recovery", "legacy"},
            "sell_representation": {"parent_and_trigger", "legacy_rank"},
            "buy_htf_filter": {"none", "weekly_trend_up", "monthly_trend_up",
                               "weekly_bias_up", "monthly_bias_up"},
        }.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"unsupported {name}")
        if self.reentry_mode not in {"fresh_support_same_episode", "new_market_episode"}:
            raise ValueError("unsupported reentry mode")
        if self.sell_detector not in {"major_area", "double_top_major", "double_top_early"}:
            raise ValueError("unsupported SELL detector")
        if self.double_top_max_separation < self.double_top_min_separation:
            raise ValueError("invalid double-top separation range")
        if self.failed_reaction_policy != "off" and self.exit_reentry_policy == "exit_reason":
            raise ValueError("v0.8 exit_reason comparison requires failed_reaction_policy=off")
        if self.failed_reaction_policy == "episode_scoped" and self.exit_reentry_policy != "episode_scoped":
            raise ValueError("episode-scoped failure memory requires episode_scoped reentry")
        if self.exit_reentry_policy == "episode_scoped" and self.failed_reaction_policy not in {"episode_scoped", "off"}:
            raise ValueError("episode_scoped cannot silently use v0.9 overlap memory")
        if self.enable_continuation_buy and not self.observe_continuation_structure:
            raise ValueError("continuation orders require their independent price-structure feed")
        if self.recovery_structure_policy != "post_exit_low" and not self.enable_recovery_buy:
            raise ValueError("preserved-support recovery requires recovery BUY to be enabled")
        if self.enable_recovery_buy and (
                self.exit_reentry_policy != "episode_scoped"
                or not self.observe_continuation_structure
                or self.management_mode != "entry_scale"
                or self.continuation_context_mode != "legacy_local"
                or self.continuation_risk_mode != "legacy_structure"
                or self.reentry_evidence_mode != "fresh_price_reaction"
                or self.attempt_allocation != "unlimited"
                or self.campaign_loss_limit_mode != "observe_only"):
            raise ValueError("recovery BUY requires the price-structure feed, v0.7-scale unlimited entries and episode_scoped clocks")
        if self.exit_reentry_policy == "selective" and (
                self.support_selection != "structural_or_leg"
                or self.continuation_context_mode != "legacy_local"):
            raise ValueError("selective reentry requires structural_or_leg and no v0.8 wave gate")
        if self.confirmed_entry_mode not in {"structure_only", "retest_only", "structure_or_retest"}:
            raise ValueError("unsupported confirmed entry mode")
        if self.top_exit_mode not in {"close_sequence", "breakdown_retest"}:
            raise ValueError("unsupported top exit mode")
        if self.confirmed_entry_mode != "structure_only" and not self.enable_post_structure_retest:
            raise ValueError("retest entry mode requires its detector")
        for name in ("top_support_radius", "top_support_bars", "reference_lookback",
                     "sweep_reclaim_wait", "fvg_min_ticks", "tactical_zone_lifetime",
                     "support_join_bars", "double_top_min_separation",
                     "double_top_max_separation", "double_top_reaction_window",
                     "failed_top_reclaim_closes", "top_leg_lookback", "top_leg_min_bars",
                     "top_cluster_bars", "structure_min_leg_bars", "structure_origin_window",
                     "cycle_reclaim_closes", "leg_ob_window_daily", "leg_ob_window_weekly",
                     "leg_ob_window_monthly", "leg_ob_accept_closes", "continuation_accept_closes",
                     "reaction_failure_closes", "wave_pivot_radius", "wave_exit_closes", "exit_reclaim_closes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("top_support_rise_atr", "double_top_tolerance_atr",
                     "double_top_max_sweep_atr", "double_top_min_depth_atr",
                     "mss_min_displacement_atr", "fvg_min_atr", "evidence_margin_atr",
                     "support_min_prior_departure_atr", "post_sell_new_low_atr",
                     "top_leg_min_atr", "top_pullback_min_fraction", "top_cluster_atr",
                     "structure_min_leg_atr", "structure_zone_tolerance_atr",
                     "reaction_min_rebound_atr", "reaction_min_close_location", "leg_ob_departure_atr"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.top_leg_lookback < self.top_leg_min_bars:
            raise ValueError("top leg lookback is shorter than its minimum duration")
        if self.top_pullback_min_fraction > 1:
            raise ValueError("top pullback fraction must be between zero and one")
        if self.reaction_min_close_location > 1:
            raise ValueError("close location must be between zero and one")
        from math import isfinite
        for name in ("wave_min_pullback_atr", "continuation_max_risk_atr"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.management_mode in {"thesis_wave", "entry_scale"} and self.support_selection not in {"structural_role", "structural_or_leg"}:
            raise ValueError("entry-scale management requires structural-role selection")
        if self.enable_continuation_buy and self.management_mode not in {"thesis_wave", "entry_scale"}:
            raise ValueError("continuation BUY requires explicit entry-scale management")
        if self.support_selection in {"structural_role", "structural_or_leg"} and self.support_candidate_mode != "independent_ob_reactions":
            raise ValueError("structural roles require independent OB observations")
        if self.cycle_mode == "major_top_boundary" and (
                self.sell_detector not in {"double_top_early", "double_top_major"}
                or self.double_top_qualification != "major_leg"
                or self.sell_representation != "parent_and_trigger"):
            raise ValueError("major-top budgets require qualified double tops and parent-and-trigger selection")


@dataclass(frozen=True)
class Evidence:
    id: str
    index: int
    at: Any
    kind: str
    direction: str
    low: float
    high: float
    source_id: str
    origin_index: int
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TacticalZone:
    id: str
    kind: str
    direction: str
    low: float
    high: float
    origin_index: int
    available_index: int
    parent_id: str | None = None


@dataclass
class TopAttempt:
    id: str
    area: HighArea
    touch_index: int
    peak_index: int
    peak: float
    support: Pivot | None = None
    support_selected_index: int | None = None
    below_count: int = 0
    first_break_index: int | None = None
    emitted: bool = False
    revision: int = 0
    linked_evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CampaignPacket:
    base: Packet
    evidence: tuple[Evidence, ...]
    tactical_zones: tuple[TacticalZone, ...]
    top_warnings: tuple[str, ...]
    top_details: tuple[dict, ...]
    major_reversal: MajorReversal | None = None
    support_roles: dict = field(default_factory=dict)
    major_structures: tuple = ()
    market_cycle: Any = None
    sell_selection: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MajorReversal:
    id: str
    index: int
    at: Any
    source_ids: tuple[str, ...]
    market_episode_id: str | None
    reason: str
    peak: float | None = None
    peak_index: int | None = None
    reference: Pivot | None = None
    atr: float | None = None
    signal_low: float | None = None
    parent_top_id: str | None = None
    parent_h1_index: int | None = None


@dataclass(frozen=True)
class MajorExit:
    index: int
    reason: str
    peak: float
    atr: float
    episode_id: str
    reversal_id: str | None = None
    peak_index: int | None = None
    reference: Pivot | None = None
    signal_low: float | None = None
    signal_index: int | None = None


@dataclass
class BuyPath:
    seed: BuyEvidence
    confirmed: BuyEvidence | None = None
    events: list[Evidence] = field(default_factory=list)
    seen_events: set[str] = field(default_factory=set)
    retest_keys: set[str] = field(default_factory=set)
    retired: bool = False


def context_label(htf: Any) -> str:
    """Trend is historical structure; the stored bias is not substituted for it."""
    if htf.weekly_completed_at is None or htf.monthly_completed_at is None:
        return "unknown_completed_context"
    if htf.weekly_trend == "up" and htf.monthly_trend in {"up", "mixed"}:
        return "uptrend_pullback_candidate"
    if htf.weekly_trend == "down":
        return "countertrend_reversal_candidate"
    return "mixed_or_transition_candidate"

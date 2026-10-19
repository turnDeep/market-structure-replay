"""Two causal post-SELL paths. Authored only; no performance claim.

Both paths still need an observable pre-existing bullish OB and an entry
trigger. Trend labels never substitute for a reclaim or a new support test.
"""
from __future__ import annotations

from dataclasses import asdict

from nq_ict_macro_reversal_examples_20260916.common import positive
from .structural_support import selected_support_role


class PostSellPathBook:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.context = None
        self.boundary = None
        self.reclaim_count = 0
        self.events = []
        self.snapshot = {}

    def on_close(self, bar, scope, cutoff, boundary, market_phase="initial"):
        token = (scope, cutoff)
        if token != self.context:
            self.context, self.boundary = token, boundary
            self.reclaim_count = 0
            self.events.append(dict(index=bar.index, action="post_sell_context_started",
                                    scope=scope, cutoff=cutoff,
                                    boundary=asdict(boundary) if boundary else None))
        b = self.boundary
        barrier = None
        if b is not None:
            if self.cfg.failed_top_reclaim_level == "peak":
                barrier = b.peak
            elif b.reference is not None:
                barrier = b.reference.price
        if (cutoff is not None and bar.index > cutoff and b is not None
                and b.atr is not None and positive(b.atr)
                and barrier is not None and positive(barrier)):
            threshold = barrier + self.rules.structure_buffer_atr * b.atr
            self.reclaim_count = self.reclaim_count + 1 if bar.close > threshold else 0
        else:
            self.reclaim_count = 0
        self.snapshot = dict(index=bar.index, scope=scope, cutoff=cutoff,
                             market_phase=market_phase,
                             reclaim_level=barrier, reclaim_closes=self.reclaim_count,
                             failed_top_reclaimed=self.reclaim_count >= self.cfg.failed_top_reclaim_closes,
                             mode=self.cfg.post_sell_mode)

    def assess(self, offer, obs, bar):
        scope, cutoff = self.context
        if (self.cfg.cycle_mode in {"structure_recovery", "independent_advance", "major_top_boundary"}
                and self.snapshot["market_phase"] == "advancing" and self.boundary is None):
            allowed = (obs is not None and obs.recovery_index is not None
                       and cutoff is not None and obs.contact_index > cutoff
                       and obs.recovery_index <= bar.index)
            detail = dict(path="established_uptrend_support" if allowed else "waiting_new_cycle_contact",
                          scope=scope, cutoff=cutoff, contact_index=obs.contact_index if obs else None)
            self.events.append(dict(index=bar.index, action="post_sell_path_assessed",
                                    candidate_key=offer.decision.key, allowed=allowed, **detail))
            return allowed, detail["path"], detail
        if self.cfg.post_sell_mode == "legacy" or cutoff is None:
            return True, "initial_or_legacy_path", dict(path="initial_or_legacy")
        b = self.boundary
        if (obs is None or b is None or b.atr is None or not positive(b.atr)
                or b.peak is None or b.peak_index is None or b.signal_low is None):
            return False, "post_sell_boundary_data_missing", dict(path="unknown_data")
        if bar.index <= cutoff or obs.contact_index <= cutoff or obs.recovery_index is None:
            return False, "post_sell_new_contact_and_recovery_required", dict(path="waiting")
        # Relative to the SELL decision, not a later/final low. The decline
        # duration starts at the known peak, not an arbitrary SELL cooldown.
        depth = (b.peak - obs.low) / b.atr
        duration = obs.low_index - b.peak_index
        incremental_drop = (b.signal_low - obs.low) / b.atr
        progressed = (obs.low_index > cutoff
                      and incremental_drop >= self.cfg.post_sell_new_low_atr
                      and depth >= self.rules.major_decline_atr
                      and duration >= self.rules.major_decline_bars)
        reclaimed = self.snapshot["failed_top_reclaimed"]
        detail = dict(scope=scope, path=("failed_top_reclaimed" if reclaimed else
                                        "decline_then_support" if progressed else "waiting"),
                      failed_top_reclaimed=reclaimed, decline_then_support=progressed,
                      decline_atr=depth, decline_bars=duration,
                      drop_below_signal_low_atr=incremental_drop,
                      known_peak=b.peak, known_peak_index=b.peak_index,
                      sell_signal_low=b.signal_low,
                      reclaim_level=self.snapshot["reclaim_level"],
                      reclaim_closes=self.reclaim_count, cutoff=cutoff,
                      contact_index=obs.contact_index, recovery_index=obs.recovery_index,
                      boundary_signal_index=getattr(b, "signal_index", b.index))
        allowed = reclaimed or progressed
        reason = detail["path"] if allowed else "post_sell_small_bounce_not_a_new_major_path"
        self.events.append(dict(index=bar.index, action="post_sell_path_assessed",
                                candidate_key=offer.decision.key, allowed=allowed, **detail))
        return allowed, reason, detail


def qualify_support(offer, obs, market, rules, cfg, post_sell_detail):
    """Fixed eligibility, not a confluence score or a probability."""
    if cfg.support_selection == "legacy":
        return True, "legacy_support_selection", {}
    if obs is None:
        return False, "support_observation_missing", {}
    features = dict(obs.features)
    if cfg.support_selection in {"structural_role", "structural_or_leg"}:
        structural = features.get("structural_support", {})
        ep = market.episode
        features["major_correction_check"] = dict(status="unknown_data", passed=None)
        if ep is not None and positive(ep.anchor.atr_at_origin):
            depth = (ep.anchor.price - obs.low) / ep.anchor.atr_at_origin
            duration = obs.low_index - ep.anchor.origin_index
            features["major_correction_check"] = dict(
                status="observed", decline_atr=depth, decline_bars=duration,
                anchor_id=ep.anchor.id,
                passed=depth >= rules.major_decline_atr and duration >= rules.major_decline_bars)
        candidates = {"major_role": structural.get("role")}
        if cfg.support_selection == "structural_or_leg":
            candidates["upward_leg_role"] = structural.get("upward_leg_role")
        features["role_checks"] = {
            name: dict(present=value is not None, known_index=value["known_index"] if value else None,
                       precontact=(value["known_index"] < obs.contact_index
                                   and value["effective_index"] <= obs.contact_index) if value else False)
            for name, value in candidates.items()}
        role = selected_support_role(obs, cfg)
        features["selected_role"] = role
        features["support_route"] = role["kind"] if role else "no_precontact_role"
        features["zone_preexisted_contact"] = offer.decision.zone.available_index < obs.contact_index
        if not features["zone_preexisted_contact"]:
            return False, "source_ob_not_available_before_contact", features
        if role is None:
            return False, ("no_major_or_bounded_leg_ob_role_known_before_contact"
                           if cfg.support_selection == "structural_or_leg"
                           else "no_major_ob_role_known_before_contact"), features
        if (role["known_index"] >= obs.contact_index
                or structural["known_through_index"] >= obs.contact_index):
            return False, "structural_role_not_precontact", features
        if post_sell_detail.get("path") in {"exit_reclaim", "fresh_structure_after_exit"}:
            return True, "precontact_support_with_exit_recovery_or_new_wave", features
        if post_sell_detail.get("path") == "fresh_major_support_after_exit":
            return _major_decline_context(obs, market, rules, features)
        if post_sell_detail.get("path") in {
                "failed_top_reclaimed", "decline_then_support", "established_uptrend_support"}:
            if (role["kind"] == "upward_leg_support"
                    and post_sell_detail.get("path") != "decline_then_support"):
                # The additional route does not turn every ordinary uptrend
                # retest into a BUY. It still needs a substantial correction.
                return _major_decline_context(obs, market, rules, features)
            return True, "precontact_structural_support_and_market_path", features
        # Initial participation still requires a genuine major decline. In this
        # mode lifetime maximum departure is descriptive, never an eligibility gate.
        ep = market.episode
        if ep is None or not positive(ep.anchor.atr_at_origin):
            return False, "major_decline_context_missing", features
        depth = (ep.anchor.price - obs.low) / ep.anchor.atr_at_origin
        duration = obs.low_index - ep.anchor.origin_index
        features.update(decline_atr=depth, decline_bars=duration)
        if depth < rules.major_decline_atr or duration < rules.major_decline_bars:
            return False, "support_outside_required_major_decline", features
        return True, "precontact_structural_support_after_major_decline", features
    # Frozen before the first contact of this reaction; the current rebound
    # cannot manufacture its own historical departure qualification.
    departure = features.get("prior_departure_atr")
    if departure is None:
        return False, "precontact_ob_history_missing", features
    if departure < cfg.support_min_prior_departure_atr:
        return False, "precontact_ob_departure_too_small", features
    if post_sell_detail.get("path") in {"failed_top_reclaimed", "decline_then_support"}:
        return True, "qualified_ob_and_post_sell_path", features
    ep = market.episode
    if ep is None or not positive(ep.anchor.atr_at_origin):
        return False, "major_decline_context_missing", features
    depth = (ep.anchor.price - obs.low) / ep.anchor.atr_at_origin
    duration = obs.low_index - ep.anchor.origin_index
    features.update(decline_atr=depth, decline_bars=duration)
    if depth < rules.major_decline_atr or duration < rules.major_decline_bars:
        return False, "support_outside_required_major_decline", features
    return True, "qualified_major_decline_ob_reaction", features


def _major_decline_context(obs, market, rules, features):
    ep = market.episode
    if ep is None or not positive(ep.anchor.atr_at_origin):
        return False, "leg_ob_major_correction_context_missing", features
    depth = (ep.anchor.price - obs.low) / ep.anchor.atr_at_origin
    duration = obs.low_index - ep.anchor.origin_index
    features.update(decline_atr=depth, decline_bars=duration,
                    decline_anchor_id=ep.anchor.id)
    if (not ep.anchor.origin_index < obs.low_index
            or depth < rules.major_decline_atr or duration < rules.major_decline_bars):
        return False, "leg_ob_outside_required_major_correction", features
    return True, "bounded_leg_ob_after_major_correction", features

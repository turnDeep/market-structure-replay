"""Support observations, fresh-event execution and historical counts. UNEXECUTED.

Market episode IDs and detector BuyEvidence are never rewritten. A separate
reversal ID groups attempts. Observed support is distinct from the original
candidate's touch/low: an old contact can precede a new, post-exit contact.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

from nq_ict_macro_reversal_examples_20260916.common import positive
from nq_ict_priority_entry_examples_20260916.common import EntryOffer
from .common import MajorReversal


class ReversalBook:
    """Source-side clock: independent of portfolio fills and stops."""

    def __init__(self):
        self.current = None
        self.armed = True
        self.seen_sources = set()
        self.events = []

    def on_close(self, bar, sells, confirmed_buys, market, top_details=(), allow_local_rearm=True,
                 refresh_context=False, budget_authority=None):
        fresh = [s for s in sells if s.source_id not in self.seen_sources]
        if sells:
            self.seen_sources.update(s.source_id for s in sells)
            if fresh and (self.armed or refresh_context):
                ids = tuple(sorted({s.source_id for s in fresh}))
                if len(sells) != 1:
                    raise ValueError("supply the canonical SELL representative to ReversalBook")
                chosen = sells[0]
                detail = next((d for d in top_details if d["top_id"] == chosen.source_id), {})
                self.current = MajorReversal(
                    f"reversal:{bar.index}:{ids[0]}", bar.index, bar.at, ids,
                    market.episode.id if market.episode else None, chosen.reason,
                    peak=detail.get("peak"), peak_index=detail.get("peak_index"),
                    reference=chosen.reference, atr=bar.atr_prior, signal_low=bar.low,
                    parent_top_id=detail.get("parent_top_id", chosen.source_id),
                    parent_h1_index=detail["h1"].origin_index if detail.get("h1") else None)
                self.armed = False
                self.events.append(dict(index=bar.index, action="major_reversal_started",
                                        reversal=self.current,
                                        context_only=refresh_context,
                                        budget_authority=(budget_authority or "IndependentAdvanceBook")
                                        if refresh_context else "legacy_scope"))
            else:
                self.events.append(dict(index=bar.index, action="same_decline_sell_no_budget_reset",
                                        reversal_id=self.current.id if self.current else None,
                                        source_ids=tuple(s.source_id for s in sells)))
        elif allow_local_rearm and confirmed_buys and self.current and bar.index > self.current.index:
            # A later independently detected upward structure permits the NEXT
            # distinct major top to start a new cycle. This is not a BUY fill.
            if not self.armed:
                self.events.append(dict(index=bar.index, action="new_reversal_can_be_armed",
                                        after_reversal_id=self.current.id))
            self.armed = True
        return self.current


@dataclass
class SupportObservation:
    zone: object
    group_id: str
    contact_index: int
    low: float
    low_index: int
    low_atr: float
    recovery_index: int | None = None
    first_below_index: int | None = None
    latest_contact_index: int | None = None
    features: dict = field(default_factory=dict)
    away_bars: int = 0


class SupportReactionBook:
    def __init__(self, rules, cfg):
        self.rules, self.cfg = rules, cfg
        self.context = None
        self.active = {}
        self.blocked = set()
        self.groups = {}
        self.offered_reactions = set()
        self.events = []

    def _group(self, scope, zone, bar):
        # Freeze grouping membership when first observed. Never retrospectively
        # merge past trades when another OB is discovered later.
        matches = [g for g in self.groups.values()
                   if bar.index - g["last_contact"] <= self.cfg.support_join_bars
                   and zone.wick_low <= g["high"] and zone.wick_high >= g["low"]]
        if matches:
            group = min(matches, key=lambda g: (g["first_contact"], g["id"]))
            group["last_contact"] = bar.index
            return group["id"]
        gid = f"{scope}:support:{bar.index}:{len(self.groups) + 1}"
        self.groups[gid] = dict(id=gid, low=zone.wick_low, high=zone.wick_high,
                                first_contact=bar.index, last_contact=bar.index)
        return gid

    def on_close(self, bar, zones, scope, cutoff, history_features=None):
        context = (scope, cutoff)
        if context != self.context:
            self.context = context
            self.active, self.blocked, self.groups = {}, set(), {}
            self.events.append(dict(index=bar.index, action="support_observation_boundary",
                                    budget_scope_id=scope, after_index=cutoff))
        if ((cutoff is None and self.cfg.support_candidate_mode == "legacy_watches")
                or (cutoff is not None and bar.index <= cutoff) or not positive(bar.atr_prior)):
            return
        views = {v.zone.id: v for v in zones}
        for zid in list(self.active):
            if zid not in views or not views[zid].observable:
                del self.active[zid]
                self.blocked.add(zid)
                self.events.append(dict(index=bar.index, action="support_retired",
                                        zone_id=zid, reason="source_not_observable"))
        for zid, view in sorted(views.items()):
            z = view.zone
            if (not view.observable or z.kind != "ob_candidate" or z.direction != "bullish"
                    or z.available_index >= bar.index or z.available_at >= bar.at):
                continue
            touch = bar.low <= z.wick_high and bar.high >= z.wick_low
            if zid in self.blocked:
                # A timed-out reaction needs a later departure and a NEW visit.
                # The source must itself permit trading; no resurrected zone.
                if view.tradeable and bar.low > z.wick_high:
                    self.blocked.remove(zid)
                continue
            old_obs = self.active.get(zid)
            if (old_obs is not None and touch and old_obs.recovery_index is not None
                    and old_obs.away_bars >= self.cfg.support_join_bars
                    and self.cfg.support_candidate_mode == "independent_ob_reactions"):
                self.events.append(dict(index=bar.index, action="separate_ob_revisit",
                                        zone_id=zid, prior_support_episode_id=old_obs.group_id,
                                        prior_contact_index=old_obs.contact_index,
                                        departure_bars=old_obs.away_bars,
                                        budget_reset=False))
                del self.active[zid]
            if zid not in self.active:
                if not touch:
                    continue
                self.active[zid] = SupportObservation(
                    z, self._group(scope, z, bar), bar.index, bar.low,
                    bar.index, bar.atr_prior, latest_contact_index=bar.index,
                    features=dict((history_features or {}).get(zid, {})))
                self.events.append(dict(index=bar.index, action="post_boundary_contact",
                                        zone_id=zid, support_episode_id=self.active[zid].group_id,
                                    source_available_index=z.available_index))
            obs = self.active[zid]
            if obs.zone != z:
                raise ValueError("support zone geometry changed")
            if bar.index - obs.contact_index > self.rules.reaction_lifetime:
                self.blocked.add(zid)
                del self.active[zid]
                self.events.append(dict(index=bar.index, action="support_retired", zone_id=zid,
                                        reason="reaction_lifetime_expired"))
                continue
            if touch:
                obs.latest_contact_index = bar.index
                self.groups[obs.group_id]["last_contact"] = bar.index
                obs.away_bars = 0
            else:
                obs.away_bars = obs.away_bars + 1 if bar.low > z.wick_high else 0
            if bar.low < obs.low:
                obs.low, obs.low_index, obs.low_atr = bar.low, bar.index, bar.atr_prior
                obs.recovery_index = None
            if bar.close < z.body_high:
                obs.recovery_index = None
            # Bounded grace after a close below the far edge. New lows do not
            # restart the same reclaim clock. Upstream state can expire sooner.
            if bar.close < z.wick_low and obs.first_below_index is None:
                obs.first_below_index = bar.index
            if (obs.first_below_index is not None
                    and bar.index - obs.first_below_index > self.cfg.sweep_reclaim_wait):
                self.blocked.add(zid)
                del self.active[zid]
                self.events.append(dict(index=bar.index, action="support_retired", zone_id=zid,
                                        reason="reclaim_grace_expired"))
                continue
            if view.tradeable and bar.close >= z.body_high and bar.close > bar.open:
                if obs.recovery_index is None:
                    obs.recovery_index = bar.index
                    kind = ("touch_recovery" if obs.low >= z.wick_low else
                            "same_bar_sweep_reclaim" if obs.low_index == bar.index else
                            "multi_bar_reclaim")
                    self.events.append(dict(index=bar.index, action="post_boundary_recovery",
                                            zone_id=zid, support_episode_id=obs.group_id,
                                            contact_index=obs.contact_index, low_index=obs.low_index,
                                            recovery_type=kind, intrabar_order="unknown_data"))
                obs.first_below_index = None

    def assess(self, offer, bar):
        c = offer.decision
        obs = self.active.get(c.zone.id)
        if obs is None:
            return False, "no_fresh_post_boundary_ob_contact", None
        if obs.recovery_index is None:
            return False, "post_boundary_ob_recovery_missing", obs
        # A structural retest retains its ORIGINAL breakout timestamp. The new
        # event is the explicitly identified later retest, not an invented MSS.
        trigger = c.index if offer.kind == "post_structure_zone_retest" else c.breakout_index
        if c.recovery_index < obs.recovery_index or trigger < obs.recovery_index:
            return False, "entry_evidence_precedes_new_support_recovery", obs
        if not (obs.contact_index <= obs.low_index <= obs.recovery_index <= bar.index):
            return False, "new_support_chronology_not_ready", obs
        return True, "fresh_support_reaction_same_market_episode_allowed", obs

    def local_offers(self, bar, previous, market, htf, watches, zones, already_offered):
        """New observed local breaks after the boundary, never stale-order replay.

        The existing tracker still owns source lifetime/revision and market
        eligibility. Its immutable seed is retained as source_early. Only the
        separate decision event gets today's recovery/trigger clock. The
        original wider stop/low is retained, not silently tightened to a bounce.
        """
        if (self.context is None or self.context[1] is None or previous is None
                or market.episode is None or market.episode.phase != "declining"):
            return ()
        views = {v.zone.id: v for v in zones}
        result = []
        for sid, watch in sorted(watches.items()):
            c = watch.source
            obs, view = self.active.get(c.zone.id), views.get(c.zone.id)
            if (obs is None or obs.recovery_index is None or view is None or not view.tradeable
                    or c.episode_id != market.episode.id
                    or watch.confirmed_trigger_emitted or c.index > bar.index
                    or obs.low < c.low or not positive(c.emergency_stop)
                    or not c.emergency_stop < bar.close):
                continue
            if any(o.decision.setup_id == sid and self.assess(o, bar)[0] for o in already_offered):
                continue  # Today's existing offer already contains the fresh reaction.
            if not (bar.close >= c.zone.body_high and bar.close > previous.high):
                continue
            token = (self.context, sid, obs.group_id, obs.low_index, obs.recovery_index)
            if token in self.offered_reactions:
                continue
            current = replace(c, key=f"{c.key}:post_boundary:{self.context[0]}:{bar.index}",
                              at=bar.at, index=bar.index, recovery_index=obs.recovery_index,
                              breakout_index=bar.index, confirmation="post_boundary_local_break", htf=htf)
            result.append(EntryOffer("early", "post_boundary_local_break", c, current))
            self.offered_reactions.add(token)
            self.events.append(dict(index=bar.index, action="fresh_support_local_offer",
                                    key=current.key, setup_id=sid, source_key=c.key,
                                    source_recovery_index=c.recovery_index,
                                    post_boundary_recovery_index=obs.recovery_index,
                                    support_episode_id=obs.group_id,
                                    original_low_and_stop_preserved=True))
        return tuple(result)


@dataclass
class CycleBudget:
    attempts: int = 0
    cumulative_loss_r: float = 0.0
    last_exit_index: int | None = None
    last_candidate: object | None = None
    last_support: dict | None = None


class CampaignAttemptGate:
    """Execution-compatible gate with an explicit scope, not rewritten evidence."""

    def __init__(self, rules, priority, cfg):
        self.rules, self.priority = rules, priority
        self.cfg = cfg
        self.budgets = {}
        self.route_fills = defaultdict(lambda: {"early": 0, "confirmed": 0})
        self.offers, self.contexts, self.reserved, self.position_episodes = {}, {}, {}, {}
        self.used_keys, self.filled_support_tokens = set(), set()
        self.last_exit_index = None
        self.events = []

    def _support_token(self, candidate, scope, obs):
        if candidate.confirmation == "uptrend_recovery":
            return ("uptrend_recovery", candidate.exit_context_id, candidate.episode_id)
        if self.cfg.reentry_evidence_mode == "fresh_price_reaction":
            # A new market scope cannot disguise the same price observation.
            # Conversely, a new visit to the same OB/FVG is not a lifetime ban.
            return (candidate.zone.id, obs["contact_index"], obs["low_index"],
                    obs["recovery_index"])
        return (scope, obs["group_id"], obs["low_index"])

    def register_offer(self, offer, scope, support=None):
        c = offer.decision
        if c.setup_id != offer.source_early.setup_id or c.episode_id != offer.source_early.episode_id:
            raise ValueError("source and decision identities must be preserved")
        if offer.route not in {"early", "confirmed"}:
            raise ValueError("unknown route")
        if c.key in self.offers and self.offers[c.key] != offer:
            raise ValueError("candidate evidence changed")
        # Store a value snapshot; a future low must not mutate a filled attempt.
        detail = (dict(group_id=support.group_id, contact_index=support.contact_index,
                       low=support.low, low_index=support.low_index, low_atr=support.low_atr,
                       recovery_index=support.recovery_index) if support else
                  dict(group_id=f"{scope}:low:{c.low_index}", contact_index=c.touch_index,
                       low=c.low, low_index=c.low_index, low_atr=c.low_atr,
                       recovery_index=c.recovery_index))
        self.offers[c.key] = offer
        self.contexts[c.key] = (scope, detail)

    def assess(self, c, bar, pivots, has_position, pending):
        result = self.diagnose(c, bar, pivots, has_position, pending)
        return result["allowed"], result["reason"]

    def diagnose(self, c, bar, pivots, has_position, pending):
        """Read-only eligibility of EVERY budget gate before any reservation."""
        checks = {}

        def add(name, passed, reason, status="evaluated", **detail):
            checks[name] = dict(passed=passed, status=status, reason=reason, **detail)

        offer = self.offers.get(c.key)
        registered = offer is not None and offer.decision == c and c.key in self.contexts
        add("position_order", not (has_position or pending or self.reserved), "position_or_order_exists")
        add("registration", registered, "unregistered_or_changed_entry_offer")
        add("today_clock", c.index == bar.index and c.at == bar.at, "fresh_today_confirmation_required")
        add("unused_evidence", c.key not in self.used_keys, "evidence_already_reserved")
        add("not_exit_day", self.last_exit_index is None or bar.index > self.last_exit_index,
            "same_day_reentry_blocked", last_exit_index=self.last_exit_index)
        success = "registered_context_required"
        if registered:
            scope, obs = self.contexts[c.key]
            budget = self.budgets.get(scope, CycleBudget())
            if self.cfg.attempt_allocation == "unlimited":
                add("total_attempts", True, "purchase_count_observed_not_limited",
                    status="not_applicable", used=budget.attempts, limit=None)
            else:
                add("total_attempts", budget.attempts < self.rules.max_attempts_per_episode,
                    "campaign_attempt_limit", used=budget.attempts, limit=self.rules.max_attempts_per_episode)
            if self.cfg.campaign_loss_limit_mode == "observe_only":
                add("cumulative_loss", True, "campaign_loss_observed_not_an_entry_gate",
                    status="not_applicable", used_r=budget.cumulative_loss_r, limit_r=None)
            else:
                add("cumulative_loss", budget.cumulative_loss_r < self.rules.max_loss_r_per_episode,
                    "campaign_loss_limit", used_r=budget.cumulative_loss_r, limit_r=self.rules.max_loss_r_per_episode)
            if self.cfg.attempt_allocation == "legacy_per_route":
                cap = (self.priority.max_early_fills_per_episode if offer.route == "early"
                       else self.priority.max_confirmed_fills_per_episode)
                used = self.route_fills.get(scope, {}).get(offer.route, 0)
                add("route_attempts", used < cap, f"{offer.route}_campaign_route_limit", used=used, limit=cap)
            else:
                add("route_attempts", True, "no_route_purchase_cap", status="not_applicable")
            token = self._support_token(c, scope, obs)
            add("unique_support", token not in self.filled_support_tokens, "same_support_reaction_already_filled")
            if self.cfg.reentry_evidence_mode == "fresh_price_reaction":
                # Use the ledger-wide exit clock, even after a scope change.
                # No monetary result, fixed retry count or new-OB requirement.
                exit_index = self.last_exit_index
                fresh = (exit_index is None or
                         (obs["recovery_index"] is not None
                          and obs["recovery_index"] > exit_index and c.index > exit_index))
                add("post_exit_clock", fresh, "recovery_and_trigger_must_follow_last_exit",
                    last_exit_index=exit_index, recovery_index=obs["recovery_index"],
                    trigger_index=c.index)
                add("material_new_support", fresh, "fresh_price_reaction_not_retry_quota",
                    same_zone_permitted=True, purchase_count_used=False)
                success = f"{offer.route}:fresh_price_evidence_without_purchase_quota"
            elif budget.last_exit_index is None:
                add("post_exit_clock", True, "first_attempt", status="not_applicable")
                add("material_new_support", True, "first_attempt", status="not_applicable")
                success = f"{offer.route}:first_attempt_in_campaign_scope"
            else:
                old, exit_index = budget.last_support, budget.last_exit_index
                trigger = c.index if offer.kind == "post_structure_zone_retest" else c.breakout_index
                clock = obs["recovery_index"] is not None and obs["recovery_index"] > exit_index and trigger > exit_index
                add("post_exit_clock", clock, "new_recovery_and_trigger_required_after_stop")
                lower = (positive(old["low_atr"]) and obs["low_index"] > exit_index
                         and obs["low"] <= old["low"] - self.rules.retry_new_low_atr * old["low_atr"])
                different = obs["contact_index"] > exit_index and c.zone.wick_high < budget.last_candidate.zone.wick_low
                higher = positive(old["low_atr"]) and any(
                    p.kind == "low" and p.radius >= self.rules.sell_support_radius
                    and exit_index < p.origin_index < trigger and p.available_index < trigger
                    and p.price >= old["low"] + self.rules.retry_higher_low_atr * old["low_atr"]
                    and p.price < c.reference.price for p in pivots)
                new_reaction = (obs["low_index"] > exit_index or
                    (obs["group_id"] != old["group_id"] and obs["contact_index"] > exit_index))
                add("material_new_support", new_reaction and (lower or different or higher),
                    "no_material_new_support_since_exit", lower=lower, different=different, higher=higher,
                    new_reaction=new_reaction)
                success = f"{offer.route}:campaign_retry_with_new_support"
        else:
            for name in ("total_attempts", "cumulative_loss", "route_attempts", "unique_support",
                         "post_exit_clock", "material_new_support"):
                add(name, None, "registered_context_missing", status="unknown_data")
        failed = [v["reason"] for v in checks.values() if v["passed"] is not True]
        return dict(allowed=not failed, reason=failed[0] if failed else success,
                    failed_reasons=failed, checks=checks, mutates_budget=False)

    def reserve(self, intent_id, candidate):
        if self.reserved or candidate.key in self.used_keys:
            raise ValueError("duplicate reservation")
        scope, obs = self.contexts[candidate.key]
        self.used_keys.add(candidate.key)
        self.reserved[intent_id] = (candidate, scope, dict(obs), self.offers[candidate.key].route)

    def cancel(self, intent_id):
        self.reserved.pop(intent_id, None)

    def filled(self, intent_id, index):
        c, scope, obs, route = self.reserved.pop(intent_id)
        b = self.budgets.setdefault(scope, CycleBudget())
        b.attempts += 1
        b.last_candidate, b.last_support = c, obs
        self.route_fills[scope][route] += 1
        self.position_episodes[intent_id] = (scope, c, obs)
        self.filled_support_tokens.add(self._support_token(c, scope, obs))
        self.events.append(dict(index=index, action="campaign_attempt_filled", position_id=intent_id,
                                market_episode_id=c.episode_id, budget_scope_id=scope,
                                support_episode_id=obs["group_id"], support=obs,
                                route=route, attempts=b.attempts))

    def closed(self, position_id, index, net_pnl, initial_risk):
        if not positive(initial_risk):
            raise ValueError("positive initial risk required")
        scope, c, obs = self.position_episodes.pop(position_id)
        b = self.budgets[scope]
        b.cumulative_loss_r += max(0.0, -net_pnl / initial_risk)
        b.last_exit_index = self.last_exit_index = index
        b.last_candidate, b.last_support = c, obs
        self.events.append(dict(index=index, action="campaign_attempt_closed", position_id=position_id,
                                budget_scope_id=scope, cumulative_loss_r=b.cumulative_loss_r))

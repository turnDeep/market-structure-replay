"""Long-only campaign using inherited accounting and a separate attempt scope.

No broker API. Running this in a future replay would simulate fills; none have
been generated as part of authoring. New SELL replaces the old local top SELL.
"""
from __future__ import annotations

from copy import deepcopy

from nq_ict_macro_reversal_examples_20260916.common import (
    OrderIntent, check_bar, down_tick, positive,
)
from nq_ict_priority_entry_examples_20260916.integration_example import PriorityLedgerExample
from .common import MajorExit, context_label
from .path_links import BuyPathBook
from .reentry import CampaignAttemptGate, SupportReactionBook
from .post_sell import PostSellPathBook, qualify_support
from .support_signals import IndependentSupportSignals, ObHistoryBook
from .structural_support import reaction_description, reaction_allowed, selected_support_role
from .structural_management import EntryScalePositionManager
from .continuation import ContinuationBuyBook
from .wave_management import CausalWaveBook, ThesisWaveManager
from .exit_reentry import ExitReasonReentry
from .selective_reentry import SelectiveReentry
from .episode_reentry import EpisodeScopedReentry
from .recovery_buy import RecoveryBuyBook


class CampaignLedger(PriorityLedgerExample):
    def __init__(self, rules, execution_cfg, campaign_cfg):
        super().__init__(rules, execution_cfg, name="major_campaign_proposal")
        self.campaign_cfg = campaign_cfg
        self.gate = CampaignAttemptGate(rules, self.priority, campaign_cfg)
        self.supports = SupportReactionBook(rules, campaign_cfg)
        self.ob_history = ObHistoryBook()
        self.independent = IndependentSupportSignals(rules, campaign_cfg)
        self.continuation = ContinuationBuyBook(rules, campaign_cfg)
        self.recovery = RecoveryBuyBook(rules, campaign_cfg)
        self.post_sell = PostSellPathBook(rules, campaign_cfg)
        self.paths = BuyPathBook(rules, campaign_cfg)
        self.waves = CausalWaveBook(rules, campaign_cfg)
        self.exit_reentry = (EpisodeScopedReentry(rules, campaign_cfg)
                             if campaign_cfg.exit_reentry_policy == "episode_scoped"
                             else SelectiveReentry(rules, campaign_cfg)
                             if campaign_cfg.exit_reentry_policy == "selective"
                             or campaign_cfg.failed_reaction_policy != "off"
                             else ExitReasonReentry(rules, campaign_cfg))
        if campaign_cfg.management_mode == "entry_scale":
            self.manager = EntryScalePositionManager(rules)
        elif campaign_cfg.management_mode == "thesis_wave":
            self.manager = ThesisWaveManager(rules, campaign_cfg, self.waves)
        self.campaign_events = []
        self.last_major_exit: MajorExit | None = None
        self.position_peaks = {}
        self.position_peak_indices = {}
        self.exit_peaks = {}
        self.exit_details = {}
        self.exit_reversal_ids = {}
        self.budget_scope_id = None
        self.support_after_index = None
        self.support_boundary = None
        self.market_phase = "initial"
        self.today_independent_confirmations = ()
        self.support_selection_rows = []
        self.entry_contexts = {}
        self.sell_decisions = []
        self.sell_execution_links = []
        self.position_evidence_rows = []
        self.seen_position_evidence = set()
        self.current_bar = None
        self.realized_net = 0.0
        self.equity_peak = 0.0
        self.daily_equity = []
        self.campaign_state = "WAIT_MAJOR_SUPPORT"

    def _close(self, at, index, price, reason):
        p = self.position
        if p is None:
            raise ValueError("no open campaign to close")
        if reason == "emergency_stop":
            reason = ("trailing_protection_stop" if self.live_stop > p.candidate.emergency_stop
                      else "initial_risk_stop")
        scope, _, support = self.gate.position_episodes[p.id]
        # Capture the already-known thesis before the inherited close mutates
        # position state. A gap exit must not use today's later high/low/close.
        frozen_exit = deepcopy(self.exit_details.get(p.id, {}))
        thesis = (self.manager.exit_context_for(p.id) if isinstance(self.manager, ThesisWaveManager)
                  else {})
        if isinstance(self.exit_reentry, (SelectiveReentry, EpisodeScopedReentry)) and isinstance(self.manager, EntryScalePositionManager):
            # Read only already-active protection; a pending next-close proposal
            # is not today's intrabar stop and must not be substituted for it.
            thesis = dict(active_protected_low=self.manager.active,
                          management_phase="entry_scale")
        super()._close(at, index, price, reason)
        self.exit_reentry.register_exit(p, index, at, reason, price, frozen_exit, thesis)
        self.realized_net += self.trades[-1]["net_pnl_usd"]
        if (reason.startswith(("campaign_major_top_", "campaign_double_top_"))
                or reason == "major_protected_low_close_failure"):
            peak = self.exit_peaks.get(p.id, self.position_peaks.get(p.id, p.entry_price))
            frozen = self.exit_details.pop(p.id)
            atr = frozen["atr"]
            self.last_major_exit = MajorExit(index, reason, peak, atr, p.candidate.episode_id,
                                            self.exit_reversal_ids.pop(p.id),
                                            peak_index=frozen["peak_index"], reference=frozen["reference"],
                                            signal_low=frozen["signal_low"], signal_index=frozen["signal_index"])
        else:
            self.exit_details.pop(p.id, None)
            self.exit_reversal_ids.pop(p.id, None)
        self.campaign_state = "WAIT_MAJOR_SUPPORT"
        self.campaign_events.append(dict(index=index, action="campaign_closed", campaign_id=p.id,
                                        reason=reason, episode_id=p.candidate.episode_id,
                                        budget_scope_id=scope, support_episode_id=support["group_id"],
                                        major_reentry_latch=self.last_major_exit))
        self.trades[-1]["entry_structural_context"] = deepcopy(self.entry_contexts.get(p.id))
        self.trades[-1]["exit_thesis"] = deepcopy(self.exit_reentry.context)
        decision = next(
            (deepcopy(d) for d in reversed(self.sell_decisions) if d["position_id"] == p.id), None)
        self.trades[-1]["sell_decision"] = decision
        link = None
        if decision is not None:
            # Keep the close-time decision immutable. A gap stop can supersede
            # the pending SELL, so its decision time is not a SELL fill time.
            link = dict(order_id=decision["order_id"], position_id=p.id,
                        decision_index=decision["index"], decision_at=decision["at"],
                        exit_index=index, exit_at=at, exit_price=price, executed_reason=reason,
                        sell_fill_at=at if reason == decision["reason"] else None,
                        status=("sell_filled" if reason == decision["reason"]
                                else "superseded_by_risk_exit"))
            self.sell_execution_links.append(link)
        self.trades[-1]["sell_execution_link"] = deepcopy(link)
        # The inherited ledger emitted its accounting event before these
        # descriptive fields were attached. Keep both exports consistent.
        for event in reversed(self.execution_rows):
            if event.get("action") == "position_closed":
                event.update(entry_structural_context=deepcopy(self.entry_contexts.get(p.id)),
                             sell_decision=deepcopy(decision), sell_execution_link=deepcopy(link),
                             exit_thesis=deepcopy(self.exit_reentry.context))
                break

    def on_open(self, bar):
        if (self.pending is not None and self.pending.side == "BUY"
                and self.pending.candidate.confirmation == "uptrend_recovery"):
            c = self.pending.candidate
            # Use the raw opening price, not today's later high/low/close or
            # adverse slippage to turn an invalid open into a valid reclaim.
            if bar.open <= c.next_open_min_price:
                self._cancel(bar.at, bar.index, "recovery_open_lost_frozen_reclaim_or_breakout")
        if (self.pending is not None and self.pending.side == "BUY"
                and self.campaign_cfg.continuation_risk_mode == "reaction"
                and self.pending.candidate.confirmation == "continuation_zone_reaction"):
            risk = self.entry_contexts.get(self.pending.id, {}).get("entry_risk", {})
            atr = risk.get("decision_atr")
            fill = bar.open + float(self.cfg.tick_size) * float(self.cfg.slippage_ticks)
            if atr is None or not positive(atr):
                self._cancel(bar.at, bar.index, "continuation_frozen_risk_atr_missing")
            elif fill - self.pending.candidate.emergency_stop > self.campaign_cfg.continuation_max_risk_atr * atr:
                self._cancel(bar.at, bar.index, "continuation_next_open_risk_exceeds_cap")
        if not self.campaign_cfg.enable_protected_intrabar_stop:
            self.pending_structural_stop = None
            if self.position is not None:
                self.live_stop = self.position.candidate.emergency_stop
        super().on_open(bar)

    def support_context(self, packet):
        """Read the market's scope; legacy modes may tighten it after an exit.

        This is called AFTER today's opening execution. An initial risk stop
        never creates a fresh budget. Modern modes keep the source-side clock
        independent of holdings, and leave exit-day restrictions to the gate.
        """
        old, reversal = self.last_major_exit, packet.major_reversal
        if self.campaign_cfg.cycle_mode in {"independent_advance", "major_top_boundary"}:
            context = packet.market_cycle
            if context is None:
                raise ValueError("missing fill-independent market context")
            # Fill clocks are enforced by the execution gate, not substituted
            # into a market observation. A flat and a long ledger see one map.
            self.market_phase, self.support_boundary = context.phase, context.boundary
            return context.scope_id, context.contact_after_index
        if self.campaign_cfg.cycle_mode == "structure_recovery":
            context = packet.market_cycle
            if context is None:
                raise ValueError("missing fill-independent market cycle")
            self.market_phase = context.phase
            self.support_boundary = context.boundary
            cutoff = context.contact_after_index
            # Execution tightens the contact clock, never creates/replenishes a
            # scope. Completed market cycles explicitly supersede older latches.
            if old is not None and old.index >= context.effective_index:
                cutoff = max(cutoff if cutoff is not None else -1, old.index)
                self.support_boundary = old
                self.market_phase = "declining_after_filled_exit"
            return context.scope_id, cutoff
        if old is not None and (reversal is None or old.index > reversal.index
                                or old.reversal_id == reversal.id):
            self.support_boundary = old
            return old.reversal_id, old.index
        if reversal is not None:
            self.support_boundary = reversal
            return reversal.id, reversal.index
        self.support_boundary = None
        episode = packet.base.signal.market.episode
        return f"market:{episode.id if episode else 'unclassified'}", None

    def reentry_allowed(self, offer, bar, episode):
        if self.support_after_index is None:
            return True, "initial_entry_no_major_reversal_boundary"
        if self.campaign_cfg.reentry_mode == "new_market_episode":
            old = self.last_major_exit
            if (episode is None or episode.started_index <= self.support_after_index
                    or (old is not None and episode.id == old.episode_id)):
                return False, "new_market_episode_after_major_exit_required"
            if episode.phase != "declining":
                return False, "new_episode_must_still_be_in_decline_for_support_reaction"
        allowed, reason, _ = self.supports.assess(offer, bar)
        return allowed, reason

    def rank_offer(self, offer, bar, previous):
        if self.recovery.is_offer(offer):
            # Preserve ordering of existing routes. Recovery is a fallback,
            # not another independent vote or a pyramiding opportunity.
            return (-2, 1, 0.0, 0, 0.0, offer.decision.key)
        obs = self.entry_observation(offer)
        if self.campaign_cfg.support_selection in {"structural_role", "structural_or_leg"}:
            role = selected_support_role(obs, self.campaign_cfg)
            r = reaction_description(offer, obs, bar, previous)
            return (role["grade"] if role else -1, int(offer.route == "confirmed"),
                    r.get("close_location") if r.get("close_location") is not None else -1.0,
                    -r.get("contact_to_recovery_bars", self.rules.reaction_lifetime),
                    -abs(r.get("distance_above_recovered_boundary_atr", 1e9)), offer.decision.key)
        if self.campaign_cfg.support_selection == "legacy":
            return (int(offer.route == "confirmed"), offer.decision.reference.origin_index, offer.decision.key)
        departure = obs.features.get("prior_departure_atr") if obs else None
        strength = (bar.close - obs.low) / obs.low_atr if obs else 0.0
        return (int(offer.route == "confirmed"), departure if departure is not None else -1.0,
                strength, offer.decision.key)

    def entry_observation(self, offer):
        if self.recovery.is_offer(offer):
            return None  # This model never claims a price-zone visit.
        if self.continuation.is_offer(offer):
            return self.continuation.observation(offer)
        return self.supports.active.get(offer.decision.zone.id)

    def assess_offer(self, offer, packet, previous, sells):
        """All gates on one close, including offers while holding or exiting.

        These are diagnostics without reservation or budget consumption. The
        next open is still unknown and cannot be pre-approved at this close.
        """
        s, bar = packet.base.signal, packet.base.signal.bar
        c = offer.decision
        obs = self.entry_observation(offer)
        continuation = self.continuation.is_offer(offer)
        recovery = self.recovery.is_offer(offer)
        family = "recovery" if recovery else "continuation" if continuation else "reversal"
        role = selected_support_role(obs, self.campaign_cfg)
        reaction = reaction_description(offer, obs, bar, previous)
        recovery_detail = {}
        checks = {}

        def record(name, passed, reason, **detail):
            checks[name] = dict(passed=bool(passed), reason=reason, status="evaluated", **detail)

        record("sell_priority", not sells, "SELL_has_priority" if sells else "no_SELL_conflict")
        exit_ok, exit_why, exit_detail = self.exit_reentry.assess(
            offer, obs, role, bar, self.waves.snapshot)
        record("exit_reason_reentry", exit_ok, exit_why, detail=deepcopy(exit_detail))
        if recovery:
            recovery_checks, role, recovery_detail = self.recovery.assess(offer, bar, self.exit_reentry)
            for name, result in recovery_checks.items():
                record(name, *result)
            for name in ("early_conditions", "reentry_context", "current_support_reaction", "support_provenance"):
                record(name, True, "recovery_uses_structure_not_ob_reaction")
                checks[name]["status"] = "not_applicable"
            checks["source_zone"]["status"] = "not_applicable"
            record("market_path", recovery_checks["recovery_structure"][0],
                   "actual_exit_reclaim_and_new_break_for_explicit_exit_scope")
            record("reaction_trigger", recovery_checks["recovery_structure"][0],
                   "accepted_structure_not_ob_touch")
            post_detail = dict(path="uptrend_recovery", exit_context_id=c.exit_context_id,
                               structure_id=c.episode_id, zone_retest_required=False,
                               recovery_structure_route=c.structure_route,
                               preserved_support_id=c.preserved_support_id)
            support_detail = dict(selected_role=deepcopy(role), support_route="recovery_structure",
                                  structure_qualification=deepcopy(recovery_detail.get("structure_qualification")),
                                  zone_preexisted_contact=None, contact_status="not_applicable")
            reaction = dict(status="not_applicable", reason="no_zone_contact_claimed")
        elif continuation:
            continuation_checks, role, obs = self.continuation.assess(offer, packet, previous)
            for name, result in continuation_checks.items():
                record(name, *result)
            record("early_conditions", True, "continuation_has_its_own_entry_conditions")
            record("reentry_context", True, "continuation_uses_current_structure_not_old_decline")
            record("market_path", continuation_checks["continuation_structure"][0],
                   "accepted_break_then_linked_zone_reaction")
            post_detail = dict(path="continuation_pullback", structure_id=c.episode_id,
                               requires_major_decline=False, requires_old_ob_seed=False,
                               exit_reentry=deepcopy(exit_detail))
            if self.campaign_cfg.continuation_context_mode == "wave_or_reclaim":
                context_ok = bool(self.waves.snapshot.get("valid") or
                                  (self.exit_reentry.snapshot.get("reclaimed") and exit_ok))
                record("continuation_market_context", context_ok,
                       "confirmed_daily_wave_or_resolved_exit_required",
                       wave=deepcopy(self.waves.snapshot), exit_reentry=deepcopy(exit_detail))
            support_detail = deepcopy(obs.features) if obs else {}
            support_detail.update(selected_role=deepcopy(role), support_route="continuation_support",
                                  zone_preexisted_contact=c.zone.available_index < c.touch_index)
        else:
            record("early_conditions", *self.paths.early_allowed(offer))
            record("reentry_context", *self.reentry_allowed(offer, bar, s.market.episode))
            fresh, why, _ = self.supports.assess(offer, bar)
            record("current_support_reaction", fresh, why)
            if self.campaign_cfg.exit_reentry_policy in {"selective", "exit_reason"} and self.exit_reentry.context is not None:
                allowed, why, post_detail = exit_ok, exit_why, deepcopy(exit_detail)
            else:
                allowed, why, post_detail = self.post_sell.assess(offer, obs, bar)
            record("market_path", allowed, why)
            allowed, why, support_detail = qualify_support(
                offer, obs, s.market, self.rules, self.campaign_cfg, post_detail)
            record("support_provenance", allowed, why)
            record("reaction_trigger", *reaction_allowed(reaction, self.campaign_cfg, role))
            view = next((v for v in packet.base.zones if v.zone.id == c.zone.id), None)
            source_ok = (view is not None and view.tradeable and obs is not None
                         and c.zone.available_index < obs.contact_index
                         and c.zone.available_at < bar.at)
            record("source_zone", source_ok,
                   "source_tradeable_and_precontact" if source_ok else "source_zone_unavailable")
        risk_ok = positive(c.emergency_stop) and c.emergency_stop < bar.close and positive(c.low_atr)
        record("close_risk_geometry", risk_ok, "close_risk_geometry_valid" if risk_ok else "invalid_close_risk_geometry",
               stop=c.emergency_stop, close=bar.close)
        entry_risk = (deepcopy(recovery_detail.get("risk", {})) if recovery else
                      deepcopy(obs.features.get("entry_risk", {})) if obs else {})
        if continuation and self.campaign_cfg.continuation_risk_mode == "reaction":
            atr = entry_risk.get("decision_atr")
            width_ok = bool(atr is not None and positive(atr)
                            and entry_risk.get("decision_index") == bar.index
                            and 0 < bar.close - c.emergency_stop <= self.campaign_cfg.continuation_max_risk_atr * atr)
            record("continuation_risk_width", width_ok, "reaction_risk_must_fit_frozen_atr_cap",
                   risk=deepcopy(entry_risk))
        htf_filter = self.campaign_cfg.buy_htf_filter
        if htf_filter != "none":
            timeframe, measure, _direction = htf_filter.split("_")
            timeframe = timeframe.lower()
            completion = getattr(s.htf, f"{timeframe}_completed_at")
            state = getattr(s.htf, f"{timeframe}_{measure}")
            completed_as_of_decision = completion is not None and completion <= bar.at
            passed = completed_as_of_decision and state == "up"
            record("buy_htf_filter", passed,
                   "latest_completed_HTF_state_up" if passed else "HTF_state_not_up_or_not_completed",
                   filter=htf_filter, state=state,
                   context_completed_at=completion, decision_at=bar.at,
                   completed_as_of_decision=completed_as_of_decision)
        budget = self.gate.diagnose(c, bar, s.pivots, self.position is not None, self.pending is not None)
        record("execution_budget", budget["allowed"], budget["reason"], components=budget["checks"])
        # Deliberately not part of today's allow/deny boolean.
        next_open = dict(status="pending_future_observation", passed=None,
                         rule="next_observed_open_plus_slippage_must_exceed_initial_stop",
                         max_gap_or_max_risk_cap="not_implemented_in_inherited_execution")
        if recovery:
            next_open.update(rule="raw_next_open_above_frozen_reclaim_and_breakout_then_inherited_stop_check",
                             minimum_raw_open_exclusive=c.next_open_min_price,
                             checkpoint_clock="frozen_at_decision_close")
        if continuation and self.campaign_cfg.continuation_risk_mode == "reaction":
            next_open.update(max_gap_or_max_risk_cap="continuation_risk_width_cap",
                             max_risk_atr=self.campaign_cfg.continuation_max_risk_atr,
                             atr_clock="frozen_decision_prior_atr", maximum_gap_cap=None)
        failures = [dict(gate=k, reason=v["reason"]) for k, v in checks.items() if not v["passed"]]
        allowed = not failures
        why = failures[0]["reason"] if failures else budget["reason"]
        row = dict(index=bar.index, key=c.key, selected_eligibility=allowed, reason=why,
                   rank=self.rank_offer(offer, bar, previous), post_sell=post_detail,
                   support_features=support_detail, selected_role=deepcopy(role),
                   support_route=role["kind"] if role else None,
                   reaction=reaction, entry_risk=entry_risk,
                   support_episode_id=obs.group_id if obs else None,
                   checks=checks, failed_conditions=failures, budget_diagnostic=budget,
                   next_open_execution=next_open, selected=False, probability=None,
                   entry_family=family,
                   probability_status="not_estimated", all_close_gates_evaluated=True)
        self.support_selection_rows.append(row)
        self.entry_audit.append(dict(
            index=bar.index, action="entry_assessed", key=c.key, setup_id=c.setup_id,
            entry_family=family,
            route=offer.route, kind=offer.kind, allowed=allowed, reason=why,
            market_episode_id=c.episode_id, budget_scope_id=self.budget_scope_id,
            support=self.gate.contexts[c.key][1], context=context_label(s.htf),
            checks=deepcopy(checks), failed_conditions=deepcopy(failures),
            next_open_execution=next_open, support_route=row["support_route"],
            post_sell=post_detail, support_features=support_detail, reaction=reaction,
            path=(self.recovery.describe(offer) if recovery else
                  self.continuation.describe(offer) if continuation else
                  self.independent.describe(c.setup_id)
                  if self.campaign_cfg.support_candidate_mode == "independent_ob_reactions"
                  else self.paths.describe(c.setup_id))))
        return offer, allowed, why, row, obs

    def feed(self, packet):
        s, bar = packet.base.signal, packet.base.signal.bar
        check_bar(bar, self.tracker.previous)
        self.current_bar = bar
        event_start = len(self.execution_rows)
        self.on_open(bar)  # Existing next-open orders, gap stops, then hard intrabar stop.
        self.waves.on_close(bar, s.pivots, s.sells)
        if isinstance(self.exit_reentry, (SelectiveReentry, EpisodeScopedReentry)):
            self.exit_reentry.supply_structures(packet.major_structures)
            self.exit_reentry.on_close(bar, s.sells, packet.top_details, self.position)
        else:
            self.exit_reentry.on_close(bar, s.sells, packet.top_details)
        for e in self.execution_rows[event_start:]:
            if e.get("action") == "buy_fill":
                self.campaign_events.append(dict(index=bar.index, action="campaign_opened",
                                                campaign_id=e["order_id"], fill=e["price"]))
        # Keep market context independent of fills. In revision 0.7 default
        # settings, scope changes and counters do not grant/revoke BUY quotas.
        self.budget_scope_id, self.support_after_index = self.support_context(packet)
        history_features = self.ob_history.on_close(bar, packet.base.zones)
        for zid, structural in packet.support_roles.items():
            if zid in history_features:
                history_features[zid]["structural_support"] = deepcopy(structural)
        self.post_sell.on_close(bar, self.budget_scope_id, self.support_after_index, self.support_boundary,
                                self.market_phase)
        self.supports.on_close(bar, packet.base.zones, self.budget_scope_id, self.support_after_index,
                               history_features)
        self.paths.on_close(packet)
        previous = self.tracker.previous
        offers = list(self.tracker.on_close(bar, s.market, s.pivots, packet.base.zones, s.htf,
                                            s.early_buys, s.confirmed_buys, packet.base.reactions))
        if self.campaign_cfg.support_candidate_mode == "independent_ob_reactions":
            # Legacy outputs remain inspectable; they do not bypass the new
            # observation population or become a future-confirmed whitelist.
            start = len(self.independent.confirmed_rows)
            offers = list(self.independent.on_close(bar, previous, s.market, s.htf,
                          packet.base.zones, self.supports, s.pivots, packet.tactical_zones, packet.evidence))
            self.today_independent_confirmations = tuple(self.independent.confirmed_rows[start:])
        else:
            self.today_independent_confirmations = ()
            offers.extend(self.supports.local_offers(bar, previous, s.market, s.htf,
                                                    self.tracker.watches, packet.base.zones, offers))
            mode = self.campaign_cfg.confirmed_entry_mode
            if mode == "retest_only":
                offers = [o for o in offers if o.route == "early"]
            if mode != "structure_only":
                offers.extend(self.paths.retest_offers(packet))
        offers.extend(self.continuation.on_close(packet, previous))
        if isinstance(self.exit_reentry, EpisodeScopedReentry):
            self.exit_reentry.supply_continuation(self.continuation.structure)
            offers.extend(self.recovery.on_close(packet, self.continuation, self.exit_reentry))
        self.eligible_setup_ids.update(c.setup_id for c in s.early_buys if c.setup_id)
        self.eligible_setup_ids.update(o.decision.setup_id for o in offers if o.decision.setup_id)
        for offer in offers:
            self.gate.register_offer(offer, self.budget_scope_id,
                                     self.entry_observation(offer))

        if self.campaign_cfg.management_mode == "entry_scale":
            self.manager.supply_structures(packet.major_structures)
        management = self.manager.on_close(bar, s.pivots, self.position)
        sells = tuple(s.sells) + ((management.exit_signal,) if management.exit_signal else ())
        # Assess every offer before choosing/reserving, even when a different
        # gate would fail first or this ledger already holds a position.
        ordered = sorted(offers, key=lambda o: self.rank_offer(o, bar, previous), reverse=True)
        assessments = [self.assess_offer(o, packet, previous, sells) for o in ordered]
        intent, selected = None, None
        reason = "hold_major_wave" if self.position else "wait_reversal_continuation_or_recovery"
        if self.position is not None:
            p = self.position
            # Post-entry FVG/MSS observations describe subsequent progress. They
            # never retroactively determine whether this BUY was allowed.
            for e in packet.evidence:
                token = (p.id, e.id)
                if (token not in self.seen_position_evidence and p.entry_index <= e.index <= bar.index
                        and e.high >= p.candidate.low and e.kind in {"structure_break", "zone_formed"}):
                    self.seen_position_evidence.add(token)
                    self.position_evidence_rows.append(dict(index=bar.index, position_id=p.id, entry_index=p.entry_index,
                                                           evidence=e, prior_entry_rewritten=False))
            if p.id not in self.position_peaks or bar.high > self.position_peaks[p.id]:
                self.position_peaks[p.id] = max(p.entry_price, bar.high)
                self.position_peak_indices[p.id] = bar.index
            parent_path = self.paths.paths.get(p.candidate.setup_id)
            structure_seen = ((parent_path is not None and parent_path.confirmed is not None)
                              or p.candidate.setup_id in self.independent.confirmed_setup_ids
                              or p.candidate.confirmation in {"continuation_zone_reaction", "uptrend_recovery"})
            if packet.top_warnings:
                self.campaign_state = "TOP_WATCH"
            elif structure_seen:
                # A new market decline does not turn an old holding into a new early BUY.
                self.campaign_state = ("LONG_STRUCTURE_WATCH" if s.market.episode
                                       and s.market.episode.phase == "declining" else "LONG_ADVANCING")
            else:
                self.campaign_state = "EARLY_LONG"
            if isinstance(self.manager, ThesisWaveManager):
                self.campaign_state = ("LONG_ADVANCING" if self.manager.phase == "wave" else "EARLY_LONG")
                if self.manager.below_count or self.manager.reaction_below_count:
                    self.campaign_state = "LONG_STRUCTURE_WATCH"
                elif packet.top_warnings:
                    self.campaign_state = "TOP_WATCH"
            eligible_sells = [x for x in sells if x.index >= p.entry_index]
            if eligible_sells:
                if len(s.sells) > 1:
                    raise ValueError("execution must receive a single canonical top SELL")
                chosen = eligible_sells[0]  # Canonical top exit, otherwise the structural protective exit.
                intent = OrderIntent(self._new_id(), "SELL", bar.at, bar.index,
                                     chosen.reason, position_id=p.id)
                detail = next((d for d in packet.top_details if d["top_id"] == chosen.source_id), None)
                self.exit_peaks[p.id] = detail["peak"] if detail else self.position_peaks[p.id]
                self.exit_details[p.id] = dict(
                    peak_index=detail["peak_index"] if detail else self.position_peak_indices[p.id],
                    reference=chosen.reference, atr=bar.atr_prior,
                    signal_low=bar.low, signal_index=bar.index, reason=chosen.reason)
                if (isinstance(self.manager, ThesisWaveManager)
                        and chosen == management.exit_signal and self.manager.exit_detail):
                    self.exit_details[p.id].update(deepcopy(self.manager.exit_detail))
                # A protective exit without a source-side top creates one
                # fallback boundary. Later exits within that decline reuse it.
                self.exit_reversal_ids[p.id] = self.budget_scope_id
                self.sell_decisions.append(dict(
                    index=bar.index, at=bar.at, order_id=intent.id, position_id=p.id,
                    source_id=chosen.source_id, reason=chosen.reason,
                    representative_selection=deepcopy(packet.sell_selection),
                    top_detail=deepcopy(detail), frozen_exit=deepcopy(self.exit_details[p.id]),
                    peak=self.exit_peaks[p.id], budget_scope_id=self.budget_scope_id,
                    fill_at=None, fill_status="next_observed_open_pending"))
                self.campaign_state, reason = "EXIT_PENDING", "major_reversal_exit_next_open"
        elif sells:
            reason = "sell_evidence_flat_no_short"
        elif self.pending is not None:
            raise ValueError("unresolved previous order")
        else:
            self.campaign_state = "WAIT_MAJOR_SUPPORT"
            # Assess all same-close offers before reserving one order. One set
            # of overlapping bands does not create additional trades or votes.
            chosen_entry = next((a for a in assessments if a[1]), None)
            if chosen_entry:
                offer, _, why, selected_row, obs = chosen_entry
                selected = offer.decision.key
                selected_row["selected"] = True
                intent = OrderIntent(self._new_id(), "BUY", bar.at, bar.index, why, offer.decision)
                role = deepcopy(selected_row["selected_role"])
                self.entry_contexts[intent.id] = dict(
                    role=deepcopy(role), decision_index=bar.index, source_zone=offer.decision.zone,
                    entry_family=selected_row["entry_family"],
                    recovery_structure_qualification=deepcopy(
                        selected_row["support_features"].get("structure_qualification")),
                    support_route=selected_row["support_route"],
                    contact_index=obs.contact_index if obs else offer.decision.touch_index,
                    close_gate_snapshot=deepcopy(selected_row["checks"]),
                    next_open_execution=deepcopy(selected_row["next_open_execution"]),
                    reaction=deepcopy(selected_row["reaction"]),
                    entry_risk=deepcopy(selected_row["entry_risk"]),
                    same_reaction_candidates=[dict(key=a[0].decision.key,
                                                   zone=a[0].decision.zone, allowed=a[1], reason=a[2])
                                              for a in assessments if a[4] is not None and obs is not None
                                              and a[4].group_id == obs.group_id],
                    stop=offer.decision.emergency_stop, budget_scope_id=self.budget_scope_id,
                    ranking_policy=self.campaign_cfg.support_selection,
                    numerical_thresholds="provisional_not_fitted")
                if self.campaign_cfg.management_mode == "entry_scale":
                    if self.recovery.is_offer(offer):
                        self.manager.register_recovery_entry(intent.id, offer.decision, role)
                    else:
                        self.manager.register_entry(intent.id, offer.decision, role)
                elif isinstance(self.manager, ThesisWaveManager):
                    self.manager.register_entry(intent.id, offer.decision, role, obs, previous, bar)
                    self.entry_contexts[intent.id]["entry_thesis"] = self.manager.thesis_for(intent.id)
                if isinstance(self.exit_reentry, (SelectiveReentry, EpisodeScopedReentry)):
                    self.exit_reentry.register_entry(intent.id, offer.decision, role, obs, previous, bar)
                    self.entry_contexts[intent.id]["reentry_entry_thesis"] = deepcopy(
                        self.exit_reentry.entries[intent.id])
                self.gate.reserve(intent.id, offer.decision)
                self.campaign_state, reason = "ENTRY_PENDING", f"{offer.kind}_next_open"
        for offer in offers:
            self.entry_audit.append(dict(index=bar.index, action="offer_disposition", key=offer.decision.key,
                                        selected=offer.decision.key == selected,
                                        holding=self.position is not None, sell_priority=bool(sells),
                                        replay_later_without_new_evidence=False))
        if (management.proposed_protected_low is not None and self.position is not None
                and self.campaign_cfg.enable_protected_intrabar_stop):
            low = management.proposed_protected_low.pivot
            self.pending_structural_stop = (self.position.id,
                down_tick(low.price - self.rules.emergency_buffer_atr * bar.atr_prior, self.rules.tick_size))
        # Otherwise the manager's pending level is kept only for future CLOSE tests.
        self.pending = intent
        self.plan_rows.append(dict(datetime=bar.at, variant=self.name, reason=reason, intent=intent,
                                   management=management, top_details=packet.top_details,
                                   sell_detector=self.campaign_cfg.sell_detector,
                                   post_sell_context=dict(self.post_sell.snapshot),
                                   market_cycle=packet.market_cycle,
                                   representative_sell=deepcopy(packet.sell_selection),
                                   attempt_allocation=self.campaign_cfg.attempt_allocation,
                                   campaign_loss_limit_mode=self.campaign_cfg.campaign_loss_limit_mode,
                                   continuation_context=deepcopy(self.continuation.snapshot),
                                   recovery_context=deepcopy(self.recovery.snapshot),
                                   wave_context=deepcopy(self.waves.snapshot),
                                   exit_reentry_context=deepcopy(self.exit_reentry.snapshot),
                                   failed_reaction_policy=self.campaign_cfg.failed_reaction_policy,
                                   management_phase=getattr(self.manager, "phase", None),
                                   budget_scope_id=self.budget_scope_id,
                                   offered_count=len(offers), state=self.campaign_state))
        self.state_rows.append(dict(datetime=bar.at, index=bar.index, state=self.campaign_state,
                                    position_id=self.position.id if self.position else None,
                                    live_stop=self.live_stop, pending_order_id=intent.id if intent else None,
                                    major_exit_latch=self.last_major_exit,
                                    episode_id=s.market.episode.id if s.market.episode else None,
                                    major_reversal_id=packet.major_reversal.id if packet.major_reversal else None,
                                    budget_scope_id=self.budget_scope_id,
                                    support_after_index=self.support_after_index,
                                    market_phase=self.market_phase,
                                    continuation_phase=self.continuation.snapshot.get("phase"),
                                    recovery_phase=self.recovery.snapshot.get("phase"),
                                    wave_context=deepcopy(self.waves.snapshot),
                                    exit_reentry_context=deepcopy(self.exit_reentry.snapshot),
                                    management_phase=getattr(self.manager, "phase", None),
                                    continuation_structure_id=(self.continuation.structure["id"]
                                                               if self.continuation.structure else None),
                                    weekly_trend=s.htf.weekly_trend, monthly_trend=s.htf.monthly_trend,
                                    weekly_bias=s.htf.weekly_bias, monthly_bias=s.htf.monthly_bias,
                                    context=context_label(s.htf), top_warnings=packet.top_warnings))
        # Cash PnL for a fixed number of contracts, not account-return percentages.
        # An open position pays entry commission only; realized trades paid both.
        floating = 0.0
        if self.position:
            p = self.position
            floating = ((bar.close - p.entry_price) * p.quantity * p.point_value
                        - p.quantity * float(self.cfg.commission_per_contract_side_usd))
        equity = self.realized_net + floating
        self.equity_peak = max(self.equity_peak, equity)
        self.daily_equity.append(dict(datetime=bar.at, realized_net_usd=self.realized_net,
                                      open_pnl_after_entry_fee_usd=floating, mtm_pnl_usd=equity,
                                      close_to_close_drawdown_usd=equity - self.equity_peak,
                                      intraday_drawdown_status="not_measured"))
        self.previous_at = bar.at

    def final_state(self):
        result = super().final_state()
        # The inherited key describes market episodes; budgets now use a
        # different identity. Do not mislabel them in the exported snapshot.
        result.pop("episode_budgets", None)
        result.update(verification_status="NOT_VALIDATED", campaign_state=self.campaign_state,
                      campaign_budgets=self.gate.budgets,
                      budget_scope_id=self.budget_scope_id,
                      support_after_index=self.support_after_index,
                      active_support_reactions=self.supports.active,
                      sell_detector=self.campaign_cfg.sell_detector,
                      attempt_allocation=self.campaign_cfg.attempt_allocation,
                      campaign_loss_limit_mode=self.campaign_cfg.campaign_loss_limit_mode,
                      reentry_evidence_mode=self.campaign_cfg.reentry_evidence_mode,
                      failed_reaction_policy=self.campaign_cfg.failed_reaction_policy,
                      continuation_context=deepcopy(self.continuation.snapshot),
                      recovery_context=deepcopy(self.recovery.snapshot),
                      wave_context=deepcopy(self.waves.snapshot),
                      exit_reentry_context=deepcopy(self.exit_reentry.snapshot),
                      position_management=deepcopy(getattr(self.manager, "snapshot", {})),
                      active_continuation_reactions=self.continuation.active,
                      post_sell_context=dict(self.post_sell.snapshot),
                      independent_support_candidates=list(self.independent.active.values()),
                      major_exit_latch=self.last_major_exit,
                      market_phase=self.market_phase,
                      entry_contexts=self.entry_contexts,
                      active_paths=[sid for sid, p in self.paths.paths.items() if not p.retired],
                      end_of_input_status="right_censored_not_forced_exit",
                      probability_status="not_estimated")
        return result

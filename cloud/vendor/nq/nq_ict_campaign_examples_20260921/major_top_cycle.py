"""Revision 0.6: a new major parent top owns the next decline's risk scope.

AUTHORED_NOT_EXECUTED_OR_VALIDATED. No new detector or fitted threshold.
The existing major-leg/parent selector provides the evidence. The separate
upward-phase observer remains descriptive/entry context, NOT a budget gate.
"""
from __future__ import annotations

from copy import deepcopy

from .cycle_recovery import IndependentAdvanceBook


class MajorTopBoundaryBook(IndependentAdvanceBook):
    """Change only risk-scope renewal; retain the existing market phase logic.

For a later cycle, the qualified impulse low must be AFTER the previous
budget-owning H2, and the new H1 AFTER that top's SELL decision. Thus a child
inside the old top, a renamed old pair, a stop or a new OB cannot renew risk.
All anchors are frozen at the observed SELL close, never revised afterwards.
"""

    def __init__(self, rules, cfg):
        super().__init__(rules, cfg)
        self.budget_parent_top = None
        self.seen_parent_groups = set()
        self.seen_parent_clusters = set()

    def _budget_reset_decision(self, bar, reversal, sells, top_details):
        # Only the canonical source SELL can start a scope. A position-manager
        # exit or an old reversal reused on a later day is not an input here.
        signal = sells[0] if len(sells) == 1 else None
        detail = next((d for d in top_details if signal is not None
                       and d.get("top_id") == signal.source_id), None)
        parent = detail.get("parent_setup") if detail else None
        q = parent.get("qualification", {}) if parent else {}
        high = parent.get("h1") if parent else None
        neck = parent.get("neckline") if parent else None
        low = q.get("impulse_low")
        local = detail.get("local_trigger", {}) if detail else {}
        peak_index = parent.get("peak_index") if parent else None
        group_id = detail.get("parent_group_id") if detail else None
        parent_id = detail.get("parent_top_id") if detail else None
        cluster_id = parent.get("cluster_id") if parent else None
        previous = deepcopy(self.budget_parent_top)

        result = dict(
            policy="major_top_boundary", renew=False, uses_independent_advance=False,
            reason="parent_top_evidence_missing", previous_budget_parent=previous,
            previous_scope_id=self.context.scope_id, parent_top_id=parent_id,
            parent_group_id=group_id, parent_cluster_id=cluster_id,
            independent_advance_required=False, checks={}, failed_conditions=[])
        if (signal is None or detail is None or parent is None
                or high is None or neck is None or low is None or peak_index is None
                or group_id is None or parent_id is None or cluster_id is None):
            result["failed_conditions"] = ["parent_top_evidence_missing"]
            return result

        checks = dict(
            current_canonical_sell=(signal.index == reversal.index == bar.index
                                    and signal.at == reversal.at == bar.at
                                    and signal.source_id in reversal.source_ids),
            matching_parent=(parent.get("top_id") == parent_id == reversal.parent_top_id
                             and high.origin_index == reversal.parent_h1_index
                             and peak_index == reversal.peak_index),
            parent_selected_now=(detail.get("parent_selection_known_index") == bar.index
                                 and parent.get("index") == bar.index),
            existing_major_leg=(q.get("mode") == "major_leg" and q.get("qualified") is True),
            qualification_available=(q.get("known_index") is not None
                                     and q["known_index"] <= bar.index),
            anchors_available=(low.available_index < high.origin_index
                               and high.available_index < bar.index
                               and neck.available_index < bar.index),
            ordered_parent=(low.origin_index < high.origin_index < neck.origin_index
                            < peak_index < bar.index),
            local_decline_confirmed=(local.get("source_id") == signal.source_id
                                     and local.get("decision_index") == bar.index
                                     and signal.reference.available_index < bar.index),
            parent_not_suppressed=not parent.get("suppressed", False),
            new_parent_identity=parent_id not in self.seen_parent_tops,
            new_parent_group=group_id not in self.seen_parent_groups,
            new_parent_cluster=cluster_id not in self.seen_parent_clusters,
            impulse_after_previous_top=(previous is None
                                       or low.origin_index > previous["peak_index"]),
            h1_after_previous_sell=(previous is None
                                    or high.origin_index > previous["decision_index"]),
        )
        # Record the group when observed, even if rejected: a later discovery
        # cannot retroactively promote this same SELL into a budget reset.
        self.seen_parent_groups.add(group_id)
        self.seen_parent_clusters.add(cluster_id)
        failed = [name for name, passed in checks.items() if not passed]
        result.update(checks=checks, failed_conditions=failed)
        frozen = dict(
            parent_top_id=parent_id, group_id=group_id, cluster_id=cluster_id,
            source_id=signal.source_id, reversal_id=reversal.id,
            decision_index=bar.index, decision_at=bar.at,
            impulse_low_index=low.origin_index, h1_index=high.origin_index,
            neckline_index=neck.origin_index, peak_index=peak_index,
            qualification=deepcopy(q), parent_setup=deepcopy(parent),
            local_trigger=deepcopy(local),
            confirmation_scope=detail.get("confirmation_scope"))
        result["observed_parent"] = frozen
        if failed:
            result["reason"] = "same_budget:" + failed[0]
            return result

        # Portfolio counters stay in CampaignAttemptGate, attached to each
        # filled trade's original scope. Only the source-side identity changes.
        self.budget_parent_top = deepcopy(frozen)
        result.update(renew=True, reason=("first_qualified_major_top" if previous is None
                                         else "new_major_top_after_separate_rally"))
        return result

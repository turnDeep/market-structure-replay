"""One representative SELL shared by execution, market state and re-entry.

Relationship sets are frozen at observation. Original candidate IDs and every
raw diagnostic remain intact; independent setups on the same day remain distinct.
"""
from __future__ import annotations

from copy import deepcopy


class SellRepresentativeBook:
    def __init__(self, cfg):
        self.cfg = cfg
        self.groups = {}
        self.events = []
        self.raw_signals = []
        self.snapshot = {}
        self.selected_details = ()

    def _related(self, left, right):
        if left["source_id"] == right["source_id"]:
            return True
        # Same high-area detector identity, or same known rally AND nearby H2.
        if left.get("area_id") and left["area_id"] == right.get("area_id"):
            return True
        if not left.get("leg_id") or left["leg_id"] != right.get("leg_id"):
            return False
        if left.get("peak_index") is None or right.get("peak_index") is None:
            return False
        return (abs(left["peak_index"] - right["peak_index"]) <= self.cfg.top_cluster_bars
                and abs(left["peak"] - right["peak"]) <= self.cfg.top_cluster_atr * left["atr"])

    def select(self, bar, signals, details):
        self.selected_details = tuple(details)
        if self.cfg.sell_representation == "parent_and_trigger":
            return self._select_parent_and_trigger(bar, signals, details)
        by_id = {d["top_id"]: d for d in details}
        ranked = []
        for s in signals:
            d = by_id.get(s.source_id)
            if d is None or d.get("peak_index") is None:
                raise ValueError("SELL representative requires source peak details")
            q = d.get("qualification") or {}
            low = q.get("impulse_low")
            high = d.get("h1")
            item = dict(source_id=s.source_id, peak=d["peak"], peak_index=d["peak_index"],
                        atr=high.atr_at_origin if high else s.reference.atr_at_origin,
                        leg_id=low.id if low else None, area_id=d.get("area_id"),
                        rank=(int(q.get("qualified", False)), q.get("impulse_bars") or 0,
                              q.get("impulse_atr") or 0.0,
                              (high.price - d["neckline"].price) if high and d.get("neckline") else 0.0))
            ranked.append((s, item))
            self.raw_signals.append(s)
        # Structure rank first; identifier only resolves exact ties.
        ranked.sort(key=lambda pair: (tuple(-v for v in pair[1]["rank"]), pair[0].source_id))
        representatives, memberships = [], []
        for signal, item in ranked:
            prior = next((g for g in self.groups.values() if self._related(g, item)), None)
            if prior is not None:
                memberships.append(dict(source_id=signal.source_id, representative_source_id=prior["source_id"],
                                        group_id=prior["group_id"], relation="same_observed_top"))
                continue
            group = dict(item, group_id=f"sell-group:{bar.index}:{signal.source_id}", known_index=bar.index)
            self.groups[group["group_id"]] = group
            representatives.append(signal)
            memberships.append(dict(source_id=signal.source_id, representative_source_id=signal.source_id,
                                    group_id=group["group_id"], relation="new_observed_top"))
        # Only one market decision on this close. Others are logged, not replayed
        # on a later date without fresh evidence.
        selected = representatives[:1]
        self.snapshot = dict(index=bar.index,
                             selected_source_id=selected[0].source_id if selected else None,
                             raw_source_ids=[s.source_id for s in signals], memberships=memberships,
                             independent_same_day_not_selected=[s.source_id for s in representatives[1:]],
                             policy="structural_rank_then_stable_id", retroactive_merge=False)
        if signals:
            self.events.append(dict(action="sell_representative_selected", **self.snapshot))
        return tuple(selected)

    def _contains(self, parent, child):
        if parent["top_id"] == child["top_id"]:
            return True
        ph, ch = parent.get("h1"), child.get("h1")
        pn, cn = parent.get("neckline"), child.get("neckline")
        if not all((ph, ch, pn, cn)) or not parent.get("qualification", {}).get("qualified"):
            return False
        # A true containing range with the same observed second-top area.
        # An older unrelated high or an invalidated pair is not a parent.
        return (ph.origin_index < ch.origin_index
                and pn.price <= cn.price
                and abs(parent["peak_index"] - child["peak_index"]) <= self.cfg.top_cluster_bars
                and abs(parent["peak"] - child["peak"]) <= self.cfg.top_cluster_atr * ph.atr_at_origin
                and ph.price - pn.price >= ch.price - cn.price
                and not parent.get("suppressed", False))

    @staticmethod
    def _parent_rank(detail):
        h, n = detail.get("h1"), detail.get("neckline")
        return ((h.price - n.price) / h.atr_at_origin if h and n else 0.0,
                detail["peak_index"] - h.origin_index if h else 0,
                h.radius if h else 0)

    def _select_parent_and_trigger(self, bar, signals, details):
        by_id = {d["top_id"]: d for d in details}
        proposals = []
        for signal in signals:
            child = by_id.get(signal.source_id)
            if child is None or child.get("peak_index") is None:
                raise ValueError("SELL requires a current observed setup detail")
            parents = [d for d in details if self._contains(d, child)]
            parent = sorted(parents, key=lambda d: (tuple(-x for x in self._parent_rank(d)),
                                                    d["top_id"]))[0]
            local = dict(source_id=signal.source_id, reference=signal.reference,
                         decision_index=bar.index, first_breaks=deepcopy(child.get("first_breaks", {})),
                         h1=child.get("h1"), neckline=child.get("neckline"),
                         peak=child["peak"], peak_index=child["peak_index"],
                         levels_known_index=child.get("levels_known_index"))
            frozen = deepcopy(parent)
            frozen.update(top_id=signal.source_id, parent_top_id=parent["top_id"],
                          local_trigger=local, parent_setup=deepcopy(parent),
                          parent_selection_known_index=bar.index,
                          parent_neckline_confirmed=bool(parent.get("route_ready", {}).get("major")),
                          confirmation_scope=("parent_neckline_failure" if
                              parent.get("route_ready", {}).get("major") else "local_decline_inside_parent"),
                          sell_decided_at=bar.at, fill_at=None, fill_status="execution_log_only")
            proposals.append((signal, frozen))
            self.raw_signals.append(signal)
        # Range depth first, span second; ONLY valid containing ranges compete.
        # Do not rank by the duration of the impulse BEFORE H1.
        proposals.sort(key=lambda pair: (tuple(-x for x in self._parent_rank(pair[1])), pair[0].source_id))
        selected = None
        memberships = []
        for signal, d in proposals:
            parent_id = d["parent_top_id"]
            high = d.get("h1")
            prior = next((g for g in self.groups.values()
                          if g.get("parent_top_id") == parent_id
                          or (high is not None and g.get("h1_index") is not None
                              and abs(d["peak_index"] - g["peak_index"]) <= self.cfg.top_cluster_bars
                              and abs(d["peak"] - g["peak"]) <= self.cfg.top_cluster_atr * high.atr_at_origin
                              and (g["h1_index"] <= high.origin_index <= g["peak_index"]
                                   or high.origin_index <= g["h1_index"] <= d["peak_index"]))), None)
            if prior is not None:
                memberships.append(dict(source_id=signal.source_id, parent_top_id=parent_id,
                                        representative_source_id=prior["source_id"],
                                        group_id=prior["group_id"], relation="same_observed_parent_top"))
                continue
            if selected is not None:
                memberships.append(dict(source_id=signal.source_id, parent_top_id=parent_id,
                                        relation="independent_same_day_not_selected"))
                continue
            group = dict(group_id=f"parent-top:{bar.index}:{parent_id}", parent_top_id=parent_id,
                         source_id=signal.source_id, h1_index=high.origin_index if high else None,
                         peak=d["peak"], peak_index=d["peak_index"], known_index=bar.index)
            self.groups[group["group_id"]] = group
            d["parent_group_id"] = group["group_id"]
            selected = (signal, d)
            memberships.append(dict(source_id=signal.source_id, parent_top_id=parent_id,
                                    representative_source_id=signal.source_id, group_id=group["group_id"],
                                    relation="parent_with_local_trigger_selected"))
        self.snapshot = dict(
            index=bar.index, selected_source_id=selected[0].source_id if selected else None,
            selected_parent_top_id=selected[1]["parent_top_id"] if selected else None,
            parent_setup=deepcopy(selected[1]["parent_setup"]) if selected else None,
            local_trigger=deepcopy(selected[1]["local_trigger"]) if selected else None,
            raw_source_ids=[s.source_id for s in signals], memberships=memberships,
            policy="valid_containing_range_and_separate_local_trigger", retroactive_merge=False,
            confirmation_scope=selected[1]["confirmation_scope"] if selected else None)
        if selected:
            sid, selected_detail = selected[0].source_id, selected[1]
            self.selected_details = tuple(selected_detail if d["top_id"] == sid else d for d in details)
        if signals:
            self.events.append(dict(action="parent_and_trigger_selected", **deepcopy(self.snapshot)))
        return (selected[0],) if selected else ()

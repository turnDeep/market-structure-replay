"""Future opt-in local replay entry point. AUTHORED, NOT EXECUTED/VALIDATED.

No imports trigger a replay. No market download, broker order or optimizer.
An explicit completed-date cutoff, session calendar and output path are needed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from . import IMPLEMENTATION_VERSION


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(__file__).resolve().parent


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.json")
    parser.add_argument("--base-config", type=Path, default=ROOT / "nq_ict_retest_trade_revision_config_20260915.json")
    parser.add_argument("--prices", type=Path, default=ROOT / "nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv")
    parser.add_argument("--objects", type=Path, default=ROOT / "nq_ict_detection_20260912_v09/objects_v09.csv")
    parser.add_argument("--contexts", type=Path, default=ROOT / "nq_ict_evaluation_20260912_v09/contexts_v09.csv")
    parser.add_argument("--calendar-json", type=Path)
    parser.add_argument("--as-of", help="caller-confirmed completed session-date cutoff")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sell-detector", choices=("major_area", "double_top_major", "double_top_early"),
                        help="explicit alternative; does not tune a model from results")
    parser.add_argument("--reentry-mode", choices=("fresh_support_same_episode", "new_market_episode"),
                        help="budget handling stays common; changes only the market-episode restriction")
    parser.add_argument("--attempt-allocation", choices=("unlimited", "shared_total", "legacy_per_route"))
    parser.add_argument("--campaign-loss-limit-mode", choices=("observe_only", "enforce"))
    parser.add_argument("--reentry-evidence-mode", choices=("fresh_price_reaction", "legacy_material_support"))
    parser.add_argument("--continuation-buy", dest="enable_continuation_buy",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--observe-continuation-structure",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="price-structure observation, independent of ordinary continuation orders")
    parser.add_argument("--continuation-accept-closes", type=int)
    parser.add_argument("--buy-htf-filter", choices=("none", "weekly_trend_up", "monthly_trend_up",
                                                       "weekly_bias_up", "monthly_bias_up"))
    parser.add_argument("--require-early-support-sweep", dest="require_early_support_sweep",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="require the early OB reaction low to wick below the OB distal edge")
    parser.add_argument("--support-candidate-mode", choices=("independent_ob_reactions", "legacy_watches"))
    parser.add_argument("--support-selection", choices=("structural_or_leg", "structural_role", "major_reaction", "legacy"))
    parser.add_argument("--post-sell-mode", choices=("two_paths", "legacy"))
    parser.add_argument("--double-top-qualification", choices=("major_leg", "legacy"))
    parser.add_argument("--reaction-filter", choices=("observe_only", "fixed_thresholds"))
    parser.add_argument("--management-mode", choices=("thesis_wave", "entry_scale", "legacy"))
    parser.add_argument("--reaction-failure-closes", type=int)
    parser.add_argument("--wave-pivot-radius", type=int)
    parser.add_argument("--wave-min-pullback-atr", type=float)
    parser.add_argument("--wave-exit-closes", type=int)
    parser.add_argument("--exit-reentry-policy", choices=("episode_scoped", "selective", "exit_reason", "off"))
    parser.add_argument("--failed-reaction-policy", choices=("episode_scoped", "reclaim_then_new_visit", "off"))
    parser.add_argument("--recovery-buy", dest="enable_recovery_buy",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recovery-structure-policy",
                        choices=("post_exit_low", "preserved_support_after_local_stop"),
                        help="strict v0.10 clock or an explicit preserved-support local-stop alternative")
    parser.add_argument("--exit-reclaim-closes", type=int)
    parser.add_argument("--continuation-context-mode", choices=("wave_or_reclaim", "legacy_local"))
    parser.add_argument("--continuation-risk-mode", choices=("reaction", "legacy_structure"))
    parser.add_argument("--continuation-max-risk-atr", type=float)
    parser.add_argument("--revision-stage", choices=("rev07_controls", "management", "management_reentry", "full",
                                                   "rev09_baseline", "rev09_failed_reaction", "rev09_selective",
                                                   "rev10_baseline", "rev10_episode_only",
                                                   "rev10_recovery_only", "rev10_combined",
                                                   "rev11_baseline", "rev11_without_continuation",
                                                   "rev11_local_stop_recovery", "rev11_combined"),
                        help="fixed prospective comparisons; no automatic fitting or model selection")
    parser.add_argument("--cycle-mode", choices=("major_top_boundary", "independent_advance", "structure_recovery", "legacy"))
    parser.add_argument("--sell-representation", choices=("parent_and_trigger", "legacy_rank"))
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--allow-unvalidated-replay", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if spec.get("implementation_version") != IMPLEMENTATION_VERSION:
        parser.error("configuration revision differs from the code; explicitly migrate the settings first")
    stages = {
        "rev07_controls": dict(management_mode="entry_scale", exit_reentry_policy="off", failed_reaction_policy="off",
                               continuation_context_mode="legacy_local", continuation_risk_mode="legacy_structure"),
        "management": dict(management_mode="thesis_wave", exit_reentry_policy="off", failed_reaction_policy="off",
                           continuation_context_mode="legacy_local", continuation_risk_mode="legacy_structure"),
        "management_reentry": dict(management_mode="thesis_wave", exit_reentry_policy="exit_reason", failed_reaction_policy="off",
                                   continuation_context_mode="wave_or_reclaim", continuation_risk_mode="legacy_structure"),
        "full": dict(management_mode="thesis_wave", exit_reentry_policy="exit_reason", failed_reaction_policy="off",
                     continuation_context_mode="wave_or_reclaim", continuation_risk_mode="reaction"),
        "rev09_baseline": dict(management_mode="entry_scale", exit_reentry_policy="off", failed_reaction_policy="off",
                               continuation_context_mode="legacy_local", continuation_risk_mode="legacy_structure"),
        "rev09_failed_reaction": dict(management_mode="entry_scale", exit_reentry_policy="off",
                                      failed_reaction_policy="reclaim_then_new_visit",
                                      continuation_context_mode="legacy_local", continuation_risk_mode="legacy_structure"),
        "rev09_selective": dict(management_mode="entry_scale", exit_reentry_policy="selective",
                                failed_reaction_policy="reclaim_then_new_visit",
                                continuation_context_mode="legacy_local", continuation_risk_mode="legacy_structure"),
    }
    # Historical comparison modes must not inherit the new default entry.
    for stage in stages.values():
        stage["enable_recovery_buy"] = False
        stage["enable_continuation_buy"] = True
        stage["observe_continuation_structure"] = True
        stage["recovery_structure_policy"] = "post_exit_low"
    v07_holding = dict(
        management_mode="entry_scale", attempt_allocation="unlimited",
        campaign_loss_limit_mode="observe_only", reentry_evidence_mode="fresh_price_reaction",
        observe_continuation_structure=True, enable_continuation_buy=True,
        recovery_structure_policy="post_exit_low", continuation_context_mode="legacy_local",
        continuation_risk_mode="legacy_structure", support_selection="structural_or_leg")
    stages.update({
        "rev10_baseline": dict(v07_holding, exit_reentry_policy="off",
                               failed_reaction_policy="off", enable_recovery_buy=False),
        "rev10_episode_only": dict(v07_holding, exit_reentry_policy="episode_scoped",
                                   failed_reaction_policy="episode_scoped", enable_recovery_buy=False),
        "rev10_recovery_only": dict(v07_holding, exit_reentry_policy="episode_scoped",
                                    failed_reaction_policy="off", enable_recovery_buy=True),
        "rev10_combined": dict(v07_holding, exit_reentry_policy="episode_scoped",
                               failed_reaction_policy="episode_scoped", enable_recovery_buy=True),
    })
    # Four prospective comparisons. The baseline is the v0.10 RECOVERY-ONLY
    # condition; legacy failure gates and holding rules are fixed across them.
    v10_recovery_control = stages["rev10_recovery_only"]
    stages.update({
        "rev11_baseline": dict(v10_recovery_control),
        "rev11_without_continuation": dict(v10_recovery_control, enable_continuation_buy=False),
        "rev11_local_stop_recovery": dict(v10_recovery_control,
                                          recovery_structure_policy="preserved_support_after_local_stop"),
        "rev11_combined": dict(v10_recovery_control, enable_continuation_buy=False,
                                recovery_structure_policy="preserved_support_after_local_stop"),
    })
    stage_overrides = stages.get(args.revision_stage, {})
    spec["campaign"].update(stage_overrides)
    overrides = {name: getattr(args, name) for name in (
                 "sell_detector", "reentry_mode", "attempt_allocation", "support_candidate_mode",
                 "support_selection", "post_sell_mode", "double_top_qualification",
                 "reaction_filter", "management_mode", "cycle_mode", "sell_representation",
                 "campaign_loss_limit_mode", "reentry_evidence_mode",
                 "buy_htf_filter", "require_early_support_sweep",
                 "observe_continuation_structure", "enable_continuation_buy",
                 "continuation_accept_closes", "enable_recovery_buy", "recovery_structure_policy",
                 "reaction_failure_closes", "wave_pivot_radius", "wave_min_pullback_atr",
                 "wave_exit_closes", "exit_reentry_policy", "exit_reclaim_closes", "failed_reaction_policy",
                 "continuation_context_mode", "continuation_risk_mode", "continuation_max_risk_atr")
                 if getattr(args, name) is not None}
    spec["campaign"].update(overrides)
    if args.describe:
        print(json.dumps(dict(status="AUTHORED_NOT_EXECUTED_OR_VALIDATED",
                              implementation_version=IMPLEMENTATION_VERSION,
                              specification=spec, cli_overrides=overrides,
                              revision_stage=args.revision_stage, stage_overrides=stage_overrides,
                              mode="local_research_only_no_live_orders"), ensure_ascii=False, indent=2))
        return
    if not args.allow_unvalidated_replay:
        parser.error("replay requires --allow-unvalidated-replay; no data has been loaded")
    if not all((args.calendar_json, args.as_of, args.output)):
        parser.error("--calendar-json, --as-of and --output are required")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")

    # Lazy imports make --describe a settings-only operation.
    import pandas as pd
    from nq_ict_retest_trade_state_machine_20260915 import load_inputs
    from nq_ict_retest_trade_state_machine_revision_20260915 import load_revision_config
    from run_nq_ict_early_buy_comparison_20260916 import make_rules, write_jsonl
    from run_nq_ict_macro_reversal_20260916 import clean
    from .common import CampaignConfig
    from .campaign import CampaignLedger
    from .source import CampaignSource, SessionSchedule

    cutoff = pd.Timestamp(args.as_of)
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        parser.error("--as-of must be a plain session date")
    execution_cfg = load_revision_config(args.base_config)
    rules = replace(make_rules(), **spec.get("legacy_rules_overrides", {}))
    cfg = CampaignConfig(**spec["campaign"])
    if cfg.top_support_radius not in {execution_cfg.local_pivot_radius, execution_cfg.major_pivot_radius,
                                     execution_cfg.trailing_pivot_radius}:
        parser.error("top support radius is not supplied by the existing pivot detector")
    if rules.pivot_radius not in {execution_cfg.local_pivot_radius, execution_cfg.major_pivot_radius,
                                 execution_cfg.trailing_pivot_radius}:
        parser.error("major structure radius is not supplied by the existing pivot detector")
    if cfg.wave_pivot_radius not in {execution_cfg.local_pivot_radius, execution_cfg.major_pivot_radius,
                                    execution_cfg.trailing_pivot_radius}:
        parser.error("wave radius is not supplied by the existing pivot detector")
    schedule = SessionSchedule(json.loads(args.calendar_json.read_text(encoding="utf-8-sig")))
    daily, objects, contexts = load_inputs(args.prices, args.objects, args.contexts,
                                           execution_cfg, args.as_of)
    schedule.validate_input(daily, cutoff)
    source = CampaignSource(daily, objects, contexts, execution_cfg, rules, cfg, schedule)
    ledger = CampaignLedger(rules, execution_cfg, cfg)
    inputs = dict(prices=args.prices, objects=args.objects, contexts=args.contexts,
                  calendar=args.calendar_json, config=args.config, base_config=args.base_config)
    dependencies = [
        ROOT / "run_nq_ict_early_buy_comparison_20260916.py",
        ROOT / "run_nq_ict_macro_reversal_20260916.py",
        ROOT / "nq_ict_retest_trade_state_machine_20260915.py",
        ROOT / "nq_ict_retest_trade_state_machine_revision_20260915.py",
    ]
    for dirname in ("nq_ict_macro_reversal_examples_20260916", "nq_ict_priority_entry_examples_20260916"):
        dependencies.extend(sorted((ROOT / dirname).glob("*.py")))
    dependencies.extend(sorted(PACKAGE.glob("*.py")))
    metadata = dict(
        implementation=f"nq_ict_campaign_proposal_{IMPLEMENTATION_VERSION}", verification_status="NOT_VALIDATED",
        run_status="started", as_of=args.as_of, daily_rows=len(daily),
        start=daily.datetime.iloc[0], end=daily.datetime.iloc[-1],
        campaign_config=asdict(cfg), legacy_rules=asdict(rules), execution_config=asdict(execution_cfg),
        cli_overrides=overrides,
        revision_stage=args.revision_stage, stage_overrides=stage_overrides,
        configured_design_variant=spec.get("design_variant"),
        revision11_factors=dict(
            observe_price_structure=cfg.observe_continuation_structure,
            ordinary_continuation_orders=cfg.enable_continuation_buy,
            recovery_structure_policy=cfg.recovery_structure_policy,
            comparison_reference="saved_rev10_recovery_only_not_yet_reproduced_by_v011"),
        input_hashes={key: dict(path=str(path.resolve()), sha256=file_hash(path)) for key, path in inputs.items()},
        code_hashes={str(path.resolve()): file_hash(path) for path in dependencies},
        assumptions=[
            "Trading-date labels and upstream completed weekly/monthly contexts are supplied inputs.",
            "Explicit caller calendar is not independently certified against CME here.",
            "Primary OBs remain v09 candidates; supplementary FVG/IFVG use a separate rule id.",
            "Daily close decisions fill at next observed open in the inherited simulation.",
            "Intrabar initial stop uses fixed slippage; intraday sequence is not reconstructed.",
            "SELL closes a long; initial position is flat; no fictitious first SELL trade.",
            "One-contract PnL is not account return; margin, funding and roll costs are not modeled.",
            "Market episodes receive new major-top signals, so packets are not identical to P0.",
            "Original detector episode IDs are not attempt-budget IDs. Both are retained separately.",
            "Reversal BUY uses a fresh post-boundary OB reaction; continuation BUY uses a separate accepted-break and OB/FVG reaction.",
            "Old pre-existing OBs are eligible; forming after SELL is not an entry requirement.",
            "Standard unlimited allocation observes counts without total or route purchase caps; historical campaign loss is also observe_only.",
            "Legacy count and campaign-loss policies require explicit configuration; they are not hidden caps in the standard proposal.",
            "A fresh recovery and decision after the ledger-wide last exit are required; the same zone can have new independent visits.",
            "Independent OB observations do not require an inherited BUY seed or a future-confirmed cohort.",
            "Only observed structural confirmations feed back after today's packet, independent of fills.",
            "Post-SELL paths use frozen decision-time peak/reference/ATR/low, not the final bottom.",
            "All new numerical thresholds are provisional design values, not fitted success rates.",
            "Pre-contact major roles OR bounded upward-leg attribution qualify OB provenance; the current reaction is separate.",
            "Upward-leg attribution uses frozen pre-origin anchors and a bounded formation window, never lifetime best departure.",
            "The old upward-leg reversal route retains its major-correction rule; continuation BUY does not inherit that rule.",
            "Major structure is a mechanical daily proxy, not an asserted weekly or Elliott degree.",
            "Reaction strength is descriptive by default; optional thresholds are unvalidated.",
            "A major parent range and its local SELL trigger are separate; early SELL is not necessarily full neckline failure.",
            "Major-top scope and legacy phase events remain contextual; standard BUY permission does not wait for allowance renewal.",
            "Continuation reuses daily structure-break events, preknown pivots and three close acceptances by default; weekly/monthly states are context only.",
            "Continuation zones must be linked to the accepted leg before a later contact. Only primary OBs and existing close-lifecycle daily FVGs are used.",
            "Revision 0.8 reaction risk uses the zone and observed reaction low; legacy_structure also includes the old structure low.",
            "Reaction-mode continuation risk is capped at the decision close and next open using frozen decision prior ATR; rejected candidates remain logged.",
            "Revision 0.11 retains entry-scale management; ordinary continuation orders can be disabled while their price-structure feed remains active.",
            "Optional thesis-wave management starts with a frozen entry recovery boundary and promotes prospectively after a confirmed larger daily wave.",
            "Daily wave H-L-H records use preknown H, an actual interval-low confirmed pivot, one depth threshold, and next-bar effectiveness.",
            "A deeper low and high breakout on the same daily candle is not accepted as an ordered H-L-break wave.",
            "Position protective levels only move upward; two-close default failures use next-observed-open exits, never retroactive intrabar stops.",
            "Optional v0.8 exit_reason reentry observes the frozen failure level, a fresh qualified lower OB reaction, or a new confirmed wave before contact.",
            "Selective v0.9 freezes the actual exit reason, not a planned SELL superseded by a stop; an exit-day later high is never used.",
            "Optional v0.9 memory uses reclaim of the prior-known holding high and a later contact across overlapping bands; v0.10 does not use this broad policy.",
            "Failure memory is based on actual initial-stop/reaction-failure exits, not loss amount, future success, scope changes, or cancelled orders.",
            "A distinct lower reversal OB after local failure must be disjoint and attributed to another precontact structure; substantial correction is still required.",
            "After local failure, continuation can use a still-valid accepted rise and fresh reaction; after top/protected-low exit it requires resolution or a new major structure before contact.",
            "Selective reentry uses the original major-structure records, not v0.8's smaller daily-wave book, as its alternative major-rise evidence.",
            "Fresh post-exit contact is stricter than merely a later recovery of the old reaction; purchase counts remain unlimited.",
            "Continuation wave-or-reclaim context and reaction risk can be switched separately for fixed comparison stages.",
            "All offered close-time gates are logged before selection, including while holding; next-open eligibility remains unknown.",
            "All BUY fills must exceed the initial stop; reversal BUY and legacy_structure continuation retain inherited uncapped risk. No generic gap cap is added.",
            "Source SELL invalidates the continuation entry context; an existing holding still uses the unchanged SELL/stop accounting.",
            "Only the selected SELL detector affects market state and execution; other outputs are diagnostics.",
            "Revision 0.10 failure memory is scoped to a source zone or the same structure/reaction, never price overlap alone, and freezes the entry reaction high.",
            "Episode memory can be resolved by reclaim followed by a new contact, or a distinct post-failure structure confirmed before contact; a new detector ID is insufficient.",
            "Ordinary reversal/continuation routes retain v0.7 market-path checks; they do not all require the last SELL checkpoint to be reclaimed.",
            "Recovery BUY requires an ACTUAL exit, frozen exit-level reclaim and a newly accepted post-exit breakout; there is no fictitious initial exit.",
            "The default recovery_structure_policy retains v0.10's post-exit-low requirement. A separate mode permits a pre-exit low only after a local reaction stop with preserved preknown major support.",
            "Preserved support is selected once at the actual local exit from prior-completed major structures below the local stop and below the exit fill; later close failure or a new major top permanently retires that frozen support for this exit.",
            "The preserved-support alternative never applies to major-top exits, protected-low exits or a failed recovery BUY. It still requires a fresh post-exit breakout, preknown pivots, reclaim and next-bar structure availability.",
            "Recovery candidates explicitly have null zone and contact fields; no OB/FVG touch or lower-timeframe confirmation is invented.",
            "Recovery uses the existing three-close continuation acceptance and two-close exit reclaim, then an effective structure; parameter values remain provisional.",
            "Recovery next-open orders are cancelled when the raw open is not above the frozen reclaim/breakout floor; same-day reentry remains blocked.",
            "Existing eligible reversal/continuation offers outrank recovery. A consumed recovery observation cannot be replayed daily or on a cancelled order without new structure.",
            "Recovery uses the same major-daily holding scale, hard initial stop, next-open accounting and one-position restriction; no pyramiding or short entry.",
            "Earlier revision results and audits do not validate revision 0.11; v0.10-control equivalence requires later verification.",
            "This proposal has no unconditional breakout-chase BUY, initial always-long entry or account-level monetary risk halt.",
            "No probabilities, unseen superiority, prefix invariance or ideal-date matches are asserted.",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=False)

    def save_json(name, value):
        (args.output / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                                  allow_nan=False), encoding="utf-8")

    save_json("metadata.json", metadata)
    try:
        for packet in source.packets():
            ledger.feed(packet)
            source.observe_independent_structure(ledger.today_independent_confirmations)
        outputs = {
            "trades.jsonl": ledger.trades,
            "execution_events.jsonl": ledger.execution_rows,
            "close_plans.jsonl": ledger.plan_rows,
            "daily_states.jsonl": ledger.state_rows,
            "daily_mtm.jsonl": ledger.daily_equity,
            "entry_audit.jsonl": ledger.entry_audit,
            "gate_events.jsonl": ledger.gate.events,
            "tracker_events.jsonl": ledger.tracker.events,
            "manager_events.jsonl": ledger.manager.events,
            "wave_management_events.jsonl": ledger.waves.events,
            "exit_reason_reentry_events.jsonl": ledger.exit_reentry.events,
            "preserved_support_events.jsonl": [e for e in ledger.exit_reentry.events
                                                if e.get("action") in {
                                                    "local_exit_major_support_frozen",
                                                    "preserved_major_support_invalidated"}],
            "campaign_events.jsonl": ledger.campaign_events,
            "campaign_sell_decisions.jsonl": ledger.sell_decisions,
            "sell_execution_links.jsonl": ledger.sell_execution_links,
            "position_evidence.jsonl": ledger.position_evidence_rows,
            "support_reaction_events.jsonl": ledger.supports.events,
            "independent_support_events.jsonl": ledger.independent.events,
            "independent_early_candidates.jsonl": ledger.independent.early_rows,
            "independent_confirmed_candidates.jsonl": ledger.independent.confirmed_rows,
            "continuation_events.jsonl": ledger.continuation.events,
            "continuation_candidates.jsonl": ledger.continuation.candidate_rows,
            "recovery_events.jsonl": ledger.recovery.events,
            "recovery_candidates.jsonl": ledger.recovery.candidate_rows,
            "independent_structure_events.jsonl": source.independent_structure_events,
            "support_selection.jsonl": ledger.support_selection_rows,
            "post_sell_path_events.jsonl": ledger.post_sell.events,
            "path_events.jsonl": ledger.paths.events,
            "path_links.jsonl": ledger.paths.links,
            "daily_evidence.jsonl": source.evidence_book.events,
            "tactical_zone_events.jsonl": source.evidence_book.zone_events,
            "major_top_events.jsonl": source.top_detector.events,
            "double_top_events.jsonl": source.double_top_detector.events,
            "double_top_major_diagnostics.jsonl": source.double_top_detector.major_signals,
            "double_top_early_diagnostics.jsonl": source.double_top_detector.early_signals,
            "major_reversal_events.jsonl": source.reversals.events,
            "major_structure_events.jsonl": source.structures.events,
            "ob_structural_role_events.jsonl": source.structures.role_events,
            "upward_leg_ob_events.jsonl": source.leg_supports.events,
            "raw_major_sell_candidates.jsonl": source.sell_selector.raw_signals,
            "sell_representative_events.jsonl": source.sell_selector.events,
            "market_cycle_events.jsonl": source.cycles.events,
            "market_episode_events.jsonl": source.market.events,
            "daily_source_counts.jsonl": source.stream_rows,
            "primary_zone_state_events.jsonl": source.zone_state_events,
            "pivot_confirmation_events.jsonl": source.pivot_events,
            "shared_early_candidates.jsonl": source.early_rows,
            "shared_confirmed_candidates.jsonl": source.confirmed_rows,
            "major_sell_candidates.jsonl": source.sell_rows,
        }
        for name, rows in outputs.items():
            write_jsonl(args.output / name, rows)
        save_json("final_state.json", ledger.final_state())
        save_json("entry_structural_contexts.json", ledger.entry_contexts)
        save_json("continuation_snapshot.json", ledger.continuation.snapshot)
        save_json("recovery_snapshot.json", ledger.recovery.snapshot)
        save_json("wave_management_snapshot.json", ledger.waves.snapshot)
        save_json("exit_reason_reentry_snapshot.json", ledger.exit_reentry.snapshot)
        save_json("position_management_snapshot.json", getattr(ledger.manager, "snapshot", {}))
        save_json("market_cycle_snapshot.json", dict(
            active=source.cycles.context, pending=source.cycles.pending,
            cycle_mode=cfg.cycle_mode,
            purchase_count_policy=cfg.attempt_allocation,
            campaign_loss_limit_mode=cfg.campaign_loss_limit_mode,
            scope_controls_standard_purchase_permission=False
            if cfg.attempt_allocation == "unlimited" and cfg.campaign_loss_limit_mode == "observe_only" else True,
            budget_parent_top=getattr(source.cycles, "budget_parent_top", None),
            seen_parent_groups=sorted(getattr(source.cycles, "seen_parent_groups", set())),
            seen_parent_clusters=sorted(getattr(source.cycles, "seen_parent_clusters", set())),
            independently_confirmed_advance=getattr(source.cycles, "advance", None),
            consumed_advances=sorted(getattr(source.cycles, "used_advances", set())),
            invalid_structures=sorted(source.cycles.invalid_structures),
            effective_policy=("major_parent_SELL_close_starts_scope; contacts_strictly_after_SELL; "
                              "independent_advance_not_a_budget_gate; upward_phase_next_bar"
                              if cfg.cycle_mode == "major_top_boundary" else
                              "next_bar_for_upward_phase; new_top_required_for_independent_advance_budget_renewal")))
        save_json("latest_snapshot.json", dict(
            datetime=daily.datetime.iloc[-1], state=ledger.state_rows[-1],
            today_plan=ledger.plan_rows[-1], final_state=ledger.final_state(),
            unfinished_candidates="right_censored_not_failures", verified=False))
        metadata.update(run_status="completed_replay_only", processed_bars=len(ledger.state_rows),
                        output_counts={name: len(rows) for name, rows in outputs.items()})
        save_json("metadata.json", metadata)
        print(f"Local replay written to {args.output}; research validation still pending.")
    except Exception as exc:
        metadata.update(run_status="failed", processed_bars=len(ledger.state_rows),
                        failure_type=type(exc).__name__, failure_message=str(exc))
        save_json("metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()

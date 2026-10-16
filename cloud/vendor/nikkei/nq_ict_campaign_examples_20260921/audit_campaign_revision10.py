"""Independent integrity and comparison audit for revision 0.10 full replays.

Predictive superiority is not inferred from the historical replay.  This audit
checks identical inputs, baseline reproduction, causal clocks, recovery-route
semantics, one-contract cash accounting, and reports stage deltas.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

try:
    from .audit_campaign_revision07 import (
        accounting_audit, buy_hold_audit, input_audit, read_json,
    )
except ImportError:  # Support direct invocation with this directory on sys.path.
    from audit_campaign_revision07 import (
        accounting_audit, buy_hold_audit, input_audit, read_json,
    )


ROOT = Path(__file__).resolve().parent
STAGES = {
    "rev10_baseline": "validation_rev10_baseline_full_20260923",
    "rev10_episode_only": "validation_rev10_episode_only_full_20260923",
    "rev10_recovery_only": "validation_rev10_recovery_only_full_20260923",
    "rev10_combined": "validation_rev10_combined_full_20260923",
}
BASELINE_CONTROL = ROOT / "validation_rev07_full_20260923"


def rows(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def check(ok, kind, **detail):
    return None if ok else {"type": kind, **detail}


def causal_audit(output: Path, prices_path: Path):
    failures = []
    with (prices_path).open(encoding="utf-8-sig", newline="") as stream:
        price_rows = list(csv.DictReader(stream))
    open_by_index = {i: float(r["open"]) for i, r in enumerate(price_rows)}
    close_by_index = {i: float(r["close"]) for i, r in enumerate(price_rows)}

    contexts = json.loads((output / "entry_structural_contexts.json").read_text(encoding="utf-8"))
    context_counts = Counter()
    for order_id, context in contexts.items():
        role = context.get("role") or {}
        family = context.get("entry_family")
        context_counts[family or "unknown"] += 1
        decision = context.get("decision_index")
        contact = context.get("contact_index")
        if family == "recovery":
            failures += [x for x in [
                check(context.get("source_zone") is None and contact is None,
                      "recovery_fabricated_zone_or_contact", order_id=order_id),
                check(role.get("kind") == "recovery_structure" and role.get("zone_id") is None,
                      "recovery_not_price_structure", order_id=order_id),
                check(decision is not None and role.get("known_index", math.inf) < decision
                      and role.get("effective_index", math.inf) <= decision,
                      "recovery_structure_not_known_at_decision", order_id=order_id,
                      decision_index=decision, role=role),
            ] if x]
        elif contact is None or not (role.get("known_index", math.inf) < contact
                                     and role.get("effective_index", math.inf) <= contact):
            failures.append({"type": "ordinary_entry_role_clock_failure", "order_id": order_id,
                             "contact_index": contact, "role": role})

    selected_count = 0
    selected_families = Counter()
    selected_failed = []
    unlimited_failures = []
    for row in rows(output / "support_selection.jsonl") or ():
        if not row.get("selected"):
            continue
        selected_count += 1
        family = row.get("entry_family") or "unknown"
        selected_families[family] += 1
        bad = [name for name, value in (row.get("checks") or {}).items()
               if isinstance(value, dict) and value.get("passed") is False]
        if bad:
            selected_failed.append({"key": row.get("key"), "failed": bad})
        components = ((row.get("checks") or {}).get("execution_budget") or {}).get("components") or {}
        for gate in ("total_attempts", "cumulative_loss", "route_attempts"):
            item = components.get(gate) or {}
            if item.get("passed") is False:
                unlimited_failures.append({"key": row.get("key"), "gate": gate, "detail": item})

    recovery_offers, recovery_failures = 0, []
    for event in rows(output / "recovery_events.jsonl") or ():
        if event.get("action") != "uptrend_recovery_offered":
            continue
        recovery_offers += 1
        c, d = event.get("candidate") or {}, event.get("detail") or {}
        s, ex, role = d.get("structure") or {}, d.get("exit_context") or {}, d.get("role") or {}
        checks = event.get("checks") or d.get("checks") or {}
        failures_here = []
        for name, value in checks.items():
            if value is not True:
                failures_here.append(f"failed_check:{name}")
        failures_here += [name for name, ok in {
            "real_executed_exit": ex.get("executed") is True and ex.get("index", math.inf) < c.get("index", -1),
            "post_exit_low": (s.get("low") or {}).get("origin_index", -1) > d.get("cutoff", math.inf),
            "post_exit_break": s.get("breakout_index", -1) > d.get("cutoff", math.inf),
            "pivots_known_before_break": (s.get("low") or {}).get("available_index", math.inf) < s.get("breakout_index", -1)
                                           and (s.get("high") or {}).get("available_index", math.inf) < s.get("breakout_index", -1),
            "structure_known_before_offer": s.get("known_index", math.inf) < c.get("index", -1)
                                             and s.get("effective_index", math.inf) <= c.get("index", -1),
            "zone_and_contact_null": c.get("zone") is None and c.get("touch_index") is None
                                     and d.get("source_zone") is None and d.get("contact_index") is None,
            "offer_close_above_frozen_floor": close_by_index.get(c.get("index", -1), -math.inf)
                                               > c.get("next_open_min_price", math.inf),
        }.items() if not ok]
        if failures_here:
            recovery_failures.append({"candidate_key": c.get("key"), "failures": failures_here})

    fills = {}
    fill_families = Counter()
    fill_clock_failures = []
    for event in rows(output / "execution_events.jsonl") or ():
        if event.get("action") != "buy_fill":
            continue
        order_id = event.get("order_id")
        fills[order_id] = event
        context = contexts.get(order_id) or {}
        fill_families[context.get("entry_family") or "unknown"] += 1
        idx = event.get("index")
        decision = context.get("decision_index")
        actual = float(event.get("price", math.nan))
        expected = open_by_index.get(idx, math.nan) + 0.25
        stop = context.get("stop")
        item = check(decision is not None and idx == decision + 1,
                     "buy_not_at_next_observed_open", order_id=order_id,
                     decision_index=decision, fill_index=idx)
        if item:
            fill_clock_failures.append(item)
        item = check(math.isfinite(expected) and abs(actual - expected) <= 1e-7,
                     "buy_fill_price_mismatch", order_id=order_id,
                     expected=expected, actual=actual)
        if item:
            fill_clock_failures.append(item)
        item = check(stop is not None and actual > float(stop),
                     "buy_fill_not_above_initial_stop", order_id=order_id,
                     stop=stop, fill_price=actual)
        if item:
            fill_clock_failures.append(item)
        if context.get("entry_family") == "recovery":
            min_open = (context.get("next_open_execution") or {}).get("minimum_raw_open_exclusive")
            raw_open = open_by_index.get(idx, math.nan)
            item = check(min_open is not None and raw_open > float(min_open),
                         "recovery_fill_below_frozen_open_floor", order_id=order_id,
                         raw_open=raw_open, minimum_raw_open_exclusive=min_open)
            if item:
                fill_clock_failures.append(item)

    candidate_row_count = sum(1 for _ in rows(output / "recovery_candidates.jsonl") or ())
    reentry_events = list(rows(output / "exit_reason_reentry_events.jsonl") or ())
    failed_reactions = [e for e in reentry_events if e.get("action") == "reaction_episode_failed"]
    failure_memory_violations = [e for e in failed_reactions
                                 if e.get("pnl_used") is not False
                                 or (e.get("failure") or {}).get("exit_index") != e.get("index")]
    return {
        "entry_context_count": len(contexts),
        "entry_context_family_counts": dict(context_counts),
        "ordinary_entry_clock_failures": [f for f in failures if f["type"] == "ordinary_entry_role_clock_failure"][:20],
        "recovery_context_failures": [f for f in failures if f["type"] != "ordinary_entry_role_clock_failure"][:20],
        "selected_close_gate_count": selected_count,
        "selected_family_counts": dict(selected_families),
        "selected_with_failed_gate_count": len(selected_failed),
        "selected_with_failed_gate_examples": selected_failed[:20],
        "unlimited_gate_failure_count": len(unlimited_failures),
        "unlimited_gate_failure_examples": unlimited_failures[:20],
        "recovery_offer_count": recovery_offers,
        "recovery_candidate_row_count": candidate_row_count,
        "recovery_candidate_event_count_mismatch": candidate_row_count != recovery_offers,
        "recovery_offer_invariant_failure_count": len(recovery_failures),
        "recovery_offer_invariant_failure_examples": recovery_failures[:20],
        "buy_fill_count": len(fills),
        "buy_fill_family_counts": dict(fill_families),
        "buy_fill_clock_or_price_failure_count": len(fill_clock_failures),
        "buy_fill_failures": fill_clock_failures[:20],
        "failed_reaction_events": len(failed_reactions),
        "failed_reaction_memory_violation_count": len(failure_memory_violations),
        "failed_reaction_memory_violation_examples": failure_memory_violations[:20],
    }


def metric_projection(audit):
    a = audit["accounting"]
    return {key: a.get(key) for key in (
        "closed_trades", "execution_buy_fill_count", "execution_sell_fill_count",
        "wins", "losses", "realized_net_usd", "final_mtm_usd", "mtm_max_drawdown_usd")}


def run_audit(args):
    prices_path = args.prices
    quality = input_audit(prices_path)
    with prices_path.open(encoding="utf-8-sig", newline="") as stream:
        quality["zero_volume_count"] = sum(float(row["volume"]) == 0
                                           for row in csv.DictReader(stream))
    reports = {}
    reference_meta = None
    hashes_match = True
    all_integrity = True
    all_policy_settings_match = True
    all_core_config_match = True
    expected_policy = {
        "rev10_baseline": ("off", "off", False),
        "rev10_episode_only": ("episode_scoped", "episode_scoped", False),
        "rev10_recovery_only": ("episode_scoped", "off", True),
        "rev10_combined": ("episode_scoped", "episode_scoped", True),
    }
    reference_core_config = None
    for stage, dirname in STAGES.items():
        path = args.root / dirname
        metadata = read_json(path / "metadata.json")
        if reference_meta is None:
            reference_meta = metadata
        config = metadata.get("campaign_config", {})
        policy_match = ((config.get("exit_reentry_policy"), config.get("failed_reaction_policy"),
                         config.get("enable_recovery_buy"))
                        == expected_policy[stage])
        core_config = {k: v for k, v in config.items()
                       if k not in {"exit_reentry_policy", "failed_reaction_policy", "enable_recovery_buy"}}
        if reference_core_config is None:
            reference_core_config = core_config
        core_match = core_config == reference_core_config
        all_policy_settings_match &= policy_match
        all_core_config_match &= core_match
        same_hashes = metadata.get("input_hashes") == reference_meta.get("input_hashes")
        hashes_match &= same_hashes
        accounting = accounting_audit(path, prices_path, metadata)
        benchmark = buy_hold_audit(path, prices_path)
        causal = causal_audit(path, prices_path)
        complete = metadata.get("run_status") == "completed_replay_only"
        integrity = (complete and metadata.get("daily_rows") == quality["rows"]
                     and accounting["daily_mtm_rows"] == quality["rows"]
                     and policy_match and core_match
                     and accounting["accounting_mismatch_count"] == 0
                     and causal["selected_with_failed_gate_count"] == 0
                     and causal["unlimited_gate_failure_count"] == 0
                     and causal["buy_fill_clock_or_price_failure_count"] == 0
                     and causal["failed_reaction_memory_violation_count"] == 0
                     and causal["recovery_offer_invariant_failure_count"] == 0
                     and not causal["recovery_candidate_event_count_mismatch"]
                     and causal["ordinary_entry_clock_failures"] == []
                     and causal["recovery_context_failures"] == [])
        all_integrity &= integrity
        reports[stage] = {
            "output": str(path.resolve()), "run_status": metadata.get("run_status"),
            "revision_stage": metadata.get("revision_stage"),
            "input_hashes_match_all_stages": same_hashes,
            "expected_stage_policy_matches": policy_match,
            "non_treatment_config_matches": core_match,
            "stage_overrides": metadata.get("stage_overrides"),
            "integrity_pass": integrity,
            "metrics": metric_projection({"accounting": accounting}),
            "accounting": accounting,
            "buy_hold": benchmark,
            "causal": causal,
        }

    old = read_json(BASELINE_CONTROL / "audit_rev07.json")
    old_meta = read_json(BASELINE_CONTROL / "metadata.json")
    current_baseline = reports["rev10_baseline"]
    old_metrics = metric_projection(old)
    new_metrics = current_baseline["metrics"]
    baseline_metric_diffs = {k: {"rev07": old_metrics[k], "rev10_baseline": new_metrics[k]}
                             for k in old_metrics if old_metrics[k] != new_metrics[k]}
    old_hashes, current_hashes = old_meta.get("input_hashes", {}), reference_meta.get("input_hashes", {})
    baseline_same_data_hashes = ({k: v.get("sha256") for k, v in old_hashes.items() if k != "config"}
                                 == {k: v.get("sha256") for k, v in current_hashes.items() if k != "config"})
    changed_hash_keys = [k for k in old_hashes.keys() | current_hashes.keys()
                         if old_hashes.get(k, {}).get("sha256") != current_hashes.get(k, {}).get("sha256")]
    old_campaign, new_campaign = old_meta.get("campaign_config", {}), reference_meta.get("campaign_config", {})
    baseline_old_config_preserved = all(new_campaign.get(k) == v for k, v in old_campaign.items())
    baseline_rule_configs_preserved = (old_meta.get("legacy_rules") == reference_meta.get("legacy_rules")
                                       and old_meta.get("execution_config") == reference_meta.get("execution_config"))
    baseline_new_policy_off = (new_campaign.get("exit_reentry_policy") == "off"
                               and new_campaign.get("failed_reaction_policy") == "off"
                               and new_campaign.get("enable_recovery_buy") is False)
    baseline_equivalent = (not baseline_metric_diffs and baseline_same_data_hashes
                           and baseline_old_config_preserved and baseline_rule_configs_preserved
                           and baseline_new_policy_off
                           and current_baseline["run_status"] == "completed_replay_only")
    complete = all(reports[s]["run_status"] == "completed_replay_only" for s in STAGES)
    status = ("PASS_WITH_RESEARCH_CAVEATS" if complete and quality["sorted_unique_dates"]
              and quality["duplicate_date_count"] == 0
              and quality["missing_or_non_numeric_count"] == 0
              and quality["negative_volume_count"] == 0
              and hashes_match and all_integrity and baseline_equivalent else "NEEDS_REVIEW")
    base = reports["rev10_baseline"]["metrics"]
    for stage, result in reports.items():
        result["delta_vs_rev10_baseline"] = {
            key: (result["metrics"].get(key) - base.get(key)
                  if isinstance(result["metrics"].get(key), (int, float))
                  and isinstance(base.get(key), (int, float)) else None)
            for key in base
        }
    episode = reports["rev10_episode_only"]["metrics"]
    recovery = reports["rev10_recovery_only"]["metrics"]
    combined = reports["rev10_combined"]["metrics"]
    factorial = {}
    for key, base_value in base.items():
        if (isinstance(base_value, (int, float))
                and all(isinstance(m.get(key), (int, float)) for m in (episode, recovery, combined))):
            factorial[key] = {
                "episode_effect_without_recovery": episode[key] - base_value,
                "recovery_effect_without_episode": recovery[key] - base_value,
                "combined_interaction": combined[key] - episode[key] - recovery[key] + base_value,
            }
    result = {
        "audit_status": status,
        "interpretation": "historical reproducibility and implementation-integrity audit; not unseen predictive validation",
        "input_quality": quality,
        "all_stages_same_input_hashes": hashes_match,
        "all_stages_use_expected_treatment_switches": all_policy_settings_match,
        "all_non_factor_configuration_identical": all_core_config_match,
        "rev07_control": {"same_market_objects_context_calendar_and_base_config_hashes": baseline_same_data_hashes,
                           "old_campaign_settings_preserved": baseline_old_config_preserved,
                           "legacy_and_execution_rules_preserved": baseline_rule_configs_preserved,
                           "new_policies_disabled_in_control": baseline_new_policy_off,
                           "changed_input_hash_keys": changed_hash_keys,
                           "equivalent_metrics": baseline_equivalent,
                           "metric_differences": baseline_metric_diffs,
                           "metrics": old_metrics},
        "factorial_effect_decomposition": factorial,
        "stages": reports,
    }
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "audit_rev10.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, dest / "report_rev10_ja.md")
    return result


def usd(value):
    return "n/a" if value is None else f"{value:,.1f}"


def write_report(result, path: Path):
    lines = [
        "# NKD=F 改訂0.10 全期間検証", "",
        f"監査判定: **{result['audit_status']}**", "",
        "これは同一の過去データ上での再現性・実装・会計監査であり、未見期間の予測優位性を示さない。",
        "4段階の比較は設定差の履歴上の影響を記述する。対象期間を繰り返し参照しているため独立テストではない。", "",
        f"入力品質: {result['input_quality']['rows']:,}行、{result['input_quality']['first']}〜{result['input_quality']['last']}、"
        f"時系列整列={result['input_quality']['sorted_unique_dates']}、OHLC範囲矛盾={result['input_quality']['impossible_ohlc_count']}、"
        f"出来高ゼロ={result['input_quality']['zero_volume_count']}。",
        f"4段階で入力ハッシュ一致: {result['all_stages_same_input_hashes']}。",
        f"3つのステージ制御値以外の設定一致: {result['all_non_factor_configuration_identical']}。",
        f"段階ごとのON/OFF指定一致: {result['all_stages_use_expected_treatment_switches']}。", "",
        "| 条件 | 完了取引 | BUY約定 | 実現損益 USD | 期末MTM USD | 最大DD USD | B&H比 期末差 USD | 回復BUY候補 / 選択 / 約定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, d in result["stages"].items():
        a, c, b = d["accounting"], d["causal"], d["buy_hold"]
        lines.append(f"| {stage} | {a['closed_trades']} | {a['execution_buy_fill_count']} | {usd(a['realized_net_usd'])} | "
                     f"{usd(a['final_mtm_usd'])} | {usd(a['mtm_max_drawdown_usd'])} | "
                     f"{usd(b['strategy_final_minus_buy_hold_usd'])} | {c['recovery_offer_count']} / "
                     f"{c['selected_family_counts'].get('recovery', 0)} / "
                     f"{c['buy_fill_family_counts'].get('recovery', 0)} |")
    lines += ["", "## 0.7基準再現性", "",
              f"Buy & Hold参照: 期末MTM {usd(result['stages']['rev10_baseline']['buy_hold']['final_mtm_usd'])} USD、"
              f"最大DD {usd(result['stages']['rev10_baseline']['buy_hold']['max_drawdown_usd'])} USD。",
              f"0.7保存済み実行と価格・オブジェクト・文脈・暦・基本設定のハッシュ一致: "
              f"{result['rev07_control']['same_market_objects_context_calendar_and_base_config_hashes']}。",
              f"0.7時点の設定値を維持: {result['rev07_control']['old_campaign_settings_preserved']}。",
              f"レガシー売買・約定ルールの設定一致: {result['rev07_control']['legacy_and_execution_rules_preserved']}。",
              f"新設ポリシーを基準で無効化: {result['rev07_control']['new_policies_disabled_in_control']}。",
              f"ハッシュ差がある入力キー: `{result['rev07_control']['changed_input_hash_keys']}` "
              "（`config`はv0.10比較用に更新された設定ファイルの差）。",
              f"主要会計指標の一致: {result['rev07_control']['equivalent_metrics']}。",
              f"差分: `{json.dumps(result['rev07_control']['metric_differences'], ensure_ascii=False)}`。", "",
              "## 条件別の変化", "",
              "比較上の制御値は `failed_reaction_policy` と `enable_recovery_buy` の2つ。"
              "`exit_reentry_policy=episode_scoped` は失敗記憶または実退出後復帰に使う状態トラッカーの選択で、"
              "`failed_reaction_policy=off` の回復BUY単独条件では通常エントリーを拒否しない。", ""]
    base = result["stages"]["rev10_baseline"]["metrics"]
    for stage, d in result["stages"].items():
        if stage == "rev10_baseline":
            continue
        delta = d["delta_vs_rev10_baseline"]
        lines.append(f"- `{stage}`−基準: 完了取引 {delta['closed_trades']:+}, BUY {delta['execution_buy_fill_count']:+}, "
                     f"実現損益 {usd(delta['realized_net_usd'])}, 期末MTM {usd(delta['final_mtm_usd'])}, "
                     f"最大DD差 {usd(delta['mtm_max_drawdown_usd'])} USD。")
    lines += ["", "## 2×2差分の分解", "",
              "差分はこの履歴上の記述値であり、統計的有意性や因果効果を意味しない。", "",
              "| 指標 | 失敗反応制限のみ | 復帰BUYのみ | 併用の相互作用 |",
              "|---|---:|---:|---:|"]
    for key in ("closed_trades", "execution_buy_fill_count", "realized_net_usd",
                "final_mtm_usd", "mtm_max_drawdown_usd"):
        f = result["factorial_effect_decomposition"].get(key, {})
        lines.append(f"| {key} | {usd(f.get('episode_effect_without_recovery'))} | "
                     f"{usd(f.get('recovery_effect_without_episode'))} | {usd(f.get('combined_interaction'))} |")
    lines += ["", "### 2016〜2018 / 2023〜終端", ""]
    for stage, d in result["stages"].items():
        lines.append(f"- `{stage}`:")
        for w in d["buy_hold"]["windows"]:
            lines.append(f"  - {w['start']}〜{w['end']}: 戦略 {usd(w['strategy_mtm_change_usd'])} USD / "
                         f"B&H {usd(w['buy_hold_mtm_change_usd'])} USD / 差 {usd(w['strategy_vs_buy_hold_change_usd'])} USD。")
    lines += ["", "## 実装・因果監査", ""]
    for stage, d in result["stages"].items():
        c = d["causal"]
        lines.append(f"- `{stage}`: 選択後ゲート違反 {c['selected_with_failed_gate_count']}、通常エントリー時計違反 "
                     f"{len(c['ordinary_entry_clock_failures'])}、回復構造違反 {c['recovery_offer_invariant_failure_count']}、"
                     f"BUY約定時刻・価格違反 {c['buy_fill_clock_or_price_failure_count']}、会計不一致 "
                     f"{d['accounting']['accounting_mismatch_count']}。")
        lines.append(f"  - 系統別完了取引: `{json.dumps(c['entry_context_family_counts'], ensure_ascii=False)}`。")
        if c["recovery_offer_count"]:
            lines.append(f"  - 回復BUY候補 {c['recovery_offer_count']}、選択 {c['selected_family_counts'].get('recovery', 0)}、"
                         f"約定 {c['buy_fill_family_counts'].get('recovery', 0)}。zone/contactは全候補でnull、"
                         "実退出後の新構造だけで成立。")
    lines += ["", "## 解釈と限界", "",
              "- `rev10_baseline` が改訂0.7を再現しない場合、他3条件の差を改訂0.10追加ルールの効果とは読めない。",
              "- 条件別差は同じ履歴へのルール追加効果であり、閾値探索・戦略選択を含む後知恵の影響が残る。",
              "- 取引はNKD=F連続先物1枚の固定USD損益。証拠金、限月ロール、資金調達、口座リスク制約を含まない。",
              "- 日足OHLCから同一足内の到達順は復元しない。スリッページ・手数料の仮定は既存比較と共通。",
              "- 入力に記録されたOHLC範囲矛盾・出来高ゼロ件数は上記品質欄を参照。元データを補正せず同じCSVのまま全条件へ適用した。",
              "- これはバックテストの完全性確認で、将来収益・口座資産倍率・ICT理論一般の有効性を保証しない。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prices", type=Path, default=ROOT.parent / "nq_25y_actual_2000_2026_20260911" / "NQ_F_1d_combined.csv")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args)
    print(json.dumps({"audit_status": result["audit_status"],
                      "same_hashes": result["all_stages_same_input_hashes"],
                      "rev07_equivalent": result["rev07_control"]["equivalent_metrics"],
                      "stages": {k: v["metrics"] for k, v in result["stages"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

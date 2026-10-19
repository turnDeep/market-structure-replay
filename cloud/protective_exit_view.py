"""Render the existing entry-scale protective SELL without inventing a top.

Presentation only: all prices, confirmation counts and activation dates come
from native frozen logs. Unknown protective routes fail rather than guess.
"""


def protective_sell(ns,sd):
    assert sd["reason"]=="major_protected_low_close_failure",sd["reason"]
    assert ns["meta"]["campaign_config"]["management_mode"]=="entry_scale"
    bars,day,mark,level,step,visual=[ns[k] for k in ["bars","day","mark","level","step","visual"]]
    i=sd["index"];ref=sd["frozen_exit"]["reference"]
    proposals=[r for r in ns["jl"]("manager_events") if r["position_id"]==sd["position_id"]
        and r["index"]<i and r.get("proposal",{}).get("pivot",{}).get("id")==ref["id"]]
    assert proposals,"Missing native protective-low activation"
    proposal=proposals[-1]["proposal"];count=ns["meta"]["legacy_rules"]["structural_exit_closes"]
    assert count==2 and ref["available_index"]<=proposal["decided_index"]<proposal["effective_from_index"]<=i-count+1
    assert all(bars[j]["close"]<ref["price"] for j in range(i-count+1,i+1))
    assert bars[i-count]["close"]>=ref["price"],"Not the first confirmed protective failure"
    marks=[mark("P","構造保護安値",ref["origin_index"],ref["price"],ref["available_index"]),
        mark("3a","安値割れ1本目",i-1,bars[i-1]["close"],i-1),mark("3b","SELL判断",i,bars[i]["close"],i)]
    levels=[level("protected-low","構造保護安値",ref["price"],proposal["effective_from_index"])]
    steps=[step("1","保護する安値を事前確定",f"{day(ref['origin_index'])} の安値 {ref['price']:,.2f}。確定 {day(ref['available_index'])}、保護水準として有効 {day(proposal['effective_from_index'])}。"),
        step("2","保護安値の下で2終値",f"{day(i-1)}・{day(i)} の終値が {ref['price']:,.2f} を下回る。H1/H2の天井SELLではなく、保有波動の構造保護決済。"),
        step("3","翌始値SELLを予約","ロング決済。判断日時点では翌始値・約定可否・最終損益は未知。")]
    ev=visual("sell",i,"事前に有効な構造保護安値 → 2終値割れ → 翌始値SELL",max(0,i-80),marks,levels,steps)
    ev["exit_reason"]=sd["reason"]
    return ev,steps,"構造保護安値割れSELL","\n".join(s["title"]+"："+s["detail"] for s in steps)


def adapt_template(source):
    start="            sd=cl['sell_decision'];t=sd['representative_selection']['local_trigger'];atr=t['h1']['atr_at_origin']"
    end="            decisions.append(dict(id=oid,index=d,date=day(d),side='SELL'"
    assert source.count(start)==1 and source.count(end)==1
    a=source.index(start);b=source.index(end,a)
    original=source[a:b]
    tail=original.split(";atr=",1)[1]
    ordinary="            atr="+tail
    guarded="            sd=cl['sell_decision'];t=sd['representative_selection']['local_trigger']\n"
    guarded+="            if t is None:\n                ev,steps,route,reason=_protective_sell(globals(),sd)\n"
    guarded+="                assert i==d+1 and abs(r['price']-(bars[i]['open']-slip))<1e-8\n"
    guarded+="            else:\n"+"\n".join("    "+line if line else line for line in ordinary.split("\n"))
    return source[:a]+guarded+source[b:]

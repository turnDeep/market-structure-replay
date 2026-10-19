"""Release identity, causality and migration boundaries, not predictive tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cloud.strategy import STRATEGY_VERSION, stamp_metadata, validate_strategy


def test_only_alpha_is_the_released_strategy():
    base=dict(legacy_rules=dict(buy_reference_radius=1,emergency_buffer_atr=.10,major_decline_bars=15),
              campaign_config=dict(enable_continuation_buy=False))
    assert stamp_metadata(deepcopy(base))["strategy_version"]==STRATEGY_VERSION=="v0.12"
    for key,value in [("buy_reference_radius",2),("emergency_buffer_atr",.12),("major_decline_bars",18)]:
        bad=deepcopy(base);bad["legacy_rules"][key]=value
        with pytest.raises(ValueError):validate_strategy(bad)


def test_presentation_labels_upgrade_inline_and_encoded_scripts_not_data():
    import base64
    from scripts.release_v012 import presentation_labels
    from scripts.import_replays import SCRIPT
    old="const version='v0.11'; const filename='NQ_v011_asof.json';"
    encoded=base64.b64encode(old.encode()).decode()
    html='<title>v0.11</title><script type="application/json">{"old":"v0.11"}</script>'+f'<script src="data:text/javascript;charset=utf-8;base64,{encoded}"></script><script>'+old+'</script>'
    result=presentation_labels(html)
    assert '<title>v0.12</title>' in result
    scripts=list(SCRIPT.finditer(result))
    assert scripts[0].group(2)=='{"old":"v0.11"}'
    assert 'v0.12' in scripts[2].group(2) and '_v012_' in scripts[2].group(2)
    assert base64.b64encode(old.replace('v0.11','v0.12').replace('_v011_','_v012_').encode()).decode() in scripts[1].group(1)


def test_refresh_count_has_a_real_anchor_when_older_snapshot_lacks_one():
    from cloud.export_adapter import refresh_metadata
    bars=[dict(date="2026-10-14"),dict(date="2026-10-15"),dict(date="2026-10-16")]
    audit=dict(captured_at_utc="now")
    first=refresh_metadata({},bars,audit,"2026-10-15",2)
    assert first["frozenThrough"]=="2026-10-15" and first["appendedBars"]==1
    second=refresh_metadata(dict(refresh=first),bars+[dict(date="2026-10-19")],audit,"2026-10-16",3)
    assert second["appendedBars"]==2 and second["latestAppendedBars"]==1


@pytest.mark.parametrize("market", ["nq","nikkei"])
def test_radius_one_matches_strict_extrema_without_future_tail(market):
    # Each vendor namespace lives in its own subprocess, as in production.
    code=r'''
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path('cloud/vendor')/sys.argv[1]))
from nq_ict_campaign_examples_20260921.source import CampaignSource
from cloud.strategy import install_pivot_supply
native_calls=[]
def native(self):
    native_calls.append(self.i)
    return []
CampaignSource.confirm_pivots=native
install_pivot_supply()
method=CampaignSource.confirm_pivots
install_pivot_supply()
assert CampaignSource.confirm_pivots is method
values=np.random.default_rng(12).normal(size=100).cumsum()+100
d=pd.DataFrame(dict(datetime=pd.date_range('2020-01-01',periods=100),high=values+1,low=values-1,atr_prior=1.))
# Include a tied high, which must not be recognized as a strict maximum.
d.loc[10:11,'high']=150
def read(frame,stop):
    obj=SimpleNamespace(d=frame,i=0,t=None,pivots=[],rules=SimpleNamespace(buy_reference_radius=1),log=lambda *a,**k:None)
    for i in range(stop):
        obj.i=i;obj.t=frame.datetime.iloc[i];method(obj)
    return [(p.kind,p.radius,p.index,p.available_at,p.price) for p in obj.pivots]
full=read(d,100)
for stop in [3,12,25,60,99]:
    prefix=read(d.iloc[:stop],stop)
    assert prefix==[p for p in full if p[2]+1<stop]
    changed=d.copy();changed.loc[stop:,'high']+=5000;changed.loc[stop:,'low']-=5000
    assert read(changed,stop)==prefix
for kind,radius,origin,available,price in full:
    assert radius==1 and available==d.datetime.iloc[origin+1]
    column='high' if kind=='high' else 'low'
    neighbors=d[column].iloc[origin-1:origin+2].tolist()
    assert price==(max(neighbors) if kind=='high' else min(neighbors))
    assert neighbors.count(price)==1
assert not any(p[0]=='high' and p[2] in [10,11] for p in full)
assert native_calls
print('PASS_RADIUS_ONE_PREFIX')
'''
    result=subprocess.run([sys.executable,"-c",code,market],capture_output=True,text=True,timeout=60,
        cwd=Path(__file__).resolve().parents[1])
    assert result.returncode==0,result.stdout+result.stderr


def test_released_view_and_seed_names_are_consistent():
    from cloud.common import CLOUD, load
    from replay_assets import load_manifest, load_replay
    from scripts.import_replays import read_snapshot
    manifest=load_manifest()
    assert manifest["strategy_version"]=="v0.12"
    for market,item in manifest["markets"].items():
        snapshot=read_snapshot(load_replay(market,item))
        assert snapshot["replay"]["strategyVersion"]=="v0.12"
        assert snapshot["replay"]["buyReferenceRadius"]==1
        assert "v0.12" in snapshot["title"]
        validate_strategy(load(CLOUD/"state"/market/"metadata.json"))

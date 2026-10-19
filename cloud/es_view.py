"""ES presentation adapter: distinguish actual FVG from inversion FVG."""


def retest_labels(source):
    replacements={
        "route='支持OB・FVG再テストBUY'":"route=f\"支持OB・{tz['kind'].upper()}再テストBUY\"",
        "に利用可能となったFVG {fmt(tz['low'])}":"に利用可能となった{tz['kind'].upper()} {fmt(tz['low'])}",
        "ev['retest_zone']=retest;":"ev['retest_zone']=retest;ev['retest_used']=bool(retest);",
        "ev['mechanism']='支持OBで構造回復 → 確認後のFVG再テスト → 翌始値BUY'":"ev['mechanism']=f\"支持OBで構造回復 → 確認後の{retest['kind']}再テスト → 翌始値BUY\"",
        "'このBUYは構造後のFVG再テストを採用。'":"f\"このBUYは構造後の{retest['kind']}再テストを採用。\"",
    }
    for old,new in replacements.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source.replace("採用FVG下端","採用FVG/IFVG下端").replace("採用FVG上端","採用FVG/IFVG上端").replace("採用FVG再テスト","採用FVG/IFVG再テスト")

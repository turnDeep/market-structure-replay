"""Current-model display labels; never rename old data, files or audit history."""
import re

from cloud.strategy import STRATEGY_VERSION

OLD_LABEL = re.compile(r"v0\.11|revision 0\.11|v011")
HISTORY_PREFIX = "旧v0.11時点（移行前）の記録："


def current_labels(snapshot):
    """Enumerate only active measurement labels, not provenance/history text."""
    yield "title", snapshot["title"]
    for name,query in snapshot["queries"].items():
        source=query.get("source",{})
        if "label" in source:
            yield f"queries.{name}.source.label",source["label"]
        for i,definition in enumerate(source.get("metricDefinitions",[])):
            if "label" in definition:
                yield f"queries.{name}.source.metricDefinitions[{i}].label",definition["label"]


def normalize_current_labels(snapshot):
    if snapshot.get("replay",{}).get("strategyVersion") != STRATEGY_VERSION:
        raise ValueError("Cannot relabel a different strategy as the current model")
    changes=[]
    def update(obj,key,path):
        old=obj.get(key)
        if isinstance(old,str) and "v0.11" in old:
            obj[key]=old.replace("v0.11",STRATEGY_VERSION)
            changes.append(path)
    update(snapshot,"title","title")
    for name,query in snapshot["queries"].items():
        source=query.get("source",{})
        update(source,"label",f"{name}.source.label")
        for i,definition in enumerate(source.get("metricDefinitions",[])):
            update(definition,"label",f"{name}.metricDefinitions[{i}].label")
        for event in source.get("evidenceFlow",[]):
            # These really were v0.11 operations. Keep their original detail;
            # make the historical scope explicit instead of falsifying it.
            if OLD_LABEL.search(event.get("detail","")) and not event.get("title","").startswith(HISTORY_PREFIX):
                event["title"]=HISTORY_PREFIX+event.get("title","")
    validate_current_labels(snapshot)
    return changes


def validate_current_labels(snapshot):
    if snapshot.get("replay",{}).get("strategyVersion") != STRATEGY_VERSION:
        raise ValueError("Unexpected current strategy version")
    stale=[path for path,text in current_labels(snapshot) if OLD_LABEL.search(text)]
    if stale:
        raise ValueError("Stale current-model version label: "+", ".join(stale))

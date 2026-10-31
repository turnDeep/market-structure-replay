"""Current-model display labels; never rename old data, files or audit history."""
import re

from cloud.strategy import STRATEGY_VERSION

OLD_LABEL = re.compile(r"v0\.11|v0\.12|revision 0\.11|revision 0\.12|v011|v012")
HISTORY_PREFIX = "旧バージョン時点（移行前）の記録："
LEGACY_HISTORY_PREFIX = "旧v0.11時点（移行前）の記録："
OLD_VERSIONS = ("v0.12", "v0.11")


def _upgrade(text):
    for old in OLD_VERSIONS:
        text = text.replace(old, STRATEGY_VERSION)
        text = text.replace("revision "+old[1:], "revision "+STRATEGY_VERSION[1:])
        text = text.replace("_"+old.replace(".", "")+"_", "_"+STRATEGY_VERSION.replace(".", "")+"_")
    return text


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
        if isinstance(old,str):
            new=_upgrade(old)
            if new != old:
                obj[key]=new
                changes.append(path)
    update(snapshot,"title","title")
    for name,query in snapshot["queries"].items():
        source=query.get("source",{})
        update(source,"label",f"{name}.source.label")
        for i,definition in enumerate(source.get("metricDefinitions",[])):
            update(definition,"label",f"{name}.metricDefinitions[{i}].label")
        for event in source.get("evidenceFlow",[]):
            # These really were older-version operations. Keep their original
            # detail; make the historical scope explicit instead of falsifying it.
            title=event.get("title","")
            if OLD_LABEL.search(event.get("detail","")) and not title.startswith((HISTORY_PREFIX, LEGACY_HISTORY_PREFIX)):
                event["title"]=HISTORY_PREFIX+title
    validate_current_labels(snapshot)
    return changes


def validate_current_labels(snapshot):
    if snapshot.get("replay",{}).get("strategyVersion") != STRATEGY_VERSION:
        raise ValueError("Unexpected current strategy version")
    stale=[path for path,text in current_labels(snapshot) if OLD_LABEL.search(text)]
    if stale:
        raise ValueError("Stale current-model version label: "+", ".join(stale))

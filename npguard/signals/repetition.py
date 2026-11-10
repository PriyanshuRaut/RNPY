"""Find distinct registered owning arrays at the same external call site."""

from collections import defaultdict


def detect_repeated_allocations(registry):
    entries = registry.values() if isinstance(registry, dict) else registry
    groups = defaultdict(list)
    seen = set()
    for info in entries:
        identity = (info["id"], info["registered_at"])
        if identity in seen or not info["owndata"]:
            continue
        seen.add(identity)
        key = (info["shape"], info["dtype"], info["callsite"])
        groups[key].append(info)
    return {key: items for key, items in groups.items() if len(items) > 1}

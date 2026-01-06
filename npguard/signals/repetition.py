from collections import defaultdict

def detect_repeated_allocations(registry):
    groups = defaultdict(list)
    for info in registry.values():
        key = (info["shape"], info["dtype"], info["callsite"])
        groups[key].append(info)

    return {k: v for k, v in groups.items() if len(v) > 1}

from collections import defaultdict

def detect_parallel_spikes(timeline):
    by_thread = defaultdict(list)
    for e in timeline:
        by_thread[e["thread"]].append(e)

    if len(by_thread) <= 1:
        return None

    total = sum(e["size"] for e in timeline)
    per_thread = {t: sum(e["size"] for e in v) for t, v in by_thread.items()}

    return {
        "threads": len(by_thread),
        "total_bytes": total,
        "per_thread": per_thread,
    }

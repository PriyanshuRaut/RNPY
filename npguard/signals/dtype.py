"""Dtype differences require an explicit caller-supplied source relationship."""

def detect_dtype_promotion(registry):
    entries = registry.values() if isinstance(registry, dict) else registry
    findings = []
    for info in entries:
        source = info.get("source")
        if not source or source["dtype"] == info["dtype"]:
            continue
        widening = (
            source["dtype_kind"] in "biufc" and info["dtype_kind"] in "biufc"
            and info["itemsize"] > source["itemsize"]
            and source["safe_cast_to_result"]
        )
        findings.append({
            "from": source["dtype"],
            "to": info["dtype"],
            "size": info["size"],
            "callsite": info["callsite"],
            "kind": "dtype_widening" if widening else "dtype_difference",
            "confidence": "high",
            "evidence": {
                "relationship": source["relationship"],
                "source_shape": source["shape"],
                "result_shape": info["shape"],
                "may_share_memory": source["may_share_memory"],
                "safe_cast": source["safe_cast_to_result"],
                "logical_size_difference_bytes": (
                    info["size"] - source["size"] if source["shape"] == info["shape"] else None
                ),
            },
            "implicit_promotion_confirmed": False,
            "allocation_confirmed": False,
        })
    return findings

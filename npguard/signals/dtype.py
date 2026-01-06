def detect_dtype_promotion(registry):
    promotions = []

    for info in registry.values():
        parent = info.get("parent_dtype")
        if parent and parent != info["dtype"]:
            promotions.append({
                "from": parent,
                "to": info["dtype"],
                "size": info["size"],
                "callsite": info["callsite"],
            })

    return promotions

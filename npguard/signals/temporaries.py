def detect_temporaries(timeline, start_time, end_time):
    """
    Detect arrays that were allocated and destroyed
    within the observation window.
    """
    temporaries = []

    seen_ids = set()
    for entry in timeline:
        if start_time <= entry["time"] <= end_time:
            arr_id = entry["id"]
            if arr_id not in seen_ids:
                temporaries.append(entry)
                seen_ids.add(arr_id)

    return temporaries

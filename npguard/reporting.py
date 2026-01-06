# npguard/reporting.py

from .registry import ArrayRegistry

def report():
    """
    Print cumulative allocation summary.
    """
    if not ArrayRegistry:
        print("\n[npguard] Allocation Summary (cumulative)")
        print("  No arrays registered.")
        return

    print("\n[npguard] Allocation Summary (cumulative)")
    totals = {}

    for info in ArrayRegistry.values():
        label = info["label"]
        totals[label] = totals.get(label, 0) + info["size"]

    for label, size in totals.items():
        print(f"  {label:<12}: {size / 1024 / 1024:.2f} MB")

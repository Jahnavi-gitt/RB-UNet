"""Structured logging for metrics, training curves, and experiment progress."""

import json
from pathlib import Path
from typing import Any, Dict, List


def save_json_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary to a JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def log_training_history(history: List[Dict[str, Any]], filepath: str) -> None:
    """Save training epoch history to CSV and JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(p.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save CSV
    if history:
        headers = list(history[0].keys())
        csv_path = p.with_suffix(".csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in history:
                f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

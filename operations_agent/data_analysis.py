from dataclasses import dataclass, asdict
from datetime import date, timedelta
import numpy as np
import pandas as pd
from database.queries import read_period, readonly_connection


@dataclass(frozen=True)
class Period:
    start: str
    end: str

    def __post_init__(self):
        a, b = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if a > b:
            raise ValueError("Period start must not exceed end")
        if (b - a).days > 365:
            raise ValueError("Period limited to 366 days")

    @property
    def days(self):
        return (date.fromisoformat(self.end) - date.fromisoformat(self.start)).days + 1


def summarize(frame, period):
    if frame.empty:
        return {
            "sufficient": False,
            "rows": 0,
            "reason": "No records for requested period",
            "period": asdict(period),
        }
    contradictions = []
    numeric = [
        "planned_minutes",
        "actual_minutes",
        "completed_quantity",
        "downtime_minutes",
        "production_target",
        "scheduled_minutes",
    ]
    for c in numeric:
        if (frame[c].dropna() < 0).any():
            contradictions.append(f"Negative {c}")
    if frame.duplicated(["production_date", "machine_id"]).any():
        contradictions.append("Duplicate machine/date")
    known = frame.downtime_minutes.notna()
    if (
        frame.loc[known, "actual_minutes"] + frame.loc[known, "downtime_minutes"]
        > frame.loc[known, "scheduled_minutes"]
    ).any():
        contradictions.append("Processing plus downtime exceeds scheduled time")
    if (
        frame[
            [
                "actual_minutes",
                "completed_quantity",
                "production_target",
                "scheduled_minutes",
                "planned_minutes",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        contradictions.append("Missing required measurement")
    if (frame.production_target <= 0).any() or (frame.scheduled_minutes <= 0).any():
        contradictions.append("Nonpositive denominator")
    if contradictions:
        return {
            "sufficient": False,
            "rows": len(frame),
            "contradictions": contradictions,
            "period": asdict(period),
        }
    per_machine = []
    for machine, g in frame.groupby("machine_id"):
        per_machine.append(
            {
                "machine": machine,
                "quantity": int(g.completed_quantity.sum()),
                "known_downtime_minutes": float(g.downtime_minutes.sum()),
                "missing_downtime": int(g.downtime_minutes.isna().sum()),
                "busy_utilization": float(
                    g.actual_minutes.sum() / g.scheduled_minutes.sum()
                ),
                "planned_minutes": float(g.planned_minutes.sum()),
                "actual_minutes": float(g.actual_minutes.sum()),
            }
        )
    return {
        "sufficient": True,
        "period": asdict(period),
        "rows": len(frame),
        "quantity": int(frame.completed_quantity.sum()),
        "target": int(frame.production_target.sum()),
        "target_attainment": float(
            frame.completed_quantity.sum() / frame.production_target.sum()
        ),
        "throughput_units_per_scheduled_minute": float(
            frame.completed_quantity.sum() / frame.scheduled_minutes.sum()
        ),
        "known_downtime_minutes": float(frame.downtime_minutes.sum()),
        "downtime_total_complete": bool(known.all()),
        "missing_values": {c: int(frame[c].isna().sum()) for c in numeric},
        "per_machine": per_machine,
    }


def compare(database, previous, current):
    if previous.days != current.days:
        raise ValueError("Compare equal-length periods")
    if not previous.end < current.start:
        raise ValueError("Comparison periods must be chronological and disjoint")
    before = read_period(database, previous.start, previous.end)
    after = read_period(database, current.start, current.end)
    a, b = summarize(before, previous), summarize(after, current)
    result = {
        "previous": a,
        "current": b,
        "sufficient": a["sufficient"] and b["sufficient"],
    }
    if not result["sufficient"]:
        return result
    with readonly_connection(database) as conn:
        entry = conn.execute(
            "SELECT value FROM metadata WHERE key=?", ("machines",)
        ).fetchone()
    expected_machines = (
        set(entry[0].split(","))
        if entry
        else set(before.machine_id) | set(after.machine_id)
    )
    expected_rows = previous.days * len(expected_machines)
    if len(before) != expected_rows or len(after) != expected_rows:
        result.update(
            sufficient=False,
            reason="Missing machine/date records across comparison periods",
        )
        return result
    delta = b["quantity"] - a["quantity"]
    result.update(
        delta_quantity=delta,
        percentage_change=100 * delta / a["quantity"] if a["quantity"] else None,
        known_downtime_delta=b["known_downtime_minutes"] - a["known_downtime_minutes"],
        downtime_comparison_complete=a["downtime_total_complete"]
        and b["downtime_total_complete"],
    )
    machine_changes = []
    old = {r["machine"]: r for r in a["per_machine"]}
    for r in b["per_machine"]:
        machine_changes.append(
            {
                "machine": r["machine"],
                "quantity_delta": r["quantity"] - old[r["machine"]]["quantity"],
                "known_downtime_delta": r["known_downtime_minutes"]
                - old[r["machine"]]["known_downtime_minutes"],
            }
        )
    result["machine_changes"] = machine_changes
    return result


def plot_comparison(metrics, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not metrics["sufficient"]:
        return
    before = metrics["previous"]["per_machine"]
    after = metrics["current"]["per_machine"]
    x = np.arange(len(before))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [
        (axes[0], "quantity", "Completed units"),
        (axes[1], "known_downtime_minutes", "Known downtime (missing values excluded)"),
    ]:
        ax.bar(x - 0.18, [r[key] for r in before], 0.36, label="previous")
        ax.bar(x + 0.18, [r[key] for r in after], 0.36, label="current")
        ax.set_xticks(x, [r["machine"] for r in before])
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

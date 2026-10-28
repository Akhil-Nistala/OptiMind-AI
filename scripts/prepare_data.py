import sqlite3
import argparse
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from common import save_json
from optimization.problem import Problem, Job, Operation


def demo_problem():
    return Problem(
        3,
        [
            Job("A", (Operation(0, 8), Operation(1, 5), Operation(2, 4)), deadline=22),
            Job(
                "B",
                (Operation(1, 6), Operation(0, 7), Operation(2, 3)),
                deadline=21,
                weight=2,
            ),
            Job("C", (Operation(2, 5), Operation(1, 7)), release=2, deadline=20),
            Job("D", (Operation(0, 4), Operation(2, 8)), deadline=19),
        ],
        crew_capacity=2,
        available_from={1: 2},
        blackouts={0: [(12, 16)]},
        name="next_cycle_demo",
        source="authored synthetic",
    )


def prepare(path="data/production.sqlite", seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(14):
        current = day >= 7
        stamp = (date(2026, 10, 5) + timedelta(days=day)).isoformat()
        for m in range(3):
            downtime = 30 + int(rng.integers(-5, 6)) + (90 if current and m == 1 else 0)
            actual = 400 - int(rng.integers(0, 11)) - (75 if current and m == 1 else 0)
            quantity = (
                100 + int(rng.integers(-4, 5)) - (25 if current and m == 1 else 0)
            )
            missing = current and m == 2 and day == 10
            rows.append(
                {
                    "production_date": stamp,
                    "job_id": f"JOB-{day:02}-{m}",
                    "machine_id": f"M{m}",
                    "planned_minutes": 400,
                    "actual_minutes": actual,
                    "completed_quantity": quantity,
                    "downtime_minutes": None if missing else downtime,
                    "production_target": 100,
                    "scheduled_minutes": 480,
                }
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(Path("database/schema.sql").read_text())
        conn.execute("DELETE FROM production")
        conn.execute("DELETE FROM metadata")
        pd.DataFrame(rows).to_sql("production", conn, if_exists="append", index=False)
        conn.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [("synthetic", "true"), ("seed", str(seed)), ("machines", "M0,M1,M2")],
        )
    # Independent fixture reference uses generator records, not the investigative SQL/Pandas tools.
    totals = {}
    for name, part in [("previous", rows[:21]), ("current", rows[21:])]:
        totals[name] = {
            "quantity": sum(r["completed_quantity"] for r in part),
            "known_downtime": sum(r["downtime_minutes"] or 0 for r in part),
            "missing_downtime": sum(r["downtime_minutes"] is None for r in part),
            "target": sum(r["production_target"] for r in part),
        }
    totals["delta_quantity"] = (
        totals["current"]["quantity"] - totals["previous"]["quantity"]
    )
    save_json(
        "data/fixture_reference.json",
        {"synthetic": True, "seed": seed, "reference": totals, "records": rows},
    )
    save_json("configs/demo_schedule.json", demo_problem().to_dict())
    save_json(
        "results/data_manifest.json",
        {
            "synthetic": True,
            "seed": seed,
            "rows": len(rows),
            "machines": ["M0", "M1", "M2"],
            "previous": ["2026-10-05", "2026-10-11"],
            "current": ["2026-10-12", "2026-10-18"],
            "mechanism": "M1 current-period output -25 and downtime +90 minutes; small seeded noise; one M2 downtime missing",
            "reference": totals,
        },
    )
    print(f"Created {len(rows)} explicitly synthetic records in {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="data/production.sqlite")
    a = p.parse_args()
    prepare(a.database)

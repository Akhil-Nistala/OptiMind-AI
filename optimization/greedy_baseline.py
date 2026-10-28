import time
from optimization.problem import schedule_metrics, validate_schedule


def greedy(problem):
    """Serial schedule generation: earliest feasible start, earliest due-date tie break."""
    began = time.perf_counter()
    rows = []
    next_op = {j.id: 0 for j in problem.jobs}
    ready = {j.id: j.release for j in problem.jobs}
    while any(next_op[j.id] < len(j.operations) for j in problem.jobs):
        candidates = []
        for job in problem.jobs:
            i = next_op[job.id]
            if i == len(job.operations):
                continue
            op = job.operations[i]
            chosen = None
            for start in range(
                max(ready[job.id], problem.available_from.get(op.machine, 0)),
                problem.horizon - op.duration + 1,
            ):
                end = start + op.duration
                if any(
                    start < b and end > a
                    for a, b in problem.blackouts.get(op.machine, [])
                ):
                    continue
                if any(
                    r["machine"] == op.machine and start < r["end"] and end > r["start"]
                    for r in rows
                ):
                    continue
                # Integer times; inspect event segments for cumulative resource demand.
                points = {start, end} | {
                    t for r in rows for t in (r["start"], r["end"]) if start < t < end
                }
                if any(
                    op.crew + sum(r["crew"] for r in rows if r["start"] <= t < r["end"])
                    > problem.crew_capacity
                    for t in sorted(points)[:-1]
                ):
                    continue
                if (
                    job.hard_deadline
                    and i == len(job.operations) - 1
                    and end > job.deadline
                ):
                    continue
                chosen = start
                break
            if chosen is not None:
                candidates.append((chosen, job.deadline, op.duration, job.id, i, op))
        if not candidates:
            return {
                "status": "HEURISTIC_FAILED",
                "schedule": None,
                "runtime_seconds": time.perf_counter() - began,
                "proven_optimal": False,
            }
        start, _, _, jid, i, op = min(candidates, key=lambda c: c[:5])
        rows.append(
            {
                "job": jid,
                "operation": i,
                "machine": op.machine,
                "start": start,
                "end": start + op.duration,
                "crew": op.crew,
            }
        )
        next_op[jid] += 1
        ready[jid] = start + op.duration
    assert not validate_schedule(problem, rows)
    return {
        "status": "GREEDY_FEASIBLE",
        "schedule": rows,
        **schedule_metrics(problem, rows),
        "runtime_seconds": time.perf_counter() - began,
        "proven_optimal": False,
    }

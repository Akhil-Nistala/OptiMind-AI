import time
from ortools.sat.python import cp_model
from optimization.problem import schedule_metrics, validate_schedule


def solve(problem, time_limit=5.0, seed=42):
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    began = time.perf_counter()
    model = cp_model.CpModel()
    horizon = problem.horizon
    starts = {}
    ends = {}
    intervals = {}
    by_machine = {m: [] for m in range(problem.machines)}
    demands = []
    for job in problem.jobs:
        for i, op in enumerate(job.operations):
            key = (job.id, i)
            name = f"{job.id}_{i}"
            s = model.new_int_var(0, horizon, "s_" + name)
            e = model.new_int_var(0, horizon, "e_" + name)
            interval = model.new_interval_var(s, op.duration, e, "i_" + name)
            starts[key] = s
            ends[key] = e
            intervals[key] = interval
            by_machine[op.machine].append(interval)
            demands.append(op.crew)
            model.add(s >= max(job.release, problem.available_from.get(op.machine, 0)))
            if i:
                model.add(s >= ends[job.id, i - 1])
        if job.hard_deadline:
            model.add(ends[job.id, len(job.operations) - 1] <= job.deadline)
    for machine, slots in problem.blackouts.items():
        for i, (start, end) in enumerate(slots):
            by_machine[machine].append(
                model.new_fixed_size_interval_var(
                    start, end - start, f"blackout_{machine}_{i}"
                )
            )
    for slots in by_machine.values():
        model.add_no_overlap(slots)
    model.add_cumulative(list(intervals.values()), demands, problem.crew_capacity)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        makespan, [ends[j.id, len(j.operations) - 1] for j in problem.jobs]
    )
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    name = solver.status_name(status)
    result = {
        "status": name,
        "schedule": None,
        "proven_optimal": status == cp_model.OPTIMAL,
        "runtime_seconds": time.perf_counter() - began,
        "bound": (
            float(solver.best_objective_bound)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            else None
        ),
        "time_limit_seconds": time_limit,
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        rows = [
            {
                "job": j.id,
                "operation": i,
                "machine": op.machine,
                "start": solver.value(starts[j.id, i]),
                "end": solver.value(ends[j.id, i]),
                "crew": op.crew,
            }
            for j in problem.jobs
            for i, op in enumerate(j.operations)
        ]
        errors = validate_schedule(problem, rows)
        if errors:
            raise RuntimeError(f"Solver output failed independent validator: {errors}")
        result.update(schedule=rows, **schedule_metrics(problem, rows))
        result["relative_bound_gap"] = (result["makespan"] - result["bound"]) / max(
            1.0, result["makespan"]
        )
    return result

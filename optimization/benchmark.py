import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import save_json, load_json
from optimization.problem import Problem, Job, Operation
from optimization.greedy_baseline import greedy
from optimization.solver import solve
from optimization.visualization import gantt


def synthetic(jobs, machines, seed):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(jobs):
        route = rng.permutation(machines)
        operations = tuple(Operation(int(m), int(rng.integers(2, 12))) for m in route)
        items.append(
            Job(
                f"J{i}",
                operations,
                release=int(rng.integers(0, 5)),
                deadline=30 + i * 3,
                weight=1 + i % 3,
            )
        )
    return Problem(
        machines,
        items,
        crew_capacity=max(2, machines - 1),
        blackouts={0: [(12, 16)]},
        name=f"synthetic_{jobs}x{machines}_{seed}",
        source="seeded synthetic",
    )


def public_example():
    # Transcribed input data from Google's public job-shop example. Known optimum verified independently by CP-SAT here.
    routes = [[(0, 3), (1, 2), (2, 2)], [(0, 2), (2, 1), (1, 4)], [(1, 4), (2, 3)]]
    return Problem(
        3,
        [
            Job(f"J{i}", tuple(Operation(m, p) for m, p in route), deadline=15)
            for i, route in enumerate(routes)
        ],
        3,
        name="google_public_3x3",
        source="https://developers.google.com/optimization/scheduling/job_shop",
    )


def ft06():
    durations = [
        [1, 3, 6, 7, 3, 6],
        [8, 5, 10, 10, 10, 4],
        [5, 4, 8, 9, 1, 7],
        [5, 5, 5, 3, 8, 9],
        [9, 3, 5, 4, 3, 1],
        [3, 3, 9, 10, 4, 1],
    ]
    machines = [
        [2, 0, 1, 3, 5, 4],
        [1, 2, 4, 5, 0, 3],
        [2, 3, 5, 0, 1, 4],
        [1, 0, 2, 3, 4, 5],
        [2, 1, 4, 5, 0, 3],
        [1, 3, 5, 0, 4, 2],
    ]
    return Problem(
        6,
        [
            Job(
                f"J{i}",
                tuple(Operation(m, d) for m, d in zip(machines[i], durations[i])),
                deadline=100,
            )
            for i in range(6)
        ],
        6,
        name="ft06",
        source="https://github.com/google/or-tools/blob/stable/examples/python/jobshop_ft06_sat.py",
    )


def benchmark(output="results/scheduling"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    config = load_json("configs/experiments.json")
    instances = [public_example(), ft06()] + [
        synthetic(j, m, config["seed"] + s)
        for j, m in config["benchmark_sizes"]
        for s in range(config["seeds_per_size"])
    ]
    rows = []
    for problem in instances:
        save_json(out / "instances" / f"{problem.name}.json", problem.to_dict())
        baseline = greedy(problem)
        optimized = solve(problem, config["solver_seconds"], config["seed"])
        save_json(out / f"{problem.name}_greedy.json", baseline)
        save_json(out / f"{problem.name}_solver.json", optimized)
        reference = optimized.get("makespan") if optimized["proven_optimal"] else None
        for method, result in [("greedy", baseline), ("cp_sat", optimized)]:
            rows.append(
                {
                    "instance": problem.name,
                    "source": problem.source,
                    "jobs": len(problem.jobs),
                    "machines": problem.machines,
                    "operations": sum(len(j.operations) for j in problem.jobs),
                    "method": method,
                    "status": result["status"],
                    "proven_optimal": result["proven_optimal"],
                    "makespan": result.get("makespan"),
                    "weighted_tardiness": result.get("weighted_tardiness"),
                    "violations": result.get("constraint_violations"),
                    "runtime_seconds": result["runtime_seconds"],
                    "solver_bound": optimized.get("bound"),
                    "gap_to_proven_optimum": (
                        (result["makespan"] - reference) / reference
                        if reference and result.get("schedule")
                        else None
                    ),
                    "relative_bound_gap": (
                        (result["makespan"] - optimized["bound"])
                        / max(1, result["makespan"])
                        if optimized.get("bound") is not None and result.get("schedule")
                        else None
                    ),
                }
            )
        if problem.name == instances[0].name:
            gantt(problem, baseline, out / "public_greedy.png")
            gantt(problem, optimized, out / "public_solver.png")
        print(
            problem.name,
            baseline.get("makespan"),
            optimized["status"],
            optimized.get("makespan"),
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, g in df.groupby("method"):
        means = g.groupby("operations")[["makespan", "runtime_seconds"]].mean()
        axes[0].plot(means.index, means.makespan, "-o", label=method)
        axes[1].plot(means.index, means.runtime_seconds, "-o", label=method)
    axes[0].set_ylabel("Makespan (mean over seeds)")
    axes[1].set_ylabel("Wall time, seconds")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Operation count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "scalability.png", dpi=150)
    plt.close(fig)
    paired = df.pivot(index="instance", columns="method", values="makespan")
    improvements = 100 * (paired.greedy - paired.cp_sat) / paired.greedy
    stats = {
        "instances": len(instances),
        "feasible_schedules": int(df.makespan.notna().sum()),
        "total_constraint_violations": int(df.violations.fillna(0).sum()),
        "solver_optimal_count": int(df[(df.method == "cp_sat")].proven_optimal.sum()),
        "mean_makespan_reduction_percent": float(improvements.mean()),
        "public_example_optimum": int(
            df[
                (df.instance == "google_public_3x3") & (df.method == "cp_sat")
            ].makespan.iloc[0]
        ),
        "config": config,
    }
    save_json(out / "summary.json", stats)
    print(stats)


if __name__ == "__main__":
    benchmark()

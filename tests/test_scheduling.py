import copy
import itertools
import pytest
from optimization.problem import Problem, Job, Operation, validate_schedule
from optimization.solver import solve
from optimization.greedy_baseline import greedy
from optimization.benchmark import public_example


def test_public_example():
    p = public_example()
    r = solve(p)
    assert r["status"] == "OPTIMAL" and r["makespan"] == 11
    assert r["bound"] == 11 and not validate_schedule(p, r["schedule"])
    assert greedy(p)["makespan"] == 12


def test_resource_calendar_and_release():
    p = Problem(
        2,
        [Job("a", (Operation(0, 3),), release=1), Job("b", (Operation(1, 3),))],
        1,
        blackouts={0: [(1, 4)]},
        available_from={1: 2},
    )
    for result in [greedy(p), solve(p)]:
        assert not validate_schedule(p, result["schedule"]) and result["makespan"] >= 8


def test_infeasible_deadline_and_unknown_limit():
    p = Problem(1, [Job("a", (Operation(0, 5),), deadline=2, hard_deadline=True)], 1)
    assert solve(p)["status"] == "INFEASIBLE"
    assert greedy(p)["status"] == "HEURISTIC_FAILED"
    r = solve(public_example(), time_limit=1e-9)
    assert (
        r["status"] == "UNKNOWN" and r["schedule"] is None and not r["proven_optimal"]
    )


def test_independent_exhaustive_optimum():
    routes = [[(0, 3), (1, 2)], [(1, 2), (0, 4)], [(0, 2), (1, 1)]]
    p = Problem(
        2,
        [
            Job(str(i), tuple(Operation(m, d) for m, d in ops))
            for i, ops in enumerate(routes)
        ],
        2,
    )
    best = 1000
    # Enumerate all topological operation orders; earliest-start schedules suffice for regular makespan.
    for order in itertools.permutations([(j, i) for j in range(3) for i in range(2)]):
        indices = {op: k for k, op in enumerate(order)}
        if any(indices[j, 0] > indices[j, 1] for j in range(3)):
            continue
        machine = [0, 0]
        job = [0, 0, 0]
        for j, i in order:
            m, d = routes[j][i]
            end = max(machine[m], job[j]) + d
            machine[m] = end
            job[j] = end
        best = min(best, max(job))
    r = solve(p)
    assert r["status"] == "OPTIMAL" and r["makespan"] == best


def test_validator_rejects_tampering():
    p = public_example()
    rows = solve(p)["schedule"]
    bad = copy.deepcopy(rows)
    bad[0]["end"] += 1
    assert validate_schedule(p, bad)
    assert validate_schedule(p, rows[:-1])
    assert validate_schedule(p, rows + [rows[0]])


@pytest.mark.parametrize("duration", [0, -1, 1.5, True])
def test_invalid_duration(duration):
    with pytest.raises(ValueError):
        Problem(1, [Job("a", (Operation(0, duration),))], 1)


def test_invalid_calendar():
    with pytest.raises(ValueError):
        Problem(1, [Job("a", (Operation(0, 1),))], 1, blackouts={0: [(2, 5), (4, 8)]})

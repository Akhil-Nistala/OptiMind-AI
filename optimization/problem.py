"""Validated fixed-route, nonpreemptive job-shop instances."""

from dataclasses import dataclass, field, asdict


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Operation:
    machine: int
    duration: int
    crew: int = 1


@dataclass(frozen=True)
class Job:
    id: str
    operations: tuple[Operation, ...]
    release: int = 0
    deadline: int = 100
    weight: int = 1
    hard_deadline: bool = False


@dataclass
class Problem:
    machines: int
    jobs: list[Job]
    crew_capacity: int
    available_from: dict[int, int] = field(default_factory=dict)
    blackouts: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    horizon_limit: int | None = None
    name: str = "instance"
    source: str = "synthetic"

    def __post_init__(self):
        integer(self.machines, "machines", 1)
        integer(self.crew_capacity, "crew_capacity", 1)
        if not self.jobs or len({j.id for j in self.jobs}) != len(self.jobs):
            raise ValueError("Jobs must be nonempty with unique IDs")
        for job in self.jobs:
            if not isinstance(job.id, str) or not job.id.strip() or not job.operations:
                raise ValueError("Each job needs an ID and operations")
            integer(job.release, "release")
            integer(job.deadline, "deadline")
            integer(job.weight, "weight", 1)
            if not isinstance(job.hard_deadline, bool):
                raise ValueError("hard_deadline must be boolean")
            for op in job.operations:
                integer(op.machine, "machine")
                integer(op.duration, "duration", 1)
                integer(op.crew, "crew", 1)
                if op.machine >= self.machines or op.crew > self.crew_capacity:
                    raise ValueError(
                        "Operation machine or crew demand exceeds capacity"
                    )
        for machine, time in self.available_from.items():
            integer(machine, "machine")
            integer(time, "availability")
            if machine >= self.machines:
                raise ValueError("Unknown availability machine")
        for machine, intervals in self.blackouts.items():
            integer(machine, "machine")
            if machine >= self.machines:
                raise ValueError("Unknown blackout machine")
            end = -1
            for start, stop in sorted(intervals):
                integer(start, "blackout start")
                integer(stop, "blackout end", 1)
                if start >= stop or start < end:
                    raise ValueError(
                        "Blackouts must be positive, non-overlapping intervals"
                    )
                end = stop
        if self.horizon_limit is not None:
            integer(self.horizon_limit, "horizon_limit", 1)

    @property
    def horizon(self):
        if self.horizon_limit is not None:
            return self.horizon_limit
        latest = max(
            [0]
            + [j.release for j in self.jobs]
            + list(self.available_from.values())
            + [end for slots in self.blackouts.values() for _, end in slots]
        )
        return latest + sum(op.duration for job in self.jobs for op in job.operations)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        allowed = {
            "machines",
            "jobs",
            "crew_capacity",
            "available_from",
            "blackouts",
            "horizon_limit",
            "name",
            "source",
        }
        if set(data) - allowed:
            raise ValueError(f"Unknown problem fields: {set(data)-allowed}")
        d = dict(data)
        d["jobs"] = [
            Job(**{**j, "operations": tuple(Operation(**op) for op in j["operations"])})
            for j in d["jobs"]
        ]
        d["available_from"] = {
            int(k): v for k, v in d.get("available_from", {}).items()
        }
        d["blackouts"] = {
            int(k): [tuple(x) for x in v] for k, v in d.get("blackouts", {}).items()
        }
        return cls(**d)


def validate_schedule(problem, schedule):
    errors = []
    expected = {
        (j.id, i): (j, op) for j in problem.jobs for i, op in enumerate(j.operations)
    }
    seen = {}
    for row in schedule:
        key = (row.get("job"), row.get("operation"))
        if key not in expected or key in seen:
            errors.append(f"Duplicate/unknown operation {key}")
            continue
        job, op = expected[key]
        seen[key] = row
        start, end = row.get("start"), row.get("end")
        if type(start) != int or type(end) != int:
            errors.append(f"Noninteger time {key}")
            continue
        if row.get("machine") != op.machine or end - start != op.duration:
            errors.append(f"Wrong machine/duration {key}")
        if (
            start < max(job.release, problem.available_from.get(op.machine, 0))
            or end > problem.horizon
        ):
            errors.append(f"Release/availability/horizon violation {key}")
        if any(start < b and end > a for a, b in problem.blackouts.get(op.machine, [])):
            errors.append(f"Blackout violation {key}")
    if set(seen) != set(expected):
        errors.append("Missing operations")
    if errors:
        return errors
    for job in problem.jobs:
        for i in range(1, len(job.operations)):
            if seen[job.id, i]["start"] < seen[job.id, i - 1]["end"]:
                errors.append("Precedence violation")
        if (
            job.hard_deadline
            and seen[job.id, len(job.operations) - 1]["end"] > job.deadline
        ):
            errors.append("Hard deadline violation")
    for m in range(problem.machines):
        ordered = sorted(
            [r for r in schedule if r["machine"] == m], key=lambda r: r["start"]
        )
        if any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])):
            errors.append("Machine overlap")
    events = []
    for key, row in seen.items():
        crew = expected[key][1].crew
        events.extend([(row["start"], crew), (row["end"], -crew)])
    demand = 0
    for _, change in sorted(events, key=lambda x: (x[0], x[1])):
        demand += change
        if demand > problem.crew_capacity:
            errors.append("Crew capacity violation")
            break
    return errors


def schedule_metrics(problem, schedule):
    completion = {
        j.id: max(r["end"] for r in schedule if r["job"] == j.id) for j in problem.jobs
    }
    return {
        "makespan": max(completion.values()),
        "weighted_tardiness": sum(
            j.weight * max(0, completion[j.id] - j.deadline) for j in problem.jobs
        ),
        "constraint_violations": len(validate_schedule(problem, schedule)),
    }

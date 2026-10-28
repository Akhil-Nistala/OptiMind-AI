"""Bounded LangGraph tool orchestration; no free-form SQL or operational execution."""

import re
import time
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from common import save_json
from operations_agent.data_analysis import Period, compare
from optimization.problem import Problem, validate_schedule
from optimization.solver import solve


@dataclass(frozen=True)
class Request:
    question: str
    database: str = "data/production.sqlite"
    previous_start: str = "2026-10-05"
    previous_end: str = "2026-10-11"
    current_start: str = "2026-10-12"
    current_end: str = "2026-10-18"
    scheduling_problem: Problem | None = None

    def __post_init__(self):
        if (
            not isinstance(self.question, str)
            or not 3 <= len(self.question.strip()) <= 2000
        ):
            raise ValueError("Operational question must be 3â€“2000 characters")
        Period(self.previous_start, self.previous_end)
        Period(self.current_start, self.current_end)


class State(TypedDict, total=False):
    request: Request
    intent: list[str]
    plan: list[str]
    logs: list[dict]
    tool_calls: list[dict]
    metrics: dict
    knowledge: dict
    optimization: dict
    errors: list[str]
    validated: bool
    report: dict


def route(question):
    q = question.lower()
    result = []
    if re.search(
        r"production|throughput|quantity|downtime|utilization|utilisation|planned|actual|missing|investigat|output|compare.*period",
        q,
    ):
        result.append("analyze")
    if re.search(
        r"why|cause|procedure|manual|compressed air|leak|compressor|pressure|document|policy|how|investigat|recommend|evidence policy",
        q,
    ):
        result.append("retrieve")
    schedule_q = re.sub(
        r"(do not|don.t|without|no)\s+(generate\s+|a\s+)?schedul\w*", "", q
    )
    if re.search(r"schedul|allocat|optimiz|next.*cycle|sequence.*job", schedule_q):
        result.append("optimize")
    return result


class Investigator:
    def __init__(self, rag=None):
        self.rag = rag
        graph = StateGraph(State)
        for name in [
            "understand",
            "plan",
            "investigate",
            "validate",
            "optimize",
            "report",
        ]:
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "understand")
        graph.add_edge("understand", "plan")
        graph.add_edge("plan", "investigate")
        graph.add_edge("investigate", "validate")
        graph.add_conditional_edges(
            "validate",
            lambda s: (
                "optimize"
                if "optimize" in s["intent"]
                and s["validated"]
                and s["request"].scheduling_problem
                else "report"
            ),
            {"optimize": "optimize", "report": "report"},
        )
        graph.add_edge("optimize", "report")
        graph.add_edge("report", END)
        self.graph = graph.compile()

    def understand(self, state):
        state.update(
            intent=route(state["request"].question),
            logs=[
                {
                    "stage": "understand",
                    "router": "bounded keyword policy; explicit dates/constraints supplied structurally",
                }
            ],
            tool_calls=[],
            errors=[],
        )
        return state

    def plan(self, state):
        state["plan"] = [
            {
                "analyze": "compare_periods",
                "retrieve": "retrieve_and_answer",
                "optimize": "solve_schedule",
            }[x]
            for x in state["intent"]
        ]
        state["logs"].append({"stage": "plan", "tools": state["plan"]})
        return state

    def call(self, state, name, inputs, fn):
        start = time.perf_counter()
        try:
            value = fn()
            state["tool_calls"].append(
                {
                    "tool": name,
                    "inputs": inputs,
                    "status": "ok",
                    "milliseconds": 1000 * (time.perf_counter() - start),
                    "output": value,
                }
            )
            return value
        except (OSError, ValueError, RuntimeError) as exc:
            state["errors"].append(f"{name}: {exc}")
            state["tool_calls"].append(
                {
                    "tool": name,
                    "inputs": inputs,
                    "status": "error",
                    "error": str(exc),
                    "milliseconds": 1000 * (time.perf_counter() - start),
                }
            )
            return None

    def investigate(self, state):
        req = state["request"]
        if "analyze" in state["intent"]:
            a = Period(req.previous_start, req.previous_end)
            b = Period(req.current_start, req.current_end)
            state["metrics"] = self.call(
                state,
                "compare_periods",
                {
                    "previous": [a.start, a.end],
                    "current": [b.start, b.end],
                    "database": req.database,
                },
                lambda: compare(req.database, a, b),
            )
        if "retrieve" in state["intent"]:

            def retrieve():
                if self.rag is None:
                    from knowledge_assistant.generation import RAG

                    self.rag = RAG()
                return self.rag.answer(req.question)

            state["knowledge"] = self.call(
                state, "retrieve_and_answer", {"question": req.question}, retrieve
            )
        state["logs"].append({"stage": "investigate"})
        return state

    def validate(self, state):
        good = not state["errors"]
        if "analyze" in state["intent"] and not (state.get("metrics") or {}).get(
            "sufficient"
        ):
            good = False
            state["errors"].append(
                "Insufficient or contradictory production evidence; scheduling blocked for this investigation"
            )
        if (
            "optimize" in state["intent"]
            and state["request"].scheduling_problem is None
        ):
            state["errors"].append(
                "Scheduling requires explicit validated jobs/machines/constraints; none supplied"
            )
        state["validated"] = good
        state["logs"].append({"stage": "validate", "valid": good})
        return state

    def optimize(self, state):
        problem = state["request"].scheduling_problem
        state["optimization"] = self.call(
            state,
            "solve_schedule",
            problem.to_dict(),
            lambda: solve(problem, time_limit=5),
        )
        solution = state["optimization"]
        if (
            solution
            and solution["schedule"]
            and validate_schedule(problem, solution["schedule"])
        ):
            raise RuntimeError("Schedule failed validation")
        state["logs"].append({"stage": "optimize", "human_review_required": True})
        return state

    def report(self, state):
        metrics = state.get("metrics") or {}
        observations = []
        hypotheses = []
        actions = []
        limitations = list(state["errors"])
        if metrics.get("sufficient"):
            observations.append(
                {
                    "claim": "Completed quantities changed between equal-length periods",
                    "previous": metrics["previous"]["quantity"],
                    "current": metrics["current"]["quantity"],
                    "delta": metrics["delta_quantity"],
                    "percentage_change": metrics["percentage_change"],
                    "evidence": "compare_periods output",
                }
            )
            observations.append(
                {
                    "claim": "Known downtime and missingness",
                    "previous_known_minutes": metrics["previous"][
                        "known_downtime_minutes"
                    ],
                    "current_known_minutes": metrics["current"][
                        "known_downtime_minutes"
                    ],
                    "complete": metrics["downtime_comparison_complete"],
                    "evidence": "compare_periods output",
                }
            )
            if not metrics["downtime_comparison_complete"]:
                limitations.append(
                    "Downtime is incomplete: sums include known records only; missing is not zero"
                )
            if metrics["delta_quantity"] < 0:
                hypotheses.append(
                    "Logged downtime and output changes may be related; this association does not establish a cause."
                )
                actions.append(
                    "Prioritize review of the largest machine-level output decline; confirm against alarm and maintenance records."
                )
        if state.get("knowledge"):
            if state["knowledge"]["abstained"]:
                limitations.append("No sufficient document evidence for this question")
            else:
                actions.append(
                    "Inspect the cited reference passage before applying its guidance to a particular machine."
                )
        solution = state.get("optimization")
        if solution and solution.get("schedule"):
            actions.append(
                "Review the feasible proposed schedule with an authorized operator before execution."
            )
        if not state["intent"]:
            limitations.append(
                "Question outside the supported analysis, retrieval and scheduling scope"
            )
        if "analyze" in state["intent"]:
            limitations.append(
                "Production records are synthetic fixtures, not real plant observations. No causal or real-world performance claim is justified."
            )
        state["report"] = {
            "question": state["request"].question,
            "plan": state["plan"],
            "observations": observations,
            "metrics": metrics,
            "document_evidence": state.get("knowledge"),
            "hypotheses": hypotheses,
            "recommended_actions": actions,
            "schedule_proposal": solution,
            "human_review_required": bool(solution and solution.get("schedule")),
            "limitations": limitations,
            "supported": (
                False
                if "analyze" in state["intent"] and not metrics.get("sufficient")
                else bool(
                    observations
                    or (state.get("knowledge") and not state["knowledge"]["abstained"])
                    or (solution and solution.get("schedule"))
                )
            ),
        }
        state["logs"].append({"stage": "report"})
        return state

    def run(self, request, output=None):
        began = time.perf_counter()
        state = self.graph.invoke({"request": request})
        result = {
            **state["report"],
            "tool_calls": state["tool_calls"],
            "trace": state["logs"],
            "latency_ms": 1000 * (time.perf_counter() - began),
        }
        if output:
            save_json(output, result)
        return result


if __name__ == "__main__":
    import argparse, json
    from common import load_json

    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--with-schedule", action="store_true")
    p.add_argument("--output", default="results/agent/example.json")
    a = p.parse_args()
    problem = (
        Problem.from_dict(load_json("configs/demo_schedule.json"))
        if a.with_schedule
        else None
    )
    result = Investigator().run(
        Request(a.question, scheduling_problem=problem), a.output
    )
    print(json.dumps(result, indent=2))

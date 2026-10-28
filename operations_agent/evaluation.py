import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from common import load_json, save_json
from scripts.prepare_data import demo_problem
from knowledge_assistant.generation import RAG
from operations_agent.workflow import Investigator, Request
from operations_agent.data_analysis import plot_comparison
from optimization.visualization import gantt

CASES = [
    ("a01", "Compare production throughput", ["compare_periods"], True, {}),
    (
        "a02",
        "Summarize machine utilization and missing downtime",
        ["compare_periods"],
        True,
        {},
    ),
    (
        "a03",
        "How should compressed air leaks be detected?",
        ["retrieve_and_answer"],
        True,
        {},
    ),
    ("a04", "Optimize the job schedule", ["solve_schedule"], True, {"schedule": True}),
    (
        "a05",
        "Investigate production throughput and recommend a schedule",
        ["compare_periods", "retrieve_and_answer", "solve_schedule"],
        True,
        {"schedule": True},
    ),
    ("a06", "Compare production, do not schedule", ["compare_periods"], True, {}),
    (
        "a07",
        "Investigate production for the empty period",
        ["compare_periods", "retrieve_and_answer"],
        False,
        {"empty": True},
    ),
    ("a08", "Tell me a joke", [], False, {}),
    ("a09", "Schedule the jobs", [], False, {}),
    (
        "a10",
        "What is the safe pressure rating of compressor ZX-99?",
        ["retrieve_and_answer"],
        False,
        {},
    ),
    (
        "a11",
        "Why did production output decline?",
        ["compare_periods", "retrieve_and_answer"],
        True,
        {},
    ),
    (
        "a12",
        "Investigate production and optimize next cycle",
        ["compare_periods", "retrieve_and_answer", "solve_schedule"],
        True,
        {"schedule": True},
    ),
]


def evaluate(output="results/agent"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rag = RAG()
    rag.answer("What is the scheduling objective?")
    agent = Investigator(rag)
    reference = load_json("data/fixture_reference.json")["reference"]
    rows = []
    reports = []
    for cid, question, expected, supported, options in CASES:
        kwargs = {}
        if options.get("schedule"):
            kwargs["scheduling_problem"] = demo_problem()
        if options.get("empty"):
            kwargs.update(
                previous_start="2025-01-01",
                previous_end="2025-01-07",
                current_start="2025-01-08",
                current_end="2025-01-14",
            )
        report = agent.run(Request(question, **kwargs))
        actual = [c["tool"] for c in report["tool_calls"]]
        numerical = None
        if report["metrics"].get("sufficient"):
            m = report["metrics"]
            numerical = (
                m["previous"]["quantity"] == reference["previous"]["quantity"]
                and m["current"]["quantity"] == reference["current"]["quantity"]
                and m["delta_quantity"] == reference["delta_quantity"]
                and m["current"]["known_downtime_minutes"]
                == reference["current"]["known_downtime"]
            )
        solution = report["schedule_proposal"]
        knowledge = report["document_evidence"]
        citations = [] if not knowledge else knowledge["citations"]
        citation_ok = all(
            c["quote"].lower()
            in next(
                p["text"] for p in rag.retriever.chunks if p["id"] == c["chunk_id"]
            ).lower()
            for c in citations
        )
        grounded = all(
            o.get("evidence") == "compare_periods output"
            for o in report["observations"]
        ) and all(
            "cause" not in h.lower() or "not establish" in h
            for h in report["hypotheses"]
        )
        rows.append(
            {
                "id": cid,
                "question": question,
                "expected_tools": ",".join(expected),
                "actual_tools": ",".join(actual),
                "tool_selection_correct": actual == expected,
                "numeric_reference_correct": numerical,
                "support_decision_correct": report["supported"] == supported,
                "grounding_structure_correct": grounded,
                "citation_span_correct": citation_ok if citations else None,
                "schedule_feasible": (
                    solution["constraint_violations"] == 0
                    if solution and solution.get("schedule")
                    else None
                ),
                "unnecessary_tool_calls": len(set(actual) - set(expected)),
                "latency_ms": report["latency_ms"],
            }
        )
        reports.append({"id": cid, **report})
        save_json(out / f"{cid}.json", report)
        print(cid, actual, "supported=", report["supported"], flush=True)
        if cid == "a05":
            save_json(out / "example.json", report)
            plot_comparison(report["metrics"], out / "production_comparison.png")
            gantt(demo_problem(), solution, out / "proposed_schedule.png")
    df = pd.DataFrame(rows)
    df.to_csv(out / "evaluation.csv", index=False)
    metrics = {
        "cases": len(df),
        "tool_selection_accuracy": float(
            accuracy_score(df.expected_tools, df.actual_tools)
        ),
        "numeric_reference_accuracy": float(
            df.numeric_reference_correct.dropna().mean()
        ),
        "support_decision_accuracy": float(df.support_decision_correct.mean()),
        "grounding_structure_accuracy": float(df.grounding_structure_correct.mean()),
        "citation_span_accuracy": float(df.citation_span_correct.dropna().mean()),
        "schedule_feasibility_rate": float(df.schedule_feasible.dropna().mean()),
        "unnecessary_tool_calls": int(df.unnecessary_tool_calls.sum()),
        "median_warm_latency_ms": float(df.latency_ms.median()),
        "p95_warm_latency_ms": float(df.latency_ms.quantile(0.95)),
        "methodology": "12 authored routing/support scenarios. Numeric references are direct Python sums of original generator records, independent of investigative SQL/Pandas calculations. Grounding is structural evidence tracing and explicit causal caveat checks, not independent semantic adjudication. Model startup excluded.",
    }
    save_json(out / "summary.json", metrics)
    print(metrics)
    return metrics


if __name__ == "__main__":
    evaluate()

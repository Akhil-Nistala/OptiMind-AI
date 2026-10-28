import pytest
from operations_agent.workflow import Investigator, Request, route
from scripts.prepare_data import prepare, demo_problem


class StubRAG:
    def answer(self, q):
        return {"abstained": True, "citations": [], "answer": "Insufficient evidence"}


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Compare production throughput", ["analyze"]),
        ("How should compressed air leaks be detected?", ["retrieve"]),
        ("Optimize the job schedule", ["optimize"]),
        (
            "Investigate production and recommend a schedule",
            ["analyze", "retrieve", "optimize"],
        ),
        ("Compare production, do not schedule", ["analyze"]),
        ("Tell me a joke", []),
    ],
)
def test_tool_selection(question, expected):
    assert route(question) == expected


def test_integrated_agent_calls_and_numeric_evidence(tmp_path):
    db = tmp_path / "production.sqlite"
    prepare(db)
    result = Investigator(StubRAG()).run(
        Request(
            "Investigate throughput and recommend a schedule",
            database=str(db),
            scheduling_problem=demo_problem(),
        )
    )
    assert [c["tool"] for c in result["tool_calls"]] == [
        "compare_periods",
        "retrieve_and_answer",
        "solve_schedule",
    ]
    assert result["supported"] and result["human_review_required"]
    assert result["schedule_proposal"]["constraint_violations"] == 0
    assert any("association" in h for h in result["hypotheses"])
    assert not result["metrics"]["downtime_comparison_complete"]


def test_missing_data_blocks_dependent_schedule(tmp_path):
    db = tmp_path / "production.sqlite"
    prepare(db)
    result = Investigator(StubRAG()).run(
        Request(
            "Investigate production and schedule jobs",
            database=str(db),
            previous_start="2025-01-01",
            previous_end="2025-01-07",
            current_start="2025-01-08",
            current_end="2025-01-14",
            scheduling_problem=demo_problem(),
        )
    )
    assert not result["supported"]
    assert not result["schedule_proposal"]
    assert "solve_schedule" not in [c["tool"] for c in result["tool_calls"]]

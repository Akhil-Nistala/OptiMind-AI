import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from common import load_json
from optimization.problem import Problem
from optimization.greedy_baseline import greedy
from optimization.solver import solve
from optimization.visualization import gantt
from operations_agent.workflow import Investigator, Request

st.set_page_config(page_title="OptiMind AI", layout="wide")
st.title("OptiMind AI")
st.caption(
    "Industrial decision support | schedules, cited knowledge and evidence-driven investigation"
)
section = st.sidebar.radio(
    "Workspace", ["Scheduling", "Knowledge Assistant", "Operations Investigator"]
)
st.sidebar.info(
    "Production records and local procedures are synthetic fixtures. Proposed schedules require human review."
)


@st.cache_resource
def get_rag():
    from knowledge_assistant.generation import RAG

    return RAG()


def render_knowledge(result):
    if result["abstained"]:
        st.warning(result["answer"])
    else:
        st.write(result["answer"])
        st.caption(f"Answer mode: {result['answer_mode']}")
        for citation in result["citations"]:
            page = (
                f", PDF page {citation['page']}"
                if citation["page"]
                else ", text document (no page number)"
            )
            st.write(citation["title"] + page)
            st.code(citation["quote"], language=None)
            if citation["source_url"]:
                st.link_button("Open source", citation["source_url"])
            else:
                st.caption("Local source: " + citation["path"])
    for passage in result.get("passages", []):
        with st.expander(passage["title"] + f" | chunk {passage['id']}"):
            st.write(passage["text"])
            st.json(
                {
                    k: passage[k]
                    for k in ["page", "path", "source_url", "score", "semantic_score"]
                }
            )
    with st.expander("Raw local model draft and validation"):
        st.write(result.get("raw_model_output"))
        st.json(result.get("supporting_evidence", []))


if section == "Scheduling":
    st.header("Schedule jobs")
    default = (
        load_json("configs/demo_schedule.json")
        if Path("configs/demo_schedule.json").exists()
        else {
            "machines": 1,
            "crew_capacity": 1,
            "jobs": [{"id": "A", "operations": [{"machine": 0, "duration": 5}]}],
        }
    )
    payload = st.text_area(
        "Jobs and machine constraints (JSON)", json.dumps(default, indent=2), height=330
    )
    time_limit = st.number_input(
        "Solver time limit (seconds)", min_value=0.1, max_value=60.0, value=2.0
    )
    if st.button("Compare schedules"):
        try:
            problem = Problem.from_dict(json.loads(payload))
            baseline = greedy(problem)
            optimized = solve(problem, time_limit)
            for col, name, result in zip(
                st.columns(2), ["Greedy", "CP-SAT"], [baseline, optimized]
            ):
                with col:
                    st.subheader(name)
                    st.write(result["status"])
                    if result["schedule"]:
                        st.metric("Makespan", result["makespan"])
                        st.write("Weighted tardiness:", result["weighted_tardiness"])
                        st.pyplot(gantt(problem, result))
                        st.dataframe(result["schedule"])
                    else:
                        st.warning(
                            "No schedule available. This heuristic failure/solver status is not automatically a proof of infeasibility."
                        )
                    st.caption(
                        f"Runtime {result['runtime_seconds']:.3f}s | proved optimal: {result['proven_optimal']}"
                    )
            st.info(
                "Proposal only. Review job durations, crew qualifications, materials and maintenance windows before operational use."
            )
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            st.error(f"Invalid scheduling input or solver failure: {exc}")

elif section == "Knowledge Assistant":
    st.header("Ask the indexed technical collection")
    question = st.text_input("Question", "What is the scheduling objective?")
    if st.button("Find cited answer"):
        try:
            with st.spinner("Retrieving evidence and running the local model..."):
                result = get_rag().answer(question)
            render_knowledge(result)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(
                f"Knowledge system unavailable: {exc}. Run asset download, ingestion and index preparation commands in README."
            )

else:
    st.header("Investigate operations")
    question = st.text_area(
        "Operational question",
        "Investigate production throughput and recommend a schedule for the next cycle.",
    )
    cols = st.columns(2)
    with cols[0]:
        previous_start = st.text_input("Previous period start", "2026-10-05")
        previous_end = st.text_input("Previous period end", "2026-10-11")
    with cols[1]:
        current_start = st.text_input("Current period start", "2026-10-12")
        current_end = st.text_input("Current period end", "2026-10-18")
    schedule = st.checkbox(
        "Supply the example next-cycle scheduling instance", value=True
    )
    if st.button("Investigate"):
        try:
            problem = (
                Problem.from_dict(load_json("configs/demo_schedule.json"))
                if schedule
                else None
            )
            request = Request(
                question,
                previous_start=previous_start,
                previous_end=previous_end,
                current_start=current_start,
                current_end=current_end,
                scheduling_problem=problem,
            )
            with st.spinner("Executing the evidence workflow..."):
                from operations_agent.workflow import route

                rag = get_rag() if "retrieve" in route(question) else None
                report = Investigator(rag).run(request)
            st.subheader("Observed facts")
            st.json(report["observations"])
            st.subheader("Hypotheses")
            st.write(report["hypotheses"])
            st.subheader("Recommended actions")
            st.write(report["recommended_actions"])
            if report["document_evidence"]:
                render_knowledge(report["document_evidence"])
            if report["schedule_proposal"] and report["schedule_proposal"].get(
                "schedule"
            ):
                st.subheader("Schedule proposal - human review required")
                st.pyplot(gantt(problem, report["schedule_proposal"]))
            st.subheader("Limitations")
            st.write(report["limitations"])
            with st.expander("Calculated metrics, tool calls and execution trace"):
                st.json(report)
            st.download_button(
                "Download investigation JSON",
                json.dumps(report, indent=2),
                "investigation.json",
                "application/json",
            )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            st.error(f"Investigation could not complete: {exc}")

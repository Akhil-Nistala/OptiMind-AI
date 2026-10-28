import json
from pathlib import Path
from streamlit.testing.v1 import AppTest


def app():
    return AppTest.from_file("app/streamlit_app.py").run(timeout=90)


def test_scheduling_interface_runs_solver():
    ui = app()
    assert not ui.exception
    ui.button[0].click().run(timeout=90)
    assert not ui.exception and not ui.error
    assert any(metric.label == "Makespan" for metric in ui.metric)


def test_invalid_schedule_is_explained():
    ui = app()
    ui.text_area[0].set_value("{invalid json")
    ui.button[0].click().run(timeout=90)
    assert not ui.exception and ui.error


def test_analysis_only_interface_uses_live_tools():
    if not Path("data/production.sqlite").exists():
        import pytest

        pytest.skip("Run data preparation for UI integration")
    ui = app()
    ui.radio[0].set_value("Operations Investigator").run(timeout=90)
    ui.text_area[0].set_value("Compare production throughput")
    ui.checkbox[0].set_value(False)
    ui.button[0].click().run(timeout=90)
    assert not ui.exception and not ui.error
    assert any("Observed facts" in header.value for header in ui.subheader)


def test_knowledge_interface_handles_local_model():
    if not Path("models/flan-t5-small/model.safetensors").exists():
        import pytest

        pytest.skip("Run model download for local LLM UI integration")
    ui = app()
    ui.radio[0].set_value("Knowledge Assistant").run(timeout=90)
    ui.button[0].click().run(timeout=90)
    assert not ui.exception and not ui.error
    assert any("minimum makespan" in text.value.lower() for text in ui.markdown)

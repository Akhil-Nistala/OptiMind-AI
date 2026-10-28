import sqlite3
import pandas as pd
import pytest
from common import load_json
from scripts.prepare_data import prepare
from operations_agent.data_analysis import Period, compare, summarize
from database.queries import readonly_connection, read_period


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "production.sqlite"
    prepare(path)
    return path


def test_calculations_against_independent_generator_reference(db):
    result = compare(
        db, Period("2026-10-05", "2026-10-11"), Period("2026-10-12", "2026-10-18")
    )
    ref = load_json("data/fixture_reference.json")["reference"]
    assert result["sufficient"]
    for period in ["previous", "current"]:
        assert result[period]["quantity"] == ref[period]["quantity"]
        assert result[period]["known_downtime_minutes"] == ref[period]["known_downtime"]
    assert result["delta_quantity"] == ref["delta_quantity"]
    assert (
        not result["downtime_comparison_complete"]
        and result["current"]["missing_values"]["downtime_minutes"] == 1
    )


def test_readonly_database(db):
    with readonly_connection(db) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM production")


def test_empty_period_and_invalid_dates(db):
    result = compare(
        db, Period("2025-01-01", "2025-01-07"), Period("2025-01-08", "2025-01-14")
    )
    assert not result["sufficient"]
    with pytest.raises(ValueError):
        Period("2026-10-12", "2026-10-05")
    with pytest.raises(ValueError):
        compare(
            db, Period("2026-10-05", "2026-10-11"), Period("2026-10-11", "2026-10-17")
        )


def test_contradictory_records(db):
    frame = read_period(db, "2026-10-05", "2026-10-11")
    frame.loc[0, "actual_minutes"] = 500
    result = summarize(frame, Period("2026-10-05", "2026-10-11"))
    assert not result["sufficient"] and result["contradictions"]


def test_missing_machine_not_hidden_by_aggregation(db):
    with sqlite3.connect(db) as connection:
        connection.execute("DELETE FROM production WHERE machine_id='M2'")
    result = compare(
        db, Period("2026-10-05", "2026-10-11"), Period("2026-10-12", "2026-10-18")
    )
    assert not result["sufficient"] and "Missing machine/date" in result["reason"]

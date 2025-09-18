import pytest
from optswmm.utils.runutils import summarize_runs, initialize_run

def test_initialize_run():
    # Test the initialization of a run
    run = initialize_run("test_run")
    assert run is not None
    assert run.name == "test_run"

def test_summarize_runs():
    # Test summarizing runs with dummy data
    runs = [
        {"name": "run1", "score": 0.9},
        {"name": "run2", "score": 0.85},
        {"name": "run3", "score": 0.95},
    ]
    summary = summarize_runs(runs)
    assert len(summary) == 3
    assert summary[0]["name"] == "run1"
    assert summary[0]["score"] == 0.9
    assert summary[1]["name"] == "run2"
    assert summary[1]["score"] == 0.85
    assert summary[2]["name"] == "run3"
    assert summary[2]["score"] == 0.95
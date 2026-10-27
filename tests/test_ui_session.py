"""Tests for the Streamlit session layer (chunk 8).

Streamlit itself is optional and not imported here; only the plain-Python
state holder is exercised, so the UI logic stays testable offline.
"""

from __future__ import annotations

import pytest

from latent_workspace.config import Settings
from latent_workspace.ui import UIError, WorkspaceSession

QUERY = "Why did global chip prices rise in 2025?"


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, mock_mode=True, data_dir=tmp_path)


def test_session_runs_and_exposes_view_rows(settings):
    session = WorkspaceSession(settings)
    result = session.ask(QUERY)

    assert session.last_result is result
    assert result.final_answer
    summary = session.summary()
    assert summary["mode"] == "mock (offline)"
    assert summary["nodes"] == session.graph.node_count
    assert summary["runs"] == 1

    graph_rows = session.graph_rows()
    assert len(graph_rows) == session.graph.node_count
    assert {"id", "type", "agent", "confidence", "content"} <= set(graph_rows[0])

    edge_rows = session.edge_rows()
    assert len(edge_rows) == session.graph.edge_count
    if edge_rows:
        assert {"source", "relationship", "target", "confidence"} <= set(edge_rows[0])

    assert session.subtask_rows(result)
    assert session.context_rows(result)


def test_blank_query_rejected(settings):
    session = WorkspaceSession(settings)
    with pytest.raises(UIError, match="non-blank"):
        session.ask("  ")


def test_save_reset_and_load_round_trip(settings):
    session = WorkspaceSession(settings)
    session.ask(QUERY)
    nodes, edges = session.graph.node_count, session.graph.edge_count
    assert nodes > 0

    path = session.save()
    assert path.exists()
    assert path.with_suffix(".faiss").exists()

    session.reset()
    assert session.graph.node_count == 0
    assert session.history == []
    assert session.last_result is None

    session.load(path)
    assert session.graph.node_count == nodes
    assert session.graph.edge_count == edges
    assert session.history == []

    # restored workspace is still usable
    result = session.ask("What evidence supports the hypothesis?")
    assert result.final_answer


def test_load_missing_snapshot_is_a_ui_error(settings):
    session = WorkspaceSession(settings)
    with pytest.raises(UIError, match="not found"):
        session.load(settings.data_dir / "missing.json")

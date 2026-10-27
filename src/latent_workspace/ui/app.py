"""Streamlit frontend (chunk 8).

Run with::

    streamlit run app.py

The heavy lifting lives in :class:`latent_workspace.ui.session.WorkspaceSession`;
this module is only presentation. It imports Streamlit lazily so the rest
of the project (and the test suite) never needs the optional dependency.
"""

from __future__ import annotations

from ..config import get_settings
from ..models import OrchestratorResult
from .session import UIError, WorkspaceSession


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "Streamlit is not installed. Install the UI extra with:\n"
            "  python -m pip install -e \".[ui]\""
        ) from exc

    st.set_page_config(
        page_title="Dynamic Relational Latent Workspace",
        layout="wide",
    )

    st.title("Dynamic Relational Latent Workspace")
    st.caption(
        "Agents exchange a typed, confidence-labeled subgraph instead of "
        "raw chat history. Relations are predictions, not verified truth."
    )

    if "session" not in st.session_state:
        st.session_state.session = WorkspaceSession()
    session: WorkspaceSession = st.session_state.session

    _sidebar(st, session)

    with st.form("ask_form"):
        query = st.text_area(
            "Question",
            value="Why did global chip prices rise in 2025?",
            height=90,
            help="The run is fully offline while DLRW_MOCK_MODE=true.",
        )
        max_rounds = st.slider(
            "Worker rounds",
            min_value=0,
            max_value=max(1, get_settings().max_rounds),
            value=get_settings().max_rounds,
            help="0 = plan + synthesize using hypotheses only.",
        )
        submitted = st.form_submit_button("Run reasoning", type="primary")

    if submitted:
        try:
            with st.spinner("Planning, retrieving subgraphs, researching, critiquing..."):
                session.ask(query, max_rounds=max_rounds)
        except UIError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - keep the UI alive
            st.error(f"{type(exc).__name__}: {exc}")

    result = session.last_result
    if result is not None:
        _render_result(st, session, result)
    else:
        st.info("Ask a question to start a run. The workspace is empty right now.")

    _render_workspace(st, session)


# ---------------------------------------------------------------------- #
# sidebar
# ---------------------------------------------------------------------- #
def _sidebar(st, session: WorkspaceSession) -> None:
    flash = st.session_state.pop("flash", None)
    if flash:
        mapping = {
            "success": st.sidebar.success,
            "warning": st.sidebar.warning,
            "error": st.sidebar.error,
        }
        mapping.get(flash[0], st.sidebar.info)(flash[1])

    with st.sidebar:
        st.header("Workspace")
        summary = session.summary()
        st.write(
            {
                "mode": summary["mode"],
                "model": summary["model"],
                "embedder": summary["embedder"],
                "nodes": summary["nodes"],
                "edges": summary["edges"],
                "runs": summary["runs"],
            }
        )

        st.divider()
        if st.button("New empty workspace", use_container_width=True):
            session.reset()
            st.session_state["flash"] = ("success", "Workspace reset.")
            st.rerun()

        if st.button("Save snapshot", use_container_width=True):
            try:
                path = session.save()
                st.session_state["flash"] = ("success", f"Saved {path}")
            except Exception as exc:  # noqa: BLE001
                st.session_state["flash"] = ("error", f"Save failed: {exc}")
            st.rerun()

        if st.button("Load snapshot", use_container_width=True):
            try:
                session.load()
                st.session_state["flash"] = ("success", "Snapshot loaded.")
            except Exception as exc:  # noqa: BLE001
                st.session_state["flash"] = ("error", f"Load failed: {exc}")
            st.rerun()

        st.divider()
        st.caption(
            "Snapshots are written to the configured data directory. "
            "Graph edges are predicted relations with confidence scores."
        )


# ---------------------------------------------------------------------- #
# run rendering
# ---------------------------------------------------------------------- #
def _render_result(st, session: WorkspaceSession, result: OrchestratorResult) -> None:
    st.subheader("Final answer")
    st.markdown(result.final_answer or "_(empty answer)_")

    grounded = result.extra.get("answer_grounded")
    if grounded:
        st.caption("Every cited unit exists in the workspace.")
    elif result.extra.get("answer_citations"):
        st.caption("Some citations are not workspace units — inspect the trace.")

    m = result.metrics
    cols = st.columns(5)
    cols[0].metric("Latency", f"{m.latency_ms:.0f} ms")
    cols[1].metric("Tokens", m.total_tokens)
    cols[2].metric("Agent calls", m.agent_calls)
    cols[3].metric("Nodes added", m.nodes_added)
    cols[4].metric("Edges added", m.edges_added)

    if result.errors:
        with st.expander(f"Errors / warnings ({len(result.errors)})", expanded=False):
            for error in result.errors:
                st.warning(error)

    trace_tab, context_tab, history_tab = st.tabs(
        ["Run trace", "Retrieved contexts", "History"]
    )

    with trace_tab:
        st.markdown("**Subtasks**")
        rows = session.subtask_rows(result)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No subtasks were produced.")

        st.markdown("**Units produced in this run**")
        if result.units:
            st.dataframe(
                [
                    {
                        "id": u.id,
                        "type": u.information_type.value,
                        "agent": u.agent_id.value,
                        "confidence": round(u.confidence, 2),
                        "content": u.content,
                    }
                    for u in result.units
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No units were produced.")

        st.markdown("**Edges added in this run**")
        if result.edges:
            lookup = {u.id: u for u in session.graph.nodes()}
            st.dataframe(
                [
                    {
                        "source": lookup.get(e.source_node_id, e.source_node_id).content
                        if e.source_node_id in lookup
                        else e.source_node_id,
                        "relationship": e.relationship_type.value,
                        "target": lookup.get(e.target_node_id, e.target_node_id).content
                        if e.target_node_id in lookup
                        else e.target_node_id,
                        "confidence": round(e.confidence, 2),
                        "rationale": e.evidence_or_rationale,
                    }
                    for e in result.edges
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No edges were added (the graph may be too small yet).")

    with context_tab:
        rows = session.context_rows(result)
        if not rows:
            st.info("No workspace context reached the agents in this run.")
        else:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            index = st.selectbox(
                "Inspect a context block",
                options=range(len(result.contexts)),
                format_func=lambda i: (
                    f"Context {i + 1} — {result.contexts[i].task_description[:70]}"
                ),
            )
            ctx = result.contexts[index]
            st.code(ctx.as_prompt_block(), language="text")
            st.caption(
                f"token_estimate={ctx.token_estimate}, truncated={ctx.truncated}, "
                f"supporting={len(ctx.supporting_ids)}, "
                f"contradicting={len(ctx.contradicting_ids)}"
            )

    with history_tab:
        if not session.history:
            st.info("No runs yet.")
        for i, past in enumerate(reversed(session.history), start=1):
            with st.expander(
                f"{len(session.history) - i + 1}. {past.query[:80]}",
                expanded=False,
            ):
                st.markdown(past.final_answer or "_(empty)_")
                st.caption(
                    f"tokens={past.metrics.total_tokens}, "
                    f"nodes+={past.metrics.nodes_added}, "
                    f"edges+={past.metrics.edges_added}"
                )


# ---------------------------------------------------------------------- #
# workspace browser
# ---------------------------------------------------------------------- #
def _render_workspace(st, session: WorkspaceSession) -> None:
    st.divider()
    st.subheader("Workspace graph")
    st.caption(
        f"{session.graph.node_count} nodes, {session.graph.edge_count} edges. "
        "Edges read source --RELATIONSHIP--> target."
    )
    nodes_tab, edges_tab = st.tabs(["Nodes", "Edges"])

    with nodes_tab:
        rows = session.graph_rows()
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("The workspace is empty.")

    with edges_tab:
        rows = session.edge_rows()
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No typed edges yet.")

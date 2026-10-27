"""Graph package: the Dynamic Relational Latent Workspace.

    from latent_workspace.graph import build_workspace
    ws = build_workspace()                       # mock components if mock mode
    result = ws.add_unit(some_information_unit)  # embed -> candidates ->
                                                 # classify -> threshold -> commit
    support = ws.find_supporting(result.unit_id)
    chain   = ws.provenance(result.unit_id)
    ws.save("data/graph.json")
"""

from __future__ import annotations

from .workspace import (
    AddUnitResult,
    KnowledgeGraph,
    KnowledgeGraphError,
    ProvenanceStep,
    build_workspace,
)

__all__ = [
    "AddUnitResult",
    "KnowledgeGraph",
    "KnowledgeGraphError",
    "ProvenanceStep",
    "build_workspace",
]

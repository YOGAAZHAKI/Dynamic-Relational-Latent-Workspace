"""Optional Streamlit frontend (chunk 8).

The frontend is split in two: :class:`WorkspaceSession` is a plain-Python,
unit-testable state holder; :func:`latent_workspace.ui.app.main` renders it
with Streamlit when the optional dependency is installed.
"""

from __future__ import annotations

from .session import UIError, WorkspaceSession

__all__ = ["UIError", "WorkspaceSession"]

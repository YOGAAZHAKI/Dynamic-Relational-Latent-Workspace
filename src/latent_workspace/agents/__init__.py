"""Agent package: the four specialists + a factory.

    from latent_workspace.agents import build_agents
    agents = build_agents()            # uses mock or real LLM per settings
    plan   = agents.planner.run("Why did ... ?")
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from ..llm.base import LLMClient
from ..llm import build_llm
from .base import BaseAgent
from .critic import CriticAgent
from .planner import PlannerAgent
from .researcher import ResearcherAgent
from .synthesizer import SynthesizerAgent


@dataclass(frozen=True)
class AgentSet:
    """All four agents sharing ONE LLM client (so token usage pools)."""

    planner: PlannerAgent
    researcher: ResearcherAgent
    critic: CriticAgent
    synthesizer: SynthesizerAgent

    def all(self) -> tuple[BaseAgent, ...]:
        return (self.planner, self.researcher, self.critic, self.synthesizer)


def build_agents(llm: LLMClient | None = None, settings: Settings | None = None) -> AgentSet:
    """Construct the four agents around a single LLM client."""
    s = settings or get_settings()
    client = llm if llm is not None else build_llm(s)
    return AgentSet(
        planner=PlannerAgent(client, s),
        researcher=ResearcherAgent(client, s),
        critic=CriticAgent(client, s),
        synthesizer=SynthesizerAgent(client, s),
    )


__all__ = [
    "AgentSet",
    "BaseAgent",
    "CriticAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "SynthesizerAgent",
    "build_agents",
]

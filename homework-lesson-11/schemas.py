from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Структурований план дослідження, який повертає Planner Agent."""

    goal: str = Field(description="Що саме ми намагаємось з'ясувати")
    search_queries: list[str] = Field(description="Конкретні пошукові запити для виконання")
    sources_to_check: list[str] = Field(
        description="Які джерела перевірити: 'knowledge_base', 'web', або обидва"
    )
    output_format: str = Field(description="Як має виглядати фінальний звіт")


class CritiqueResult(BaseModel):
    """Структурована оцінка дослідження, яку повертає Critic Agent."""

    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(description="Чи базуються знахідки на актуальних, свіжих джерелах?")
    is_complete: bool = Field(description="Чи повністю дослідження покриває оригінальний запит?")
    is_well_structured: bool = Field(description="Чи логічно організовані знахідки, готові стати звітом?")
    strengths: list[str] = Field(description="Що добре в дослідженні")
    gaps: list[str] = Field(description="Що відсутнє, застаріле або погано структуроване")
    revision_requests: list[str] = Field(
        description="Конкретні речі для виправлення, якщо verdict — REVISE"
    )

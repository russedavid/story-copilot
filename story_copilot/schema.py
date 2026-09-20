from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn: int = Field(ge=1)
    quote: str = Field(min_length=1)


class EventCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[
        "entity", "fact", "claim", "resource", "action", "resolve", "knowledge"
    ]
    entity: str = Field(min_length=1, max_length=160)
    attribute: str = Field(min_length=1, max_length=160)
    value: str | int | float | bool | None
    delta: int | None = None
    stage: Literal[
        "hypothetical", "declared", "requested", "reported", "established"
    ] = "established"
    visibility: Literal["public", "private"] = "public"
    evidence: list[Evidence] = Field(min_length=1, max_length=12)
    resolves: str | None = None
    supersedes: str | None = None
    rationale: str = Field(default="", max_length=3000)


class Event(EventCandidate):
    @model_validator(mode="after")
    def coherent(self):
        if self.kind != "resource" and self.value is None:
            raise ValueError(
                "State proposals need an explicit value; put unknown facts in uncertainties."
            )
        if self.kind == "action" and self.stage == "established":
            raise ValueError(
                "An action is pending or hypothetical. Describe a completed effect as a fact or resolution."
            )
        if self.kind == "resource":
            if self.stage != "established":
                raise ValueError("A resource change needs an established outcome.")
            if (self.delta is None) == (self.value is None):
                raise ValueError("Specify either an integer resource total or a delta.")
            if self.value is not None and type(self.value) is not int:
                raise ValueError("Resource totals must be integers.")
        elif self.delta is not None:
            raise ValueError("Only resource events can have a delta.")
        if (
            self.kind in {"entity", "fact", "knowledge", "resolve"}
            and self.stage != "established"
        ):
            raise ValueError("This kind of state update needs an established outcome.")
        if self.kind == "resolve" and not self.resolves:
            raise ValueError("A resolution must identify an existing pending action.")
        if self.kind != "resolve" and self.resolves:
            raise ValueError("Only resolution events can resolve an action.")
        return self


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[EventCandidate] = Field(default_factory=list, max_length=50)
    uncertainties: list[str] = Field(default_factory=list, max_length=30)


class NarrationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    narration: str = Field(min_length=1)
    questions: list[str] = Field(default_factory=list)
    requested_checks: list[str] = Field(default_factory=list)
    private_notes: str = ""

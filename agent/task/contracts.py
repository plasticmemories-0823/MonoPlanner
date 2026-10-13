"""Validated operations and work requests; no provider calls."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Operation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    op: Literal["read", "search", "shell"]
    path: str = "."
    query: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=80, ge=1, le=2000,
                       description='read: default 2000, range 1..2000. Other operations: default 80, maximum 1000.')
    from_step: str = ""
    command: str = Field(default="", max_length=30000)
    expected_exit: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=4)

    @model_validator(mode="before")
    @classmethod
    def read_window_default(cls, value):
        if isinstance(value, dict) and value.get('op') == 'read' and 'limit' not in value:
            return {**value, 'limit': 2000}
        return value

    @model_validator(mode="after")
    def complete_parameters(self):
        if self.op != 'read' and self.limit > 1000:
            raise ValueError('Only read accepts more than 1000 lines; other operation limits are unchanged')
        if self.op == "shell" and not self.command.strip():
            raise ValueError("shell requires actual executable commands, not a task description")
        if self.op == "search" and not self.query.strip():
            raise ValueError("search requires a literal query")
        if self.from_step and self.op != "read":
            raise ValueError("Only read may refer to an earlier search")
        return self


class WorkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["inspect", "candidate", "blocked"]
    note: str = Field(min_length=1, max_length=1600)
    steps: list[Operation] = Field(default_factory=list, max_length=12)
    test_code: str = Field(default="", max_length=32768)
    test_driver: list[str] = Field(default_factory=list, max_length=32)
    regression_command: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def valid_sequence(self):
        ids = {}
        for step in self.steps:
            if step.id in ids:
                raise ValueError("Step ids must be unique")
            if step.from_step and ids.get(step.from_step) != "search":
                raise ValueError("from_step must reference an earlier search in this plan")
            ids[step.id] = step.op
        if self.kind == "inspect" and not self.steps:
            raise ValueError("inspect requires at least one concrete lookup")
        if self.kind == "blocked" and self.steps:
            raise ValueError("blocked plans must not contain executable steps")
        return self

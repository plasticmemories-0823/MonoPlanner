"""经过校验的操作与工作请求，不调用服务商。"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from agent.acceptance.execution import ExecutionCheck, EXECUTION_DESCRIPTION


def _exact_mode_schema(schema):
    """单一字面值沿用严格工具支持的 enum 表示。"""
    schema.pop("const", None)
    schema["enum"] = ["exact"]


class Operation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    op: Literal["read", "search", "shell"] = Field(
        description=(
            'read: known original ranges; search: locate evidence through RAG; '
            'shell: editing, execution and runtime diagnostics, never evidence search.'
        ),
    )
    path: str = Field(
        default=".",
        description=(
            'Authorized repository scope, $record path or host-registered '
            '$container namespace. Never widen the registered scope.'
        ),
    )
    query: str = Field(
        default="",
        description=(
            'Search text: one complete case-sensitive literal string.'
        ),
    )
    offset: int = Field(default=0, ge=0)
    limit: int = Field(
        default=80,
        ge=1,
        le=2000,
        description='read: default 2000, range 1..2000. Other operations: default 80, maximum 1000.',
    )
    mode: Literal["exact"] = Field(
        default="exact",
        json_schema_extra=_exact_mode_schema,
        description="One case-sensitive literal substring query. Preserve whitespace and repeated text; BM25 ranks matching files automatically.",
    )
    intent: Literal["locate", "diagnostic", "audit"] = "locate"
    required_entities: list[str] = Field(default_factory=list, max_length=16)
    required_kinds: list[Literal["code", "document", "tool_result", "log"]] = Field(
        default_factory=list, max_length=4,
    )
    cursor: str = Field(
        default="", max_length=8192,
        description='Opaque continuation from search or read; empty for an initial request.',
    )
    from_step: str = ""
    command: str = Field(
        default="", max_length=30000,
        description=(
            'Shell code for editing, execution or runtime diagnostics; '
            'do not search existing source, documents or saved logs here.'
        ),
    )
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
            raise ValueError(
                'Only read accepts more than 1000 lines; other operation limits are unchanged'
            )
        if self.op == "shell" and not self.command.strip():
            raise ValueError("shell requires actual executable commands, not a task description")
        if self.op == "search":
            from agent.rag.integration import validate_arguments

            validate_arguments({
                "query": self.query, "path": self.path, "mode": self.mode,
            }, records=True, containers=True)
        if self.from_step and self.op != "read":
            raise ValueError("Only read may refer to an earlier search")
        if self.cursor and (self.op == 'shell' or self.from_step):
            raise ValueError('A continuation cursor requires one explicit read or search scope')
        return self


class WorkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["inspect", "candidate", "blocked"]
    note: str = Field(min_length=1, max_length=1600)
    steps: list[Operation] = Field(default_factory=list, max_length=12)
    test_code: str = Field(default="", max_length=32768)
    test_driver: list[str] = Field(default_factory=list, max_length=32)
    execution_checks: list[ExecutionCheck] = Field(
        default_factory=list, max_length=8, description=EXECUTION_DESCRIPTION,
    )
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

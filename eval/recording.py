"""记录真实模型调用且不保存凭据，共用有上限的评测预算。"""

from dataclasses import asdict, dataclass
from copy import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
import time
import uuid

from langchain_core.messages import message_to_dict, messages_to_dict
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.model.errors import (
    ModelRequestNotSent,
    ModelResponseNotSaved,
    ModelCallError,
    is_transient_model_error,
    is_context_overflow,
)
from agent.model.client import configure_call


# 只串行处理额度预留和账本快照，不串行限制真实模型调用。
# 锁不作为 dataclass 字段，确保 JSON 快照中只保留普通数据。
_LEDGER_LOCK = RLock()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    try:
        # Windows 读取方或杀毒软件可能短暂阻止文件替换。
        # 只重试这次本地写入，不重复付费模型请求。
        for delay in (0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
            if delay:
                time.sleep(delay)
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if delay == 0.8:
                    raise
    finally:
        temporary.unlink(missing_ok=True)


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(item, secrets) for item in value]
    return value


# ============================================================
# 共享调用预算与真实用量账本
# ============================================================
@dataclass
class TokenLedger:
    """已报告的用量按事实记录；不确定用量保留额度，不能记为零。"""

    limit: int | None = 1_000_000
    reported_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    unknown_calls: int = 0
    reserved_unknown_tokens: int = 0
    calls: int = 0
    request_sequence: int = 0
    locally_blocked_calls: int = 0
    last_guard_reason: str | None = None

    def reserve(self, upper_bound, protected=0):
        with _LEDGER_LOCK:
            if (
                self.limit is not None
                and self.reported_tokens + self.reserved_unknown_tokens + upper_bound + protected
                > self.limit
            ):
                raise ModelRequestNotSent(
                    "token_budget_exhausted",
                    "Evaluation token allowance cannot cover another request.",
                )
            self.calls += 1
            self.request_sequence = max(self.request_sequence + 1, self.calls)
            # 发送前先保存：进程崩溃不能让在途请求被当作免费。
            self.unknown_calls += 1
            self.reserved_unknown_tokens += upper_bound
            return self.request_sequence

    def cancel(self, reservation):
        """只撤销本次尚未发送的请求，不影响其他正在进行的请求。"""
        with _LEDGER_LOCK:
            self.calls -= 1
            self.unknown_calls -= 1
            self.reserved_unknown_tokens -= reservation

    def settle(self, usage, reservation):
        with _LEDGER_LOCK:
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            if type(total) is not int or total < 0:
                return
            self.unknown_calls -= 1
            self.reserved_unknown_tokens -= reservation
            self.reported_tokens += total
            for name in ("input_tokens", "output_tokens"):
                value = usage.get(name)
                if type(value) is int and value >= 0:
                    setattr(self, name, getattr(self, name) + value)


# ============================================================
# 模型调用的归档与续接
# ============================================================
class RecordingModel:
    """正常 Agent 仍通过 bind_tools/invoke 调用模型，这里不另设修复循环。"""

    def __init__(
        self,
        model,
        directory,
        ledger,
        ledger_path,
        max_output_tokens=8192,
        secrets=(),
        schemas=(),
        deadline=None,
        task_token_limit=None,
        task_started_charged=None,
        report_token_reserve=0,
        report_time_reserve=0,
    ):
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.ledger_path = Path(ledger_path)
        self.max_output_tokens = max_output_tokens
        self.secrets = tuple(secrets)
        self.schemas = list(schemas)
        self.deadline = deadline
        self.task_token_limit = task_token_limit
        self.task_started_charged = (
            ledger.reported_tokens + ledger.reserved_unknown_tokens
            if task_started_charged is None
            else task_started_charged
        )
        self.report_token_reserve = report_token_reserve
        self.report_time_reserve = report_time_reserve
        self.purpose = "work"

    def bind_tools(self, tools):
        return RecordingModel(
            self.model.bind_tools(tools),
            self.directory,
            self.ledger,
            self.ledger_path,
            self.max_output_tokens,
            self.secrets,
            [convert_to_openai_tool(tool, strict=True) for tool in tools],
            self.deadline,
            self.task_token_limit,
            self.task_started_charged,
            self.report_token_reserve,
            self.report_time_reserve,
        )

    def for_purpose(self, purpose, timeout=None):
        result = copy(self)
        result.purpose = purpose
        result.model = configure_call(self.model, purpose, timeout)
        cap = getattr(result.model, "max_output_tokens", self.max_output_tokens)
        result.max_output_tokens = cap if type(cap) is int else self.max_output_tokens
        if purpose in {"finalization", "recovery", "report"}:
            result.schemas = []
        return result

    @property
    def context_envelope(self):
        value = getattr(self.model, "context_envelope", None)
        return value if isinstance(value, dict) else {}

    def _save_before_request(self):
        try:
            write_json(self.ledger_path, asdict(self.ledger))
        except OSError as exc:
            raise ModelRequestNotSent(
                "recording_error", f"请求未发送：无法保存评测账本（{type(exc).__name__}）。"
            ) from None

    def invoke(self, messages):
        reporting = self.purpose == "report"
        protected = 0 if reporting else self.report_token_reserve
        cutoff = (
            self.deadline - (0 if reporting else self.report_time_reserve)
            if self.deadline is not None
            else None
        )
        request = {"messages": messages_to_dict(messages), "tools": self.schemas}
        from agent.model.payload import request_reservation, wire_input

        reservation = request_reservation(messages, self.schemas, self.max_output_tokens)
        request["wire_input"] = wire_input(messages, self.schemas)
        options = getattr(self.model, "request_options", {})
        request.update(
            started_utc=datetime.now(timezone.utc).isoformat(),
            reserved_tokens=reservation,
            purpose=self.purpose,
            request_options=options if isinstance(options, dict) else {},
            max_output_tokens=self.max_output_tokens,
        )
        with _LEDGER_LOCK:
            if cutoff is not None and time.monotonic() >= cutoff:
                self.ledger.locally_blocked_calls += 1
                self.ledger.last_guard_reason = "task_timeout"
                self._save_before_request()
                raise ModelRequestNotSent(
                    "task_timeout", "Evaluation task wall-time allowance exhausted."
                )
            task_charged = (
                self.ledger.reported_tokens
                + self.ledger.reserved_unknown_tokens
                - self.task_started_charged
            )
            if (
                self.task_token_limit is not None
                and task_charged + reservation + protected > self.task_token_limit
            ):
                self.ledger.locally_blocked_calls += 1
                self.ledger.last_guard_reason = "task_token_allowance"
                self._save_before_request()
                raise ModelRequestNotSent(
                    "token_budget_exhausted",
                    "Evaluation per-task token allowance cannot cover another request.",
                )
            try:
                call_id = self.ledger.reserve(reservation, protected)
            except ModelRequestNotSent:
                self.ledger.locally_blocked_calls += 1
                self.ledger.last_guard_reason = "evaluation_token_allowance"
                self._save_before_request()
                raise
            stem = f"call-{call_id:04d}"
            request["call_id"] = call_id
            try:
                write_json(self.ledger_path, asdict(self.ledger))
                write_json(self.directory / f"{stem}.request.json", redact(request, self.secrets))
            except OSError as exc:
                self.ledger.cancel(reservation)
                self.ledger.locally_blocked_calls += 1
                self.ledger.last_guard_reason = "recording_error"
                try:
                    write_json(self.ledger_path, asdict(self.ledger))
                except OSError:
                    pass
                raise ModelRequestNotSent(
                    "recording_error", f"请求未发送：调用记录保存失败（{type(exc).__name__}）。"
                ) from None
        started = time.monotonic()
        try:
            response = self.model.invoke(messages)
        except ModelRequestNotSent:
            with _LEDGER_LOCK:
                self.ledger.cancel(reservation)
                self.ledger.locally_blocked_calls += 1
                try:
                    write_json(self.ledger_path, asdict(self.ledger))
                except OSError:
                    pass
            raise
        except Exception as exc:
            try:
                write_json(
                    self.directory / f"{stem}.response.json",
                    redact(
                        {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_seconds": time.monotonic() - started,
                            "usage": None,
                            "retryable": is_transient_model_error(exc),
                            "context_overflow": is_context_overflow(exc),
                        },
                        self.secrets,
                    ),
                )
            except OSError:
                pass  # 保留真实 API 错误，并将未确认的用量标为未知。
            # 通信出错时用量未知，继续保留保守预留额度。
            raise ModelCallError(
                redact(f"{type(exc).__name__}: {exc}", self.secrets),
                retryable=is_transient_model_error(exc),
                context_overflow=is_context_overflow(exc),
            ) from None
        with _LEDGER_LOCK:
            self.ledger.settle(getattr(response, "usage_metadata", None), reservation)
            try:
                write_json(
                    self.directory / f"{stem}.response.json",
                    redact(
                        {
                            "status": "ok",
                            "response": message_to_dict(response),
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        self.secrets,
                    ),
                )
                write_json(self.ledger_path, asdict(self.ledger))
            except OSError as exc:
                raise ModelResponseNotSaved(
                    response, f"API 已返回，但调用记录未保存（{type(exc).__name__}）。"
                ) from None
        return response

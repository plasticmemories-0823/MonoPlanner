"""一次任务共用的预算账本；不保存聊天记录。"""

from dataclasses import dataclass, field
from threading import RLock
import time

from agent.model.errors import ModelRequestNotSent


# Only bookkeeping is locked. Never hold this lock while invoking a model.
# Keeping the lock outside dataclasses preserves vars()/asdict() JSON support.
_BUDGET_LOCK = RLock()


@dataclass
class RequestReservation:
    """One request's immutable identity and reservation-time balance snapshot."""

    owner: int
    call_id: int
    used_steps: int
    purpose: str
    remaining_steps: int
    working_steps: int
    report_steps: int
    reported_tokens: int
    unknown_usage_steps: int
    reserved_tokens: int = 0
    active: bool = True


@dataclass
class StepBudget:
    max_steps: int = 1000
    used_steps: int = 0
    reported_tokens: int = 0
    unknown_usage_steps: int = 0
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    report_reserve: int = 0
    report_used_steps: int = 0
    report_token_reserve: int = 0
    report_time_reserve: float = 0
    unknown_reserved_tokens: int = 0
    inflight_reserved_tokens: int = 0
    request_sequence: int = 0
    transport_retries: int = 0
    transport_retry_limit: int = 2
    admission_events: list = field(default_factory=list)
    normal_fraction: float = 0.70
    milestone_stage: int = 0
    notified_roles: dict = field(default_factory=dict)
    phase_events: list = field(default_factory=list)
    cancelled: bool = False
    cancel_reason: str = ""
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        if not 0 < self.normal_fraction < 1:
            raise ValueError("normal_fraction 必须在 0 和 1 之间")

    @property
    def remaining_steps(self) -> int:
        with _BUDGET_LOCK:
            return max(0, self.max_steps - self.used_steps)

    @property
    def working_steps(self) -> int:
        with _BUDGET_LOCK:
            return max(0, self.remaining_steps - max(0, self.report_reserve - self.report_used_steps))

    def available_steps(self, purpose="work"):
        with _BUDGET_LOCK:
            if self.cancelled:
                return 0
            if purpose == "report":
                return min(self.remaining_steps, max(0, self.report_reserve - self.report_used_steps))
            return self.working_steps

    def request_cancel(self, reason="用户请求停止任务。"):
        """Stop admitting new work; already-sent requests still must settle."""
        with _BUDGET_LOCK:
            self.cancelled = True
            self.cancel_reason = str(reason) or "用户请求停止任务。"

    def remaining_time(self, purpose="work"):
        if self.timeout_seconds is None:
            return None
        reserve = self.report_time_reserve if purpose != "report" and self.report_reserve else 0
        return max(0, self.timeout_seconds - (time.monotonic() - self.started_at) - reserve)

    def check_request(self, reservation, purpose="work", own_reserved=0):
        with _BUDGET_LOCK:
            if self.cancelled:
                raise ModelRequestNotSent("task_cancelled", self.cancel_reason)
            reserve = self.report_token_reserve if purpose != "report" and self.report_reserve else 0
            if self.max_tokens is not None and (
                    self.reported_tokens + self.unknown_reserved_tokens + self.inflight_reserved_tokens
                    + reservation + reserve > self.max_tokens):
                # Only a peer's *pending* reservation can be released. Unknown
                # usage is retained, and a request cannot wait on its own ticket.
                alone = self.reported_tokens + self.unknown_reserved_tokens + own_reserved + reservation + reserve
                waiting = self.inflight_reserved_tokens > own_reserved and alone <= self.max_tokens
                status = "budget_wait" if waiting else "token_budget_exhausted"
                self.admission_events.append({"status": status, "used_steps": self.used_steps,
                    "reported_tokens": self.reported_tokens, "unknown_reserved_tokens": self.unknown_reserved_tokens,
                    "inflight_reserved_tokens": self.inflight_reserved_tokens, "own_reserved_tokens": own_reserved,
                    "additional_reservation": reservation, "report_reserve": reserve, "max_tokens": self.max_tokens})
                raise ModelRequestNotSent(status, "并行请求尚未结算，保留现场并等待余额释放。" if waiting else
                                          "任务 Token 余额不足以覆盖本次请求及预留汇报。")
            remaining = self.remaining_time(purpose)
            if remaining is not None and remaining <= 0:
                raise ModelRequestNotSent("task_timeout", "当前阶段时间额度已用完，停止新模型请求。")

    def reserve_request(self, reservation=0, purpose="work") -> RequestReservation:
        """Atomically claim one call and its known token allowance.

        The executor initially claims the call with zero tokens so it can put
        this request's unique ID in the input, then calls reserve_tokens before
        sending. Both guards are atomic, and either rejection sends no request.
        """
        if reservation < 0:
            raise ValueError("reservation 不能小于 0")
        with _BUDGET_LOCK:
            if self.cancelled:
                raise ModelRequestNotSent("task_cancelled", self.cancel_reason)
            if self.available_steps(purpose) == 0:
                exhausted = self.remaining_steps == 0
                raise ModelRequestNotSent(
                    "budget_exhausted" if exhausted else "work_budget_exhausted",
                    "本次任务的总调用次数已用完。" if exhausted else "工作调用额度已用完，剩余额度仅用于最终汇报。",
                )
            self.check_request(reservation, purpose)
            self.used_steps += 1
            self.request_sequence = max(self.request_sequence + 1, self.used_steps)
            if purpose == "report":
                self.report_used_steps += 1
            self.inflight_reserved_tokens += reservation
            return RequestReservation(
                id(self), self.request_sequence, self.used_steps, purpose, self.remaining_steps,
                self.working_steps, max(0, self.report_reserve - self.report_used_steps),
                self.reported_tokens, self.unknown_usage_steps, reservation,
            )

    def _check_ticket(self, ticket):
        if ticket.owner != id(self) or not ticket.active:
            raise ValueError("请求凭证不属于本预算，或已完成记账")

    def reserve_tokens(self, ticket: RequestReservation, reservation: int):
        """Account for in-flight requests before releasing the caller to HTTP."""
        if reservation < 0:
            raise ValueError("reservation 不能小于 0")
        with _BUDGET_LOCK:
            self._check_ticket(ticket)
            change = reservation - ticket.reserved_tokens
            self.check_request(change, ticket.purpose, ticket.reserved_tokens)
            self.inflight_reserved_tokens += change
            ticket.reserved_tokens = reservation

    def settle_request(self, ticket: RequestReservation, response):
        with _BUDGET_LOCK:
            self._check_ticket(ticket)
            self.inflight_reserved_tokens -= ticket.reserved_tokens
            self.record_usage(response, ticket.reserved_tokens)
            ticket.active = False

    def cancel_request(self, ticket: RequestReservation):
        """Only call when the model request is confirmed not sent."""
        with _BUDGET_LOCK:
            self._check_ticket(ticket)
            self.inflight_reserved_tokens -= ticket.reserved_tokens
            self.refund_step(ticket.purpose)
            ticket.active = False

    def refund_step(self, purpose="work"):
        with _BUDGET_LOCK:
            self.used_steps -= 1
            if purpose == "report":
                self.report_used_steps -= 1

    def record_usage(self, response, reservation=0) -> None:
        """只累计模型实际返回的用量；缺失不能算作已知的零。"""
        with _BUDGET_LOCK:
            usage = getattr(response, "usage_metadata", None)
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            if type(total) is int and total >= 0:
                self.reported_tokens += total
            else:
                self.unknown_usage_steps += 1
                self.unknown_reserved_tokens += reservation

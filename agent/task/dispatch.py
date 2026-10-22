"""用 LangGraph 调度工作边界；任务原件和恢复状态仍由现有内核保存。"""

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.acceptance import revisions
from agent.acceptance import boundaries
from agent.model.errors import ModelInterrupted, ModelRequestNotSent
from agent.task import control, dialogue, repair
from agent.task.contracts import WorkRequest


RUNNING = {"needs_plan", "executing"}
# 图不再另设较低的循环次数限制；仍由既有调用预算、时限和暂停决定是否继续。
RECURSION_LIMIT = 2_147_483_647


class DispatchState(TypedDict, total=False):
    next_action: str
    plan: object
    response: dict | None
    error: ValueError | None
    result: dict


class TaskDispatch:
    def __init__(self, owner):
        self.owner = owner
        builder = StateGraph(DispatchState)
        # 图只保留四个业务节点；控制、配对、拒收和收尾仍是既有内部处理。
        self.actions = {
            "boundary": self.boundary,
            "finish_rejection": self.finish_rejection,
            "execute_plan": self.execute_plan,
            "verify_candidate": self.verify_candidate,
            "resume_web": self.resume_web,
            "release_candidate": self.release_candidate,
            "request_plan": self.request_plan,
            "route_response": self.route_response,
            "reply_text": self.reply_text,
            "reply_web": self.reply_web,
            "read_boundaries": self.read_boundaries,
            "review_acceptance": self.review_acceptance,
            "record_blocked": self.record_blocked,
            "accept_plan": self.accept_plan,
            "reject_reply": self.reject_reply,
            "finish_cycle": self.finish_cycle,
            "finalize": self.finalize,
        }
        self.rejects = {
            "request_plan", "route_response", "reply_text", "reply_web",
            "review_acceptance", "record_blocked", "accept_plan", "read_boundaries",
        }
        destinations = {name: name for name in ("search", "plan", "execute", "accept", END)}
        self.active_phase = None
        for phase in ("search", "plan", "execute", "accept"):
            builder.add_node(phase, self.stage(phase), retry_policy=())
            builder.add_conditional_edges(phase, self.route, destinations)
        builder.add_conditional_edges(START, self.route, destinations)
        # 明确禁止继承外层图的检查点，也不复制任务的完整历史到第二套存储。
        self.graph = builder.compile(checkpointer=False, name="monoplanner_work")

    def route(self, state):
        action = state["next_action"]
        if action == END:
            return END
        if action in {"resume_web", "reply_web", "read_boundaries"}:
            return "search"
        if action in {"review_acceptance", "release_candidate", "verify_candidate"}:
            return "accept"
        if action == "accept_plan" and state["plan"].kind == "candidate":
            return "accept"
        if action == "execute_plan":
            from agent.task.operations import pending_phase

            return pending_phase(self.owner.state["pending"])
        return "plan"

    def stage(self, phase):
        def invoke(state):
            current = dict(state)
            self.active_phase = phase
            try:
                while self.route(current) == phase:
                    action = current["next_action"]
                    result = self.wrap(self.actions[action], action in self.rejects)(current)
                    current.update(result)
                return current
            finally:
                self.active_phase = None

        return invoke

    def wrap(self, action, reject_invalid=False):
        def invoke(state):
            try:
                return action(state) or {"next_action": "finish_cycle"}
            except ValueError as exc:
                if reject_invalid:
                    return {"next_action": "reject_reply", "error": exc}
                self.owner.state.update(status="runtime_error", reason=f"ValueError: {exc}")
            except ModelInterrupted as exc:
                self.owner.on_model_interrupted(exc.kind, sent=exc.sent)
            except ModelRequestNotSent as exc:
                self.owner.state.update(status=exc.status, reason=exc.reason)
            except Exception as exc:
                self.owner.state.update(
                    status="runtime_error", reason=f"{type(exc).__name__}: {exc}",
                )
            return {"next_action": "finalize", "plan": None, "response": None, "error": None}

        return invoke

    # ============================================================
    # 每轮边界先处理控制输入和已保存工作，不抢先请求模型
    # ============================================================
    def boundary(self, state):
        owner = self.owner
        if owner.state["status"] not in RUNNING:
            return {"next_action": "finalize"}
        control.check_interrupted(pause_only=True)
        if owner.state.get("rejection_pending"):
            return {"next_action": "finish_rejection"}
        owner.budget.check_request(0)
        if dialogue.apply_controls(owner):
            return {"next_action": "finish_cycle"}
        for field, action in (
            ("pending", "execute_plan"),
            ("web_pending", "resume_web"),
            ("acceptance_release", "release_candidate"),
        ):
            if owner.state.get(field):
                if field == "pending":
                    from agent.task.operations import pending_phase

                    if pending_phase(owner.state["pending"]) == "accept":
                        action = "verify_candidate"
                return {"next_action": action}
        return {"next_action": "request_plan"}

    def finish_rejection(self, state):
        repair.finish_rejection(self.owner)

    def execute_plan(self, state):
        self.owner.execute_pending()

    def verify_candidate(self, state):
        owner = self.owner
        pending = owner.state["pending"]
        # 已提交的验收结果只补齐计划收尾，恢复时绝不再次运行验收。
        if pending["rows"] and pending["rows"][-1]["status"] == "failed":
            owner.finish_plan("execution_failed")
        elif owner.state["rounds"] and owner.state["rounds"][-1]["plan"] == pending["id"]:
            outcome = ("verified_success" if owner.state["verification"]["status"] == "passed"
                       else "verification_failed")
            owner.finish_plan(outcome)
        else:
            owner.final_acceptance()

    def resume_web(self, state):
        from agent.execution.web import execute_calls

        execute_calls(self.owner, self.owner.state["web_pending"]["response"])

    def release_candidate(self, state):
        release = self.owner.state["acceptance_release"]
        self.owner.accept_plan(WorkRequest.model_validate(release["plan"]), release["response"])

    # ============================================================
    # 请求、响应分类与工具处理由独立节点和条件边连接
    # ============================================================
    def request_plan(self, state):
        raw = self.owner.state.get("response_pending")
        plan, response = self.owner.decode_plan(raw) if raw else self.owner.call_planner()
        return {"plan": plan, "response": response, "next_action": "route_response"}

    def route_response(self, state):
        owner, plan = self.owner, state["plan"]
        if dialogue.apply_controls(owner):
            return {"next_action": "finish_cycle"}
        if owner.interactive:
            dialogue.publish(owner, state["response"])
        if isinstance(plan, dialogue.TextReply):
            action = "reply_text"
        elif isinstance(plan, dialogue.WebReply):
            action = "reply_web"
        elif isinstance(plan, boundaries.Lookup):
            action = "read_boundaries"
        elif isinstance(plan, revisions.Review):
            action = "review_acceptance"
        elif plan.kind == "blocked":
            action = "record_blocked"
        else:
            action = "accept_plan"
        return {"next_action": action}

    def reply_text(self, state):
        dialogue.accept_text(self.owner, state["response"])

    def reply_web(self, state):
        from agent.execution.web import execute_calls

        execute_calls(self.owner, state["response"])

    def review_acceptance(self, state):
        revisions.review(self.owner, state["plan"], state["response"])

    def read_boundaries(self, state):
        boundaries.execute(self.owner, state["plan"], state["response"])

    def record_blocked(self, state):
        self.owner.state.update(
            response_pending=None, status="task_blocked", reason=state["plan"].note,
        )

    def accept_plan(self, state):
        self.owner.accept_plan(state["plan"], state["response"])

    def reject_reply(self, state):
        repair.prepare_rejection(self.owner, state["error"])

    # ============================================================
    # 单轮让出、任务结束，以及工作服务的同租约续接
    # ============================================================
    def finish_cycle(self, state):
        if self.owner.yield_boundary or self.owner.state["status"] not in RUNNING:
            action = "finalize"
        else:
            action = "boundary"
        return {"next_action": action, "plan": None, "response": None, "error": None}

    def finalize(self, state):
        result = self.owner.result()
        action = "boundary" if self.owner.continue_after_return() else END
        return {"result": result, "next_action": action}

    def run(self):
        # 单路径同步运行保持现有 SQLite 租约与节点在同一线程；公开流仍用原事件通道。
        result = self.graph.invoke(
            {"next_action": "boundary"},
            {"recursion_limit": RECURSION_LIMIT, "max_concurrency": 1},
            stream_mode="values", subgraphs=False,
        )
        return result["result"]


def run(owner):
    if not hasattr(owner, "_task_dispatch"):
        owner._task_dispatch = TaskDispatch(owner)
    return owner._task_dispatch.run()

"""单一规划器依次规划、确定性执行、最终验收，再按需重新规划。"""

from dataclasses import asdict
import json
from pathlib import Path
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import message_to_dict, messages_from_dict, messages_to_dict
from agent.task.instructions import ARTIFACT_RULES
from agent.acceptance.artifact_checks import required_consumer, requested_calls
from agent.model.client import configure_call, request_timeout
from agent.model.errors import (
    ModelInterrupted,
    ModelRequestNotSent,
    ModelResponseNotSaved,
    is_context_overflow,
)
from agent.model.payload import request_reservation, estimate_input_tokens
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions
from agent.acceptance.contracts import VerificationSpec
from agent.execution.executor import OperationExecutor
from agent.storage.records import digest, save
from agent.task.contracts import WorkRequest
from agent.task.instructions import TASK_RULES
from agent.task import control, dialogue, operations, repair
from agent.model.context import admit, CONTEXT_WINDOW_TOKENS
from agent.model import compaction
from agent.source import lifecycle as source_runtime
from agent.acceptance import revisions

ENGINE = "MonoPlanner 1.0"


class TaskRunner:
    def __init__(
        self,
        model,
        environment,
        artifacts_root=None,
        *,
        max_rounds=3,
        task_token_limit=None,
        task_timeout=7200,
        state_dir=None,
        pause_after_steps=None,
        persistence=True,
        optimize_source=True,
        overlap_preparation=True,
        interactive=False,
        yield_boundary=False,
        event_sink=None,
        control_source=None,
        runtime_settings=None,
        work_rules_factory=None,
        user_request_id=None,
    ):
        self.interactive, self.yield_boundary = interactive, yield_boundary
        self.event_sink, self.control_source = event_sink, control_source
        self.model, self.environment = model, environment
        self.artifacts_root = Path(artifacts_root or Path.home() / ".local/state/task")
        self.explicit_state = Path(state_dir).resolve() if state_dir else None
        self.max_rounds, self.task_token_limit, self.task_timeout = (
            max_rounds,
            task_token_limit,
            task_timeout,
        )
        self.pause_after_steps = pause_after_steps
        self.steps_this_invocation = 0
        self.state = {}
        self.persistence = persistence
        self.history_store = None
        self.optimize_source, self.overlap_preparation = optimize_source, overlap_preparation
        self.preparation, self.preparation_errors = None, []
        self.runtime_settings = dict(runtime_settings or {})
        self.work_rules_factory = work_rules_factory
        self.user_request_id = user_request_id
        from agent.model.tools import CODING_TOOLS

        self.tools = CODING_TOOLS

    # ========================================================================
    # 状态持久化与公开进度
    # ========================================================================
    def persist(self, event, payload=None):
        self.state["budget"] = asdict(self.budget)
        self.state["elapsed_seconds"] = self.prior_elapsed + time.monotonic() - self.run_started
        if self.history_store is not None:
            self.history_store.record(self.state, event, payload or {})
        else:
            save(self.root / "state.json", self.state)
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"time": time.time(), "kind": event, "payload": payload or {}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if self.event_sink and event in dialogue.PROGRESS:
            phase, text = dialogue.PROGRESS[event]
            try:
                self.event_sink({"type": "progress", "phase": phase, "text": text})
            except (OSError, RuntimeError):
                pass  # 展示流断开不能使已保存的执行失效。

    # ========================================================================
    # 初始仓库材料与规划器输入
    # ========================================================================
    def initial_material(self):
        files = self.executor.inventory()
        names = sorted(files)
        docs = [
            n for n in names if "/" not in n and n.lower().startswith(("readme", "contributing"))
        ][:2]
        return {
            "root_entries": sorted({n.split('/')[0] for n in names}),
            "file_count": len(names),
            "readme": [self.executor.read(n, 0, 150) for n in docs],
            "test_config_paths": [
                n
                for n in names
                if n
                in {
                    "bin/test",
                    "pytest.ini",
                    "tox.ini",
                    "pyproject.toml",
                    "setup.cfg",
                    "runtests.py",
                }
            ],
            "runtime_facts": getattr(self.environment, "runtime_facts", {}),
            "source": self.evidence.current,
        }

    def planner_messages(self):
        return compaction.messages(self)

    def call_planner(self):
        control.check_interrupted(pause_only=True)
        self.budget.check_request(0)
        if self.state.get("repair_draft"):
            repair.read(self, self.state["repair_draft"])
        purpose = "plan_repair" if self.state.get("plan_error") else "plan"
        selected = configure_call(
            self.model.bind_tools(self.tools),
            purpose,
            timeout=request_timeout(self.model, self.budget.remaining_time()),
        )
        cap = getattr(selected, "max_output_tokens", 256000)
        cap = cap if type(cap) is int else 256000
        compaction.prepare(self, cap)
        for attempt in range(compaction.CONFIG['max_overflow_retries'] + 1):
            messages = self.planner_messages()
            admit(self.state, messages, self.tools, getattr(selected, "context_envelope", {}))
            try:
                raw = self.request_model(messages, purpose)
            except Exception as exc:
                if (
                    not is_context_overflow(exc)
                    or attempt == compaction.CONFIG['max_overflow_retries']
                ):
                    raise
                compaction.prepare(self, cap, force=True)
                continue
            return self.decode_plan(raw)

    # ========================================================================
    # 模型请求的预算、记录与异常结算
    # ========================================================================
    def request_model(self, messages, purpose):
        retrieval_rewrite = purpose == 'rag_rewrite'
        request_tools = () if retrieval_rewrite else self.tools
        pending_field = 'rag_request_pending' if retrieval_rewrite else 'request_pending'
        selected = configure_call(
            self.model.bind_tools(request_tools),
            purpose,
            timeout=request_timeout(self.model, self.budget.remaining_time()),
        )
        cap = getattr(selected, "max_output_tokens", 256000)
        cap = cap if type(cap) is int else 256000
        if (purpose != 'compaction' and not retrieval_rewrite
                and self.state.get('baseline') and self.environment is not None):
            source_runtime.begin(self)

        # ========================================================================
        # 发送前预留额度并保存请求原件
        # ========================================================================
        ticket = self.budget.reserve_request(
            request_reservation(messages, request_tools, cap), purpose=purpose
        )
        folder = self.root / "requests" / ("call-%04d" % ticket.call_id)
        from agent.storage.requests import save_request

        save_request(self.root, folder, messages_to_dict(messages), purpose, self.history_store)
        self.state[pending_field] = {
            "call": ticket.call_id,
            "reservation": ticket.reserved_tokens,
            "purpose": purpose,
        }
        if not retrieval_rewrite:
            self.state['context_usage'] = {
                'used_tokens': estimate_input_tokens(messages, self.tools),
                'window_tokens': CONTEXT_WINDOW_TOKENS,
                'estimated': True,
                'call': ticket.call_id,
            }
        self.persist("model_request_started", self.state[pending_field])

        # ========================================================================
        # 记录实际调用时长，按需展示思考片段
        # ========================================================================
        model_started = time.monotonic()
        live = None
        received = False
        ending = 'unknown'
        if (not retrieval_rewrite and self.event_sink
                and callable(getattr(type(selected), 'invoke_with_progress', None))):
            from agent.storage.live_reasoning import LiveReasoning

            public = self.state.get('public_events', [])
            anchor = (self.state.get('conversation_turn') or {}).get('id') or (
                public[-1]['id'] if public else ''
            )
            live = LiveReasoning(self.root, ticket.call_id, anchor)
        try:
            # 请求原件与预算保存期间也可能收到暂停，发送前再次核验。
            control.check_interrupted()
            response = (
                selected.invoke_with_progress(messages, live.append)
                if live
                else selected.invoke(messages)
            )
            received = True
        except ModelInterrupted as exc:
            if self.preparation:
                self.preparation.cancel()
            if exc.sent:
                self.budget.settle_request(ticket, None)
                self.state.setdefault('model_usage', []).append(
                    {
                        'call': ticket.call_id,
                        'purpose': purpose,
                        'seconds': time.monotonic() - model_started,
                        'usage': None,
                        'interrupted': True,
                    }
                )
            else:
                self.budget.cancel_request(ticket)
            self.state[pending_field] = None
            if purpose == 'compaction':
                # 尚未提交摘要，保留原始请求视图；
                # 后续明确继续时可以重新开始生成摘要。
                self.state['compaction_pending'] = None
            ending = 'interrupted'
            record = {
                'call': ticket.call_id,
                'purpose': purpose,
                'kind': exc.kind,
                'request_sent': exc.sent,
                'usage': 'unknown' if exc.sent else 'not_sent',
                'partial_output_executable': False,
            }
            if live:
                live.finish(ending)
                record['reasoning_content'] = live.row['detail']
                live = None
            save(folder / 'interruption.json', record)
            self.state.setdefault('interrupted_requests', []).append(record)
            self.persist('model_request_interrupted', record)
            raise
        except ModelRequestNotSent:
            self.budget.cancel_request(ticket)
            self.state[pending_field] = None
            self.persist("model_request_not_sent")
            raise
        except ModelResponseNotSaved as exc:
            self.budget.settle_request(ticket, exc.response)
            self.state[pending_field] = None
            self.persist("model_recording_error")
            raise
        except Exception:
            self.budget.settle_request(ticket, None)
            self.state[pending_field] = None
            if retrieval_rewrite:
                self.state.setdefault('model_usage', []).append({
                    'call': ticket.call_id, 'purpose': purpose, 'usage': None,
                    'seconds': time.monotonic() - model_started,
                })
            self.persist("model_request_error_unknown_usage")
            raise
        finally:
            if live and not received:
                live.finish(ending)
            if self.preparation is not None:
                self.preparation.note_request(model_started, time.monotonic())

        # ========================================================================
        # 成功返回后结算用量，并在执行工具前保存完整响应
        # ========================================================================
        self.budget.settle_request(ticket, response)
        self.state.setdefault('model_usage', []).append(
            {
                'call': ticket.call_id,
                'purpose': purpose,
                'seconds': time.monotonic() - model_started,
                'usage': response.usage_metadata,
                'max_output_tokens': cap,
            }
        )
        usage = response.usage_metadata
        if (not retrieval_rewrite and usage
                and all(type(usage.get(k)) is int for k in ('input_tokens', 'output_tokens'))):
            self.state['context_usage'].update(
                used_tokens=usage['input_tokens'] + usage['output_tokens'], estimated=False
            )
        raw = message_to_dict(response)
        save(folder / "response.json", raw)
        self.state[pending_field] = None
        if retrieval_rewrite:
            self.state['rag_rewrite_response'] = {
                'call': ticket.call_id,
                'record': (folder / 'response.json').relative_to(self.root).as_posix(),
            }
        elif purpose == 'compaction':
            self.state['compaction_pending'].update(
                stage='response_saved', response=raw, call=ticket.call_id
            )
        else:
            self.state["response_pending"] = raw
        self.persist("model_response_received", {"call": ticket.call_id})
        if not retrieval_rewrite and purpose != 'compaction':
            from agent.rag.integration import audit_work_citations

            audit_work_citations(self, raw, messages, folder)
        if live:
            ending = (
                'interrupted'
                if response.response_metadata.get('finish_reason') == 'length'
                else 'completed'
            )
            live.finish(ending, response.additional_kwargs.get('reasoning_content', ''))
        if purpose != 'compaction' and not retrieval_rewrite:
            from agent.task.output_limit import guard

            guard(self, raw)
        if not retrieval_rewrite and self.state.get('baseline') and self.environment is not None:
            source_runtime.collect(self)
        return raw

    # ========================================================================
    # 回复解码与工作计划校验
    # ========================================================================
    def decode_plan(self, raw):
        # 完整响应已归档；暂停后不再校验或派发纠错请求。
        control.check_interrupted(pause_only=True)
        self.rejected_input = None
        self.reviewing = False
        from agent.task.output_limit import guard

        guard(self, raw)
        response = messages_from_dict([raw])[0]
        finish = response.response_metadata.get("finish_reason")
        if finish in {"length", "content_filter", "aborted", "insufficient_system_resource"}:
            raise ValueError("Incomplete model response: " + str(finish))
        calls = response.tool_calls
        if (
            calls
            and not response.invalid_tool_calls
            and all(c['name'] in {'web_search', 'web_fetch'} for c in calls)
        ):
            return dialogue.WebReply(calls), raw
        if (
            self.interactive
            and not calls
            and not response.invalid_tool_calls
            and isinstance(response.content, str)
            and response.content.strip()
        ):
            return dialogue.TextReply(response.content), raw
        if (
            len(calls) == 1
            and not response.invalid_tool_calls
            and calls[0]["name"] in {"freeze_acceptance", "revise_acceptance"}
        ):
            self.reviewing = True
            return (
                self.validate_work(
                    {
                        **calls[0]["args"],
                        "decision": (
                            "freeze" if calls[0]["name"] == "freeze_acceptance" else "revise"
                        ),
                    },
                    revisions.Review,
                ),
                raw,
            )
        if response.invalid_tool_calls:
            invalid = response.invalid_tool_calls
            if not calls and len(invalid) == 1 and invalid[0]["name"] == "submit_plan":
                self.rejected_input = invalid[0]["args"]
                repair.loads(self.rejected_input)
            raise ValueError("Expected exactly one valid work tool call; no actions were executed")
        if len(calls) != 1 or calls[0]["name"] not in {"submit_plan", "repair_work"}:
            raise ValueError("Expected exactly one valid work tool call; no actions were executed")
        value = (
            repair.merge(self, calls[0]["args"])
            if calls[0]["name"] == "repair_work"
            else calls[0]["args"]
        )
        self.rejected_input = value
        return self.validate_work(value), raw

    def validate_work(self, value, contract=WorkRequest):
        errors = []
        plan = None
        try:
            plan = contract.model_validate(value)
        except ValueError as exc:
            errors.append((repair.fields(exc, value), exc))
        if isinstance(value, dict) and value.get("kind") == "candidate":
            current = self.state.get("verification_spec")
            if current:
                frozen = {
                    "test_code": current.get("test_code") or "",
                    "test_driver": list(current.get("test_driver", ())),
                    "regression_command": self.state["regression_command"],
                }
                for name, original in frozen.items():
                    supplied = (
                        list(value[name])
                        if name == "test_driver" and isinstance(value.get(name), (list, tuple))
                        else value.get(name)
                    )
                    if supplied and supplied != original:
                        errors.append(
                            (
                                ["/" + name],
                                ValueError(
                                    "Fixed " + name + " cannot be replaced during source repair"
                                ),
                            )
                        )
            elif not self.state.get("verify_command") and isinstance(
                value.get("test_code", ""), str
            ):
                try:
                    revisions.spec_for(
                        self, value.get("test_code", ""), value.get("test_driver", ())
                    )
                except ValueError as exc:
                    errors.append((repair.fields(exc, value), exc))
        elif (
            isinstance(value, dict)
            and value.get("decision") == "revise"
            and isinstance(value.get("test_code", ""), str)
        ):
            try:
                revisions.spec_for(self, value.get("test_code", ""), value.get("test_driver", ()))
            except ValueError as exc:
                errors.append((repair.fields(exc, value), exc))
        if errors:
            raise repair.ValidationErrors(errors)
        return plan

    # ========================================================================
    # 冻结验收并接收计划
    # ========================================================================
    def freeze_contract(self, plan):
        current = self.state.get("verification_spec")
        if current:
            if plan.test_code and plan.test_code != (current.get("test_code") or ""):
                raise ValueError("Fixed task tests cannot be replaced during repair")
            if (
                plan.regression_command
                and plan.regression_command != self.state["regression_command"]
            ):
                raise ValueError("Fixed regression command cannot be replaced")
            if plan.test_driver and tuple(plan.test_driver) != tuple(
                current.get("test_driver", ())
            ):
                raise ValueError("Fixed test_driver cannot be replaced during source repair")
            return
        request = self.state["request"] + ''.join(
            '\n' + row['text'] for row in self.state.get('user_updates', [])
        )
        criteria = [request] + ([self.state["acceptance"]] if self.state["acceptance"] else [])
        command = self.state["verify_command"]
        consumer = (
            required_consumer(request + "\n" + (self.state["acceptance"] or ""))
            if not command
            else None
        )
        spec = VerificationSpec(
            tuple(criteria),
            command=command,
            test_code=None if command else plan.test_code,
            required_consumer=consumer,
            required_calls=requested_calls(request) if consumer else (),
            test_driver=() if command else plan.test_driver,
        )
        self.state["verification_spec"] = asdict(spec)
        self.state["regression_command"] = plan.regression_command
        save(
            self.root / "frozen-acceptance.json",
            {"spec": asdict(spec), "regression": plan.regression_command},
        )
        self.persist("acceptance_frozen", {"digest": digest(asdict(spec))})

    def accept_plan(self, plan, response):
        if plan.kind == "candidate":
            if not self.state.get("verification_spec") and not self.state.get("verify_command"):
                revisions.stage_candidate(self, plan, response)
                return
            self.freeze_contract(plan)
        fingerprint = digest([plan.model_dump(), self.evidence.current])
        if fingerprint in self.state["seen_plans"]:
            self.state.update(
                status="stalled",
                reason="Identical plan for identical source repeated; no new decision",
            )
            return
        self.state["seen_plans"].append(fingerprint)
        number = self.state["plan_count"] + 1
        name = "p%04d" % number
        value = {
            "id": name,
            "parent": self.state.get("last_plan"),
            "plan": plan.model_dump(),
            "response": response,
            "base_source": self.evidence.current,
            "cursor": 0,
            "rows": [],
        }
        if self.state.get("acceptance_release"):
            value["origin_response"] = self.state["acceptance_release"]["origin_response"]
        save(self.root / "plans" / name / "plan.json", value)
        self.state.update(
            pending=value,
            plan_count=number,
            plan_error="",
            repair_count=0,
            response_pending=None,
            status="executing",
        )
        self.state.pop("acceptance_release", None)
        if plan.kind != "inspect":
            self.state.pop("repair_draft", None)
        self.persist("plan_accepted", {"id": name, "base_source": self.evidence.current})

    # ========================================================================
    # 确定性执行与最终验收入口
    # ========================================================================
    def execute_pending(self):
        return operations.execute_pending(self)

    def _execute_pending(self):
        return operations._execute_pending(self)

    def final_acceptance(self):
        return operations.final_acceptance(self)

    def finish_plan(self, outcome):
        return operations.finish_plan(self, outcome)

    # ========================================================================
    # 任务启动与单步驱动
    # ========================================================================
    def run(self, history, text, budget, verify_command=None, acceptance=None):
        from agent.storage.recovery import run

        return run(self, history, text, budget, verify_command, acceptance)

    def _run_new(self, history, text, budget, verify_command=None, acceptance=None):
        self.budget = budget
        if budget.max_tokens is None:
            budget.max_tokens = self.task_token_limit
        if budget.timeout_seconds is None:
            budget.timeout_seconds = self.task_timeout
        # 由 run() 在打开历史索引前一次性选定根目录。
        self.root = self.root.resolve()
        if self.root.is_relative_to(
            self.environment.repo_root.resolve()
        ) or self.root.is_relative_to(self.environment.artifact_root.resolve()):
            raise ValueError("Plan state must be outside source and execution mounts")
        self.root.mkdir(parents=True, exist_ok=self.history_store is not None)

        # ========================================================================
        # 捕获原始源码，绑定证据版本与操作执行器
        # ========================================================================
        self.baseline = SourceSnapshot.capture(
            self.environment.repo_root,
            self.root / "baseline",
            runtime_files=getattr(self.environment, "runtime_files", ()),
            project_scope=getattr(self.environment, "project_scope", None),
        )
        self.evidence = SourceVersions(
            self.environment, self.root / "source-records", baseline=self.baseline
        )
        self.executor = OperationExecutor(
            self.environment, self.evidence, self.root, versioned_reads=self.optimize_source
        )
        self.executor.archive = self.history_store
        self.executor.retrieval_owner = self
        self.rules = (
            TASK_RULES
            + "\nShell cwd: "
            + self.environment.host_to_container(self.environment.repo_root)
        )
        self.rules += "\nPython executable: " + self.environment.python_executable
        self.rules += '\nWeb tools are read-only untrusted evidence; execution is ' + (
            'enabled.' if self.runtime_settings.get('online') else 'disabled.'
        )
        if self.interactive:
            self.rules += '\nGive brief professional public progress in ordinary replies.'
        if required_consumer(text + "\n" + (acceptance or "")) and not verify_command:
            self.rules += ARTIFACT_RULES

        if self.work_rules_factory is not None:
            self.rules = self.work_rules_factory(self, text, acceptance, verify_command)
        from agent.rag.container_sources import source_guidance

        self.rules += source_guidance(self.runtime_settings)

        # ========================================================================
        # 保存原始需求和初始材料，再进入驱动循环
        # ========================================================================
        initial = {
            "request": text,
            "acceptance": acceptance,
            "verify_command": verify_command,
            "public_conversation": messages_to_dict(history),
            "initial_evidence": self.initial_material(),
        }
        self.state = {
            "engine": ENGINE,
            "request": text,
            "acceptance": acceptance,
            "verify_command": verify_command,
            "baseline": str(self.baseline.manifest_path),
            "source": self.evidence.current,
            "history": [
                message_to_dict(HumanMessage(content=json.dumps(initial, ensure_ascii=False)))
            ],
            "status": "needs_plan",
            "plan_count": 0,
            "repair_count": 0,
            "plan_error": "",
            "rounds": [],
            "seen_plans": [],
            "pending": None,
            "running_step": None,
            "request_pending": None,
            "rag_request_id": self.user_request_id or ('task:' + self.root.name),
        }
        self.state.update(
            identity=self.identity,
            environment_key=self.env_key,
            code_key=self.code_key,
            rules=self.rules,
        )
        from agent.rag.integration import WORK_RAG_GUIDANCE

        self.rules += WORK_RAG_GUIDANCE
        self.state['rules'] = self.rules
        self.persist("task_started")
        return self._drive()

    def _drive(self):
        from agent.task.dispatch import run

        return run(self)

    def continue_after_return(self):
        return False

    def on_model_interrupted(self, kind, *, sent=True):
        # 纯边界暂停没有半截模型回复，不能改动已保存检查点的历史前缀。
        if sent:
            self.state['history'].append(
                message_to_dict(
                    HumanMessage(
                        content='The user interrupted the previous model request. Its partial reply was not submitted or executed. '
                        'Use completed records and the latest user requirements; do not replay completed operations.'
                    )
                )
            )
        self.state.pop('dispatch', None)
        self.state.pop('turn_status', None)
        self.state.update(
            status='needs_plan' if kind == 'message' else 'paused',
            reason='User interrupted model reply',
            pause_requested=kind == 'pause',
        )
        dialogue.apply_controls(self)
        self.persist('model_wait_interrupted', {'kind': kind})

    # ========================================================================
    # 结果汇总与验证证据
    # ========================================================================
    def result(self):
        source_runtime.close(self)
        from agent.rag.container_sources import expire_container_sources

        expire_container_sources(self)
        self.persist("task_returned", {"status": self.state["status"]})
        report = f"{self.state['status']}: {self.state.get('reason', '')}"
        from agent.storage.usage import summarize
        from agent.task.output_limit import public as public_limit

        outcome = {
            "status": self.state["status"],
            "stop_reason": self.state.get("reason", ""),
            'pause_requested': self.state.get('pause_requested', False),
            'output_limit': public_limit(self.state),
            'usage': summarize(self.state),
            'existing_runs': self.state.get('existing_runs', []),
            "messages": [HumanMessage(content=self.state["request"]), AIMessage(content=report)],
            "task_kind": "coding",
            "workflow": "task",
            "artifact_dir": str(self.root),
            "history_dir": str(self.root / "requests"),
            "verification": self.state.get("verification", {}),
            "rounds": self.state["rounds"],
            "report_status": "program_report",
            "final_report": report,
            "worker_answer": "",
            "plan_path": str(self.root / "plans"),
            "patch_path": str(self.root / "candidate.patch"),
            "planner_calls": self.budget.used_steps,
            "executor_model_calls": 0,
            "verifier_model_calls": 0,
            "local_acceptance_calls": 0,
            "state_file": str(self.root / "state.json"),
            "calls_this_invocation": self.budget.used_steps - self.calls_before,
            "fresh_verification": self.fresh_verification,
            "result_reused": self.reused_result,
            "source_performance": self.state.get("source_performance", {}),
            "local_state_stats": dict(self.history_store.stats) if self.history_store else {},
            "compactions": self.state.get("compactions", []),
            "web_usage": self.state.get('web_usage', []),
            "history_database": str(self.root / "history.sqlite3") if self.history_store else None,
        }
        if self.interactive:
            public = [
                (AIMessage if row['type'] == 'ai' else HumanMessage)(
                    content=row['content'], id=row['id']
                )
                for row in self.state.get('public_events', [])
            ]
            outcome['messages'] = [
                *self.public_history,
                HumanMessage(content=self.state['request']),
                *public,
            ]
        save(self.root / "result.json", {k: v for k, v in outcome.items() if k != "messages"})
        return outcome

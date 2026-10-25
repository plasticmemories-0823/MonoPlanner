"""固定工作主干与按已确认交付类型选择的执行说明。"""

from agent.settings import PROJECT_ROOT


TASK_RULES = (PROJECT_ROOT / "resources/prompts/task.txt").read_text(encoding="utf-8")


_COMMAND_GUIDANCE = (
    "For generated acceptance of a repository_entry, use consumer=command with the confirmed "
    "relative entry_file, interpreter and literal argv. The host syntax-checks and runs the "
    "actual entry; test_code must still check business behavior. Keep producer/numeric and "
    "bound-test fields empty. Do not substitute a wrapper or a code generator's Python producer "
    "for the requested artifact."
)

_RUNTIME_GUIDANCE = {
    ("python_api", "python"): (
        "For generated acceptance of a python_api, use consumer=python_tests and "
        "interpreter='python'. Bind api_entry to the original module:function or "
        "module:Class.method and test_ids to distinct Class.test_method entries in test_code. "
        "Those tests must actually call the synchronous Python API; their execution supplies "
        "the receipt without another command. Put imports/setup/assertions in test_code. "
        "Declaration imports/setup/expression, entry_file, python_module, argv, variables, "
        "cases and result_variable must be empty. Unsupported API implementations are a "
        "capability gap, not permission to substitute an adjacent entry."
    ),
    ("file_artifact", "python"): (
        "For generated acceptance of a file_artifact, use consumer=python_tests and "
        "interpreter='python'. Bind entry_file to the original relative delivery path and "
        "test_ids to distinct Class.test_method entries in test_code. Those tests must read "
        "the actual delivered bytes through acceptance_artifacts.read_artifact(relative_path) "
        "and assert their required content/behavior; reading alone does not prove correctness. "
        "The single-file limit is 512 MiB. Put imports/setup/assertions in test_code; leave "
        "producer, command, numeric and API-specific declaration fields empty."
    ),
    ("generated_code", "c"): (
        "For generated acceptance of generated_code targeting C, imports/setup/expression "
        "must invoke the confirmed actual producer with its original input construction. "
        "Its unchanged returned text must reach the compiler and runtime; compiling only the "
        "producer or repairing/substituting its output is insufficient. Use c_expression for "
        "numeric expressions or c_statement for assignment statements, with variables, "
        "result_variable where required and cases of values plus independently justified "
        "expected results. Keep command and bound-test fields empty. Cover required inputs "
        "and relevant boundaries; do not relabel generated code as a repository entry."
    ),
    ("repository_entry", "python"): _COMMAND_GUIDANCE + (
        " For confirmed Python module invocation, set python_module to its dotted name, "
        "bind entry_file to that module's .py or package __main__.py and retain argv. "
        "An empty python_module preserves file invocation."
    ),
    ("repository_entry", "node"): _COMMAND_GUIDANCE,
    ("repository_entry", "shell"): _COMMAND_GUIDANCE,
    ("repository_entry", "wine-cmd"): (
        "For explicitly confirmed Wine CMD compatibility, use artifact_type=repository_entry, "
        "consumer=command, interpreter='wine-cmd', the real .bat/.cmd entry_file and literal "
        "argv. The unchanged file runs once through Wine CMD; there is no separate full-file "
        "compilation pass. Only exercised branches are validated, not native Windows services, "
        "registry or permissions. Keep safe fixtures and syntax/behavior assertions in "
        "test_code. Paths/arguments with control characters or % ! \" & | < > ^ are unsupported; "
        "do not silently transform them. Leave producer, numeric, module and bound-test "
        "fields empty."
    ),
}

_UNSUPPORTED_GUIDANCE = (
    "No execution adapter is defined for this delivery type/runtime. Report the capability gap "
    "without changing the confirmed target, downgrading its consumer or claiming verification."
)


def execution_guidance(targets):
    """只读取已校验目标的类型；同类目标共享说明，不猜题名或改写约束。"""
    kinds = sorted({(item["artifact_type"], item["target_runtime"]) for item in targets})
    return {
        kind + ":" + runtime: _RUNTIME_GUIDANCE.get((kind, runtime), _UNSUPPORTED_GUIDANCE)
        for kind, runtime in kinds
    }

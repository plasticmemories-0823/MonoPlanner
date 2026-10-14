"""从资源文件原样加载固定任务指令。"""

from agent.settings import PROJECT_ROOT


TASK_RULES = (PROJECT_ROOT / "resources/prompts/task.txt").read_text(encoding="utf-8")
ARTIFACT_RULES = (PROJECT_ROOT / "resources/prompts/artifact_acceptance.txt").read_text(
    encoding="utf-8"
)

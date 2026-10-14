"""将固定的测试驱动原文复制到执行后端。"""

from agent.settings import PROJECT_ROOT


_BOOTSTRAP = (PROJECT_ROOT / "resources/drivers/unittest.py").read_text(encoding="utf-8")
_REPOSITORY_BOOTSTRAP = (PROJECT_ROOT / "resources/drivers/repository.py").read_text(
    encoding="utf-8"
)

"""Fixed test-driver text copied unchanged into the execution backend."""
from agent.settings import PROJECT_ROOT


_BOOTSTRAP = (PROJECT_ROOT / "resources/drivers/unittest.py").read_text(encoding="utf-8")
_REPOSITORY_BOOTSTRAP = (PROJECT_ROOT / "resources/drivers/repository.py").read_text(encoding="utf-8")

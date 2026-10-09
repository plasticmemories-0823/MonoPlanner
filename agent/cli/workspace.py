"""只保存上次选择的仓库路径；聊天记录和模型密钥不写入这里。"""

import json
import os
from pathlib import Path


STATE_PATH = (
    Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".config")
    / "langgraph-simple-agent"
    / "workspace.json"
)


def load_saved_repo() -> str | None:
    if not STATE_PATH.exists():
        return None
    data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    repo = data.get("repo") if isinstance(data, dict) else None
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError("保存的仓库配置无效，请重新选择仓库")
    return repo


def save_repo(repo: Path) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"repo": str(repo.resolve())}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

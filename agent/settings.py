from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_settings(overrides=None) -> dict:
    """只在真正执行模型任务时读取配置，不自动修改 .env。"""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)

    from agent.storage.preferences import Preferences
    preferences = Preferences()
    api_key = preferences.key()
    if not api_key:
        raise ValueError("请配置 DEEPSEEK_API_KEY")

    return {
        "api_key": api_key,
        **preferences.validate(overrides or preferences.defaults()),
        "temperature": 0,
    }

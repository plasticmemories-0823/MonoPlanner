"""Wine 可选环境的受信任安装资料；模型不能提供包名、脚本或安装命令。"""

import hashlib
from pathlib import Path
import shutil

from agent.settings import PROJECT_ROOT
from agent.execution.managed_plan import _digest, PreparationError


FILES = ('install-wine.py', 'compact-wine-prefix.py', 'wine')


def recipe_key():
    folder = PROJECT_ROOT / 'resources/environments'
    return _digest({name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    for name in FILES})


def copy_installer(destination, expected):
    if recipe_key() != expected:
        raise PreparationError('Wine 安装配方已变化，请停止后重新准备。')
    destination = Path(destination)
    destination.mkdir()
    for name in FILES:
        shutil.copyfile(PROJECT_ROOT / 'resources/environments' / name, destination / name)
    if _digest({name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                for name in FILES}) != expected:
        raise PreparationError('Wine 安装资料校验失败，尚未开始安装。')


def verify_receipt(value):
    if (not isinstance(value, dict) or value.get('version') != 'wine-8.0 (Debian 8.0~repack-4)'
            or value.get('cmd_exit_code') != 0 or value.get('cmd_probe') != 'MONOPLANNER_WINE_READY'):
        raise PreparationError('Wine CMD 运行检查未通过；没有发布可用环境。')

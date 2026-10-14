"""明确记录任务所需的本地运行材料，不复制全部被忽略文件。"""

import hashlib
import json
from pathlib import Path

from agent.source.snapshots import _walk, _reason, _safe_file, _inspect_file


def package_roots(repo):
    root = Path(repo).resolve()
    return sorted(
        p.parent.relative_to(root).as_posix()
        for base in (root, root / "src")
        if base.is_dir()
        for p in base.glob("*/__init__.py")
        if p.parent.name not in {"tests", "test", "docs", "examples", "build", "setup"}
        and not _reason(p.relative_to(root))
    )


def runtime_inventory(repo):
    """保留包内的 Python 和原生模块，排除缓存、构建产物及秘密资料。

    工作进程启动前冻结准确路径和哈希。该清单是执行环境输入，
    不是仓库文字提供的答案或授权。
    """
    root, files = Path(repo).resolve(), {}
    packages = package_roots(root)
    for relative, skip in _walk(root):
        if (
            skip
            or _reason(relative)
            or relative.suffix.lower() not in {".py", ".so", ".pyd", ".dll", ".dylib"}
        ):
            continue
        name = relative.as_posix()
        if not any(name.startswith(package + "/") for package in packages):
            continue
        path = _safe_file(root, relative)
        digest, private = _inspect_file(path)
        if not private:
            files[name] = digest
    return {
        "packages": packages,
        "files": files,
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def probe_runtime(environment, packages, directory, *, metadata_only=False):
    """检查真实执行后端；聊天开工准备不导入项目模块。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    modules = [] if metadata_only else sorted({Path(p).name for p in packages})
    # 使用 ASCII 脚本和 JSON，兼容原始 Python 3.6 镜像的 ASCII 区域设置。
    code = '''import importlib, importlib.util, json, shutil, sys, os
if not METADATA_ONLY:
    sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
report = {"python": sys.version, "imports": {}, "checkers": {}}
if METADATA_ONLY:
    report["checker_errors"] = {}
for name in MODULES:
    try:
        module = importlib.import_module(name)
        filename = getattr(module, "__file__", None)
        local = bool(filename and os.path.commonpath([os.getcwd(), os.path.abspath(filename)]) == os.getcwd())
        report["imports"][name] = {"ok": local, "file": filename, "from_task_source": local}
    except Exception as exc:
        report["imports"][name] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
for name in ("pytest", "flake8", "ruff"):
    try:
        report["checkers"][name] = bool(shutil.which(name) or importlib.util.find_spec(name))
    except Exception as exc:
        if not METADATA_ONLY:
            raise
        report["checkers"][name] = None
        report["checker_errors"][name] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(report))
'''.replace(
        "MODULES", repr(modules)
    ).replace(
        "METADATA_ONLY", repr(metadata_only)
    )
    script = directory / "runtime_probe.py"
    script.write_text(code, encoding="ascii")
    path = (
        environment.host_to_container(script)
        if hasattr(environment, "host_to_container")
        else str(script)
    )
    stdout, stderr = directory / "stdout.log", directory / "stderr.log"
    flags = ["-I", "-B"] if metadata_only else ["-B"]
    outcome = environment.run_process(
        [environment.python_executable, *flags, path],
        stdout_path=stdout,
        stderr_path=stderr,
        timeout=60,
        **({"isolated_launcher": True} if metadata_only else {}),
    )
    report = {"process": outcome}
    if outcome.get("exit_code") == 0 and not outcome.get("timed_out") and not outcome.get("error"):
        try:
            report.update(json.loads(stdout.read_text(encoding="utf-8").splitlines()[-1]))
        except (ValueError, IndexError):
            report["parse_error"] = "Runtime import probe did not return JSON."
    if metadata_only:
        report["ok"] = (
            isinstance(report.get("python"), str)
            and bool(report["python"].strip())
            and set(report.get("checkers", {})) == {"pytest", "flake8", "ruff"}
            and all(
                value is None or isinstance(value, bool) for value in report["checkers"].values()
            )
        )
    else:
        report["ok"] = bool(report.get("imports")) and all(
            v["ok"] for v in report.get("imports", {}).values()
        )
    if report.get("python"):
        facts = {
            "python": report["python"],
            "checkers": report.get("checkers", {}),
            "note": "来自任务执行后端的开工检查；代码与测试须兼容此版本，修改环境后需要重新查证。",
        }
        if metadata_only:
            facts = {
                **getattr(environment, "runtime_facts", {}),
                **facts,
                "checker_errors": report.get("checker_errors", {}),
            }
        environment.runtime_facts = facts
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return report

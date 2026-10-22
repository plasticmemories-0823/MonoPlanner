"""按固定清单准备 Windows rg；只在显式执行本脚本时下载。"""

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def prepare():
    manifest = json.loads((ROOT / "resources/ripgrep.json").read_text("utf-8"))
    request = urllib.request.Request(manifest["url"], headers={"User-Agent": "MonoPlanner-build"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if hashlib.sha256(raw).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("The ripgrep release archive failed its fixed SHA256 check.")
    target = ROOT / "launcher/runtime/ripgrep"
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        staged = {}
        for name in archive.namelist():
            leaf = Path(name).name
            if leaf not in manifest["files"]:
                continue
            if leaf in staged:
                raise ValueError("The runtime archive contains a duplicate required file.")
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != manifest["files"][leaf]:
                raise ValueError("The ripgrep runtime file failed verification: " + leaf)
            staged[leaf] = data
    if set(staged) != set(manifest["files"]):
        raise ValueError("The runtime archive is missing required files.")
    target.mkdir(parents=True, exist_ok=True)
    for name, data in staged.items():
        (target / name).write_bytes(data)
    print("Prepared verified ripgrep " + manifest["version"])


if __name__ == "__main__":
    prepare()

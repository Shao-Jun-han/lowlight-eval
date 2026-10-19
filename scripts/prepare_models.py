"""下载固定版本官方代码和作者公开权重；不安装官方老环境，不改网络源码。"""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hvi-weight", action="store_true")
    args = parser.parse_args()
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    for repo in sources["repositories"]:
        target = ROOT / repo["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            subprocess.run(["git", "clone", "--depth", "1", repo["url"], str(target)], check=True)
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if dirty.strip():
            raise RuntimeError(f"{target} 存在本地修改，请保留你的修改并提供干净的官方仓库。")
        current = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if current != repo["revision"]:
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", repo["revision"]], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", repo["revision"]], check=True)
        print(f"{repo['name']}: {repo['revision']}")
    if not args.skip_hvi_weight:
        weight = sources["hvi_weight"]
        target = ROOT / weight["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            temporary = target.with_suffix(".download")
            urllib.request.urlretrieve(weight["url"], temporary)
            temporary.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if weight.get("sha256") and digest != weight["sha256"]:
            raise RuntimeError(f"权重校验失败：{target}")
        print(f"HVI-CIDNet weight SHA256: {digest}")


if __name__ == "__main__":
    main()

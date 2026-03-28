from __future__ import annotations

import sys
from pathlib import Path


def check(path: Path, description: str) -> bool:
    ok = path.exists()
    mark = "OK " if ok else "ERR"
    print(f"[{mark}] {description}: {path}")
    return ok


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    repo_dir = base / "repos" / "tqsdk-python"
    venv_python = base / ".venv" / "Scripts" / "python.exe"
    local_skill = base / "skills" / "tqsdk-trading-and-data" / "tqsdk-trading-and-data" / "SKILL.md"
    codex_skill = Path(r"C:\Users\shirosk\.codex\skills\tqsdk-trading-and-data\SKILL.md")

    all_ok = True
    all_ok &= check(repo_dir / ".git", "源码仓库")
    all_ok &= check(venv_python, "虚拟环境解释器")
    all_ok &= check(local_skill, "本地 skills 文件")
    all_ok &= check(codex_skill, "Codex skills 安装")

    try:
        import tqsdk  # type: ignore

        print(f"[OK ] tqsdk import: version={tqsdk.__version__}")
    except Exception as exc:  # pragma: no cover
        all_ok = False
        print(f"[ERR] tqsdk import failed: {exc}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

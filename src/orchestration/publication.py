"""Explicit publication allowlist with working tree and complete-history scans."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_" + rb"[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_" + rb"[A-Za-z0-9_]{30,}"),
    re.compile(rb"sk-" + rb"[A-Za-z0-9_-]{30,}"),
    re.compile(rb"AKIA" + rb"[A-Z0-9]{16}"),
    re.compile(rb"(?i)(?:api_key|password|access_token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
)


def allowed(path: str) -> bool:
    item = Path(path)
    if item.is_absolute() or ".." in item.parts or any(part.startswith(".") for part in item.parts[:-1]):
        return False
    if path in {".gitignore", "README.md", "LICENSE", "pyproject.toml", "requirements.txt", "requirements-dev.txt"}:
        return True
    if item.parts[0] in {"src", "scripts", "tests"} and item.suffix == ".py":
        return True
    if item.parts[0] == "src" and item.name == "LICENSE":
        return True
    if item.parent == Path("configs") and item.suffix == ".yaml" and (not item.name.startswith("local") or item.name == "local.example.yaml"):
        return True
    if item.parent in {Path("configs/samr"), Path("configs/mesa")} and item.suffix == ".yaml" and not item.name.startswith("local"):
        return True
    if item.parts[0] == "docs" and item.suffix == ".md":
        return True
    return path == "reports/development/code_validation_report.md" or (
        item.parent in {Path("reports/architecture"), Path("reports/samr"), Path("reports/mesa")} and item.suffix == ".md")


def content_issues(path: str, data: bytes) -> list[str]:
    findings = []
    if not allowed(path):
        findings.append("path is outside publication allowlist")
    if len(data) > 1024 * 1024:
        findings.append("file exceeds one MiB source limit")
    if b"\0" in data:
        findings.append("binary file")
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        findings.append("credential pattern")
    if Path.home().as_posix().encode() in data:
        findings.append("personal machine path")
    return findings


def candidates(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                  if path.is_file() and allowed(path.relative_to(root).as_posix()))


def scan(root: Path, *, staged: bool = False, history: bool = True) -> dict:
    names = (subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                                     cwd=root, text=True).splitlines() if staged else candidates(root))
    failures = {}
    for name in names:
        data = (subprocess.check_output(["git", "show", f":{name}"], cwd=root) if staged else (root / name).read_bytes())
        issues = content_issues(name, data)
        if issues:
            failures[name] = issues
    history_files = 0
    if history:
        revisions = subprocess.run(["git", "rev-list", "--all"], cwd=root, capture_output=True, text=True, check=False)
        seen = set()
        for revision in revisions.stdout.splitlines():
            lines = subprocess.check_output(["git", "ls-tree", "-r", revision], cwd=root, text=True).splitlines()
            for line in lines:
                metadata, name = line.split("\t", 1)
                blob = metadata.split()[2]
                if (blob, name) in seen:
                    continue
                seen.add((blob, name))
                data = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=root)
                issues = content_issues(name, data)
                if issues:
                    failures[f"history:{revision}:{name}"] = issues
        history_files = len(seen)
    return {"passed": not failures, "files": names, "history_blobs_checked": history_files, "failures": failures}

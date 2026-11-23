import subprocess
from pathlib import Path

import pytest

from orchestration.publication import allowed, content_issues, scan


@pytest.mark.parametrize("name", ["data/table.png", "models/model.safetensors", "runs/result.jsonl",
                                 ".env", "configs/local.yaml", "reports/final/goal_completion_report.md",
                                 "src/private.key", "src/snapshot.pt", "../data.py", "logs/run.log"])
def test_artifacts_are_excluded(name):
    assert not allowed(name)


def test_secrets_and_large_files_are_rejected():
    fake_token = ("ghp_" + "a" * 40).encode()
    assert "credential pattern" in content_issues("src/example.py", fake_token)
    assert "file exceeds one MiB source limit" in content_issues("src/example.py", b"x" * (1024**2 + 1))
    assert "binary file" in content_issues("src/example.py", b"\x00\x01")


def test_complete_history_scan_catches_deleted_sensitive_file(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / ".env").write_text("EXAMPLE=dummy\n")
    subprocess.run(["git", "add", "--", ".env"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "Synthetic test fixture"], cwd=tmp_path, check=True)
    subprocess.run(["git", "rm", "-q", ".env"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "Remove fixture"], cwd=tmp_path, check=True)
    result = scan(tmp_path)
    assert not result["passed"]
    assert any("history:" in key for key in result["failures"])


@pytest.mark.parametrize("name", [".env", "dataset/page.png", "experiments/result.json",
                                 "models/model.safetensors", "configs/local.yaml", "reports/final/report.md"])
def test_actual_gitignore_blocks_artifacts(name):
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "check-ignore", name], cwd=project, capture_output=True, check=False)
    assert result.returncode == 0

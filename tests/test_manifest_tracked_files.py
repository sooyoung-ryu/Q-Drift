import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_files_match_artifact_manifest():
    # prepare_paper_artifacts.py refuses to overwrite files that differ from the manifest,
    # so tracked files listed there must keep their recorded hashes.
    manifest = json.loads((REPO_ROOT / "manifests" / "paper_artifacts.json").read_text())["files"]
    tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.splitlines())
    checked = [path for path in manifest if path in tracked]
    assert checked
    for path in checked:
        digest = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        assert digest == manifest[path]["sha256"], path

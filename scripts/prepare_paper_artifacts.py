#!/usr/bin/env python3
"""Install or verify the fixed paper artifact bundle.

The bundle manifest pins every checkpoint, calibration statistic, and
configuration file by byte size and SHA256. Existing files are never replaced
when their content differs from the manifest; remove the mismatching file
manually after confirming it is not needed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


CHUNK_SIZE = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        raise SystemExit(f"Missing manifest: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from None
    if not isinstance(data, dict):
        raise SystemExit(f"Manifest must be a JSON object: {path}")
    return data


def repo_relative_path(repo_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise SystemExit(f"Unsafe artifact path in manifest: {relative_path!r}")
    target = (repo_root / candidate).resolve()
    root = repo_root.resolve()
    if target != root and root not in target.parents:
        raise SystemExit(f"Artifact path escapes repository root: {relative_path!r}")
    return target


def validate_entry(relative_path: str, entry: Any) -> tuple[int, str]:
    if not isinstance(entry, dict):
        raise SystemExit(f"Artifact entry must be an object: {relative_path}")
    expected_bytes = entry.get("bytes")
    expected_hash = entry.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise SystemExit(f"Artifact entry has invalid byte size: {relative_path}")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise SystemExit(f"Artifact entry has invalid SHA256: {relative_path}")
    encoding = entry.get("encoding")
    if encoding not in (None, "gzip"):
        raise SystemExit(f"Artifact entry has unsupported encoding: {relative_path}")
    parts = entry.get("parts")
    if parts is not None:
        validate_parts(relative_path, parts)
    return expected_bytes, expected_hash.lower()


def validate_parts(relative_path: str, parts: Any) -> list[dict[str, Any]]:
    if not isinstance(parts, list) or not parts:
        raise SystemExit(f"Artifact entry has invalid parts list: {relative_path}")
    validated = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise SystemExit(f"Artifact part must be an object: {relative_path} part {index}")
        path = part.get("remote_path", part.get("path"))
        expected_bytes = part.get("bytes")
        expected_hash = part.get("sha256")
        if not isinstance(path, str) or not path:
            raise SystemExit(f"Artifact part has invalid remote_path: {relative_path} part {index}")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise SystemExit(f"Artifact part has invalid byte size: {relative_path} part {index}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise SystemExit(f"Artifact part has invalid SHA256: {relative_path} part {index}")
        validated.append({"remote_path": path, "bytes": expected_bytes, "sha256": expected_hash.lower()})
    return validated


def verify_file(path: Path, expected_bytes: int, expected_hash: str) -> bool:
    return path.exists() and path.stat().st_size == expected_bytes and sha256(path) == expected_hash


def explain_existing_mismatch(path: Path, expected_bytes: int, expected_hash: str) -> str:
    actual_bytes = path.stat().st_size
    actual_hash = sha256(path)
    return (
        f"Existing artifact differs from the paper bundle: {path}\n"
        f"  expected bytes={expected_bytes} sha256={expected_hash}\n"
        f"  actual   bytes={actual_bytes} sha256={actual_hash}\n"
        "Refusing to overwrite it silently. Move or remove the file, then rerun."
    )


def describe_file_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    return f"bytes={path.stat().st_size} sha256={sha256(path)}"


def describe_response(response: Any) -> str:
    headers = response.headers
    interesting_headers = [
        "content-length",
        "content-type",
        "content-encoding",
        "etag",
        "accept-ranges",
        "x-linked-size",
        "x-repo-commit",
    ]
    details = [
        f"status={response.status_code}",
        f"url={response.url}",
    ]
    for name in interesting_headers:
        value = headers.get(name)
        if value is not None:
            details.append(f"{name}={value}")
    return "\n  ".join(details)


def atomic_copy_verified(source: Path, target: Path, expected_bytes: int, expected_hash: str) -> None:
    if not verify_file(source, expected_bytes, expected_hash):
        raise SystemExit(
            f"Downloaded artifact failed verification before install: {source}\n"
            f"  expected bytes={expected_bytes} sha256={expected_hash}\n"
            f"  actual   {describe_file_state(source)}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        with source.open("rb") as stream:
            shutil.copyfileobj(stream, handle, length=CHUNK_SIZE)
        handle.flush()

    try:
        if not verify_file(tmp_path, expected_bytes, expected_hash):
            raise SystemExit(f"Internal copy verification failed for {target}")
        tmp_path.replace(target)
    finally:
        tmp_path.unlink(missing_ok=True)


def download_artifact(repo_id: str, revision: str, relative_path: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it or run `hf auth login` "
            "and `hf download` manually, then rerun this script with --verify-only."
        ) from None

    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=relative_path, revision=revision)
    except Exception as exc:  # noqa: BLE001 - keep HF's exact failure context visible.
        raise SystemExit(f"Failed to download {relative_path} from {repo_id}@{revision}: {exc}") from None
    return Path(downloaded)


def download_public_artifact(url_root: str, relative_path: str, target: Path,
                             expected_bytes: int, expected_hash: str,
                             parts: list[dict[str, Any]] | None = None,
                             encoding: str | None = None) -> None:
    """Download through the anonymous proxy without HF credentials."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qdrift-download-", dir=target.parent) as directory:
        downloaded = Path(directory) / "artifact"
        if parts:
            download_public_parts(url_root, parts, downloaded, expected_bytes, expected_hash, encoding)
        else:
            download_public_single(url_root, relative_path, downloaded, expected_bytes, expected_hash)
        downloaded.replace(target)


def download_public_single(url_root: str, relative_path: str, downloaded: Path,
                           expected_bytes: int, expected_hash: str) -> None:
    from urllib.parse import quote
    import requests
    with requests.get(url_root.rstrip("/") + "/" + quote(relative_path, safe="/"),
                      stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with downloaded.open("wb") as handle:
            for block in response.iter_content(CHUNK_SIZE):
                if block:
                    handle.write(block)
    if not verify_file(downloaded, expected_bytes, expected_hash):
        raise SystemExit(
            f"Anonymous artifact checksum mismatch: {relative_path}\n"
            f"  expected bytes={expected_bytes} sha256={expected_hash}\n"
            f"  actual   {describe_file_state(downloaded)}\n"
            f"  response {describe_response(response)}"
        )


def download_public_parts(url_root: str, parts: list[dict[str, Any]], downloaded: Path,
                          expected_bytes: int, expected_hash: str, encoding: str | None) -> None:
    downloaded.unlink(missing_ok=True)
    assembled = downloaded.with_suffix(downloaded.suffix + ".parts")
    assembled.unlink(missing_ok=True)
    try:
        with assembled.open("wb") as output:
            for index, part in enumerate(parts):
                with tempfile.NamedTemporaryFile(prefix=f"part-{index:04d}-", suffix=".tmp",
                                                 dir=downloaded.parent, delete=False) as handle:
                    part_path = Path(handle.name)
                try:
                    download_public_single(url_root, part["remote_path"], part_path, part["bytes"], part["sha256"])
                    with part_path.open("rb") as stream:
                        shutil.copyfileobj(stream, output, length=CHUNK_SIZE)
                finally:
                    part_path.unlink(missing_ok=True)
        if encoding == "gzip":
            with gzip.open(assembled, "rb") as stream, downloaded.open("wb") as output:
                shutil.copyfileobj(stream, output, length=CHUNK_SIZE)
        else:
            assembled.replace(downloaded)
        if not verify_file(downloaded, expected_bytes, expected_hash):
            raise SystemExit(
                f"Reconstructed artifact checksum mismatch after downloading {len(parts)} parts\n"
                f"  expected bytes={expected_bytes} sha256={expected_hash}\n"
                f"  actual   {describe_file_state(downloaded)}"
            )
    finally:
        assembled.unlink(missing_ok=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check local files only; do not import Hugging Face tools or access the network.")
    parser.add_argument("--manifest", type=Path, default=repo_root / "manifests/paper_artifacts.json", help="Artifact manifest with paths, byte sizes, and SHA256 hashes.")
    parser.add_argument("--download-manifest", type=Path, default=repo_root / "manifests/paper_artifacts_download.json", help="Download manifest containing repo_id and revision.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"Artifact manifest must contain a non-empty 'files' object: {args.manifest}")

    download_manifest: dict[str, Any] = {}
    url_root = ""
    if not args.verify_only:
        download_manifest = load_json(args.download_manifest)
        url_root = download_manifest.get("url_root", "")
        repo_id = download_manifest.get("repo_id", "")
        revision = download_manifest.get("revision")
        if not url_root and (not isinstance(repo_id, str) or not repo_id):
            raise SystemExit(f"Download manifest has invalid repo_id: {args.download_manifest}")
        if not url_root and (not isinstance(revision, str) or not revision):
            raise SystemExit(f"Download manifest has invalid revision: {args.download_manifest}")
    else:
        repo_id = ""
        revision = ""

    installed = 0
    verified = 0
    for relative_path, entry in sorted(files.items()):
        if not isinstance(relative_path, str):
            raise SystemExit("Artifact manifest contains a non-string path key.")
        expected_bytes, expected_hash = validate_entry(relative_path, entry)
        target = repo_relative_path(repo_root, relative_path)

        if target.exists():
            if not verify_file(target, expected_bytes, expected_hash):
                raise SystemExit(explain_existing_mismatch(target, expected_bytes, expected_hash))
            verified += 1
            print(f"Verified {relative_path}", flush=True)
            continue

        if args.verify_only:
            raise SystemExit(f"Missing artifact: {target}")

        remote_path = entry.get("remote_path", relative_path)
        repo_relative_path(repo_root, remote_path)  # Validate aliases just like local paths.
        encoding = entry.get("encoding")
        parts = entry.get("parts")
        if parts is not None:
            parts = validate_parts(relative_path, parts)
            for part in parts:
                repo_relative_path(repo_root, part["remote_path"])
        if url_root:
            download_public_artifact(url_root, remote_path, target, expected_bytes, expected_hash, parts, encoding)
        else:
            source = download_artifact(repo_id, revision, remote_path)
            atomic_copy_verified(source, target, expected_bytes, expected_hash)
        installed += 1
        print(f"Installed {relative_path}", flush=True)

    print(f"Paper artifacts ready: verified={verified}, installed={installed}, total={len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

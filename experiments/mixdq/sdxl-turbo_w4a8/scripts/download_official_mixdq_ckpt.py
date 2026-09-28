#!/usr/bin/env python3
"""
Download the official MixDQ SDXL-Turbo PTQ `ckpt.pth` referenced in MixDQ/README.md (Google Drive).

Why: the repo's PTQ checkpoint contains quantization parameters for 2/4/8-bit and is required to
reproduce the paper-quality W4A8 results. Using other artifacts (e.g., kernels/output/new_ckpt.pth)
can lead to severe quality collapse.

This script writes into `../mixdq_qdiff_ckpt/ckpt.pth` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


GDRIVE_FILE_ID = "1m2wS2gpgVtA6HhX-zUnlVWMtVD-et2bK"
EXPECTED_SHA256 = "a26a12ae57883ca8b470c6a22f35c535974d345800e4dfbad1712cedeb57cfbc"


def _download_google_drive(file_id: str, dst: Path) -> None:
    import re
    import requests

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    def _looks_like_html(resp) -> bool:
        ct = (resp.headers.get("content-type") or "").lower()
        if "text/html" in ct:
            return True
        # Some responses omit content-type; sniff the first bytes.
        body = resp.text[:200].lstrip().lower()
        return body.startswith("<!doctype html") or body.startswith("<html")

    def _extract_download_form(resp_text: str) -> tuple[str, dict[str, str]] | None:
        """
        Newer Google Drive flow serves a "Virus scan warning" HTML with a form:
          <form id="download-form" action="https://drive.usercontent.google.com/download">
            <input name="id" value="...">
            <input name="export" value="download">
            <input name="confirm" value="t">
            <input name="uuid" value="...">
        """
        m = re.search(r'<form[^>]+id="download-form"[^>]*action="([^"]+)"', resp_text)
        if not m:
            return None
        action = m.group(1)
        params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', resp_text))
        if not params:
            return None
        return action, params

    def _get_confirm_token(resp) -> str | None:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        m = re.search(r'confirm=([0-9A-Za-z_\\-]+)', resp.text)
        if m:
            return m.group(1)
        return None

    # Probe request (non-stream) so we can parse confirm token / download form.
    probe = session.get(URL, params={"id": file_id})
    probe.raise_for_status()

    download_url = URL
    download_params: dict[str, str] = {"id": file_id}

    if _looks_like_html(probe):
        extracted = _extract_download_form(probe.text)
        if extracted:
            download_url, download_params = extracted
        else:
            token = _get_confirm_token(probe)
            if token:
                download_params = {"id": file_id, "confirm": token}
            else:
                raise RuntimeError(
                    "Google Drive returned an HTML page and no confirm token/form could be parsed. "
                    "This usually means a consent/virus-scan interstitial was returned."
                )

    resp = session.get(download_url, params=download_params, stream=True)
    resp.raise_for_status()

    tmp = dst.with_suffix(dst.suffix + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dst)

    # Sanity check: make sure we didn't save the HTML warning page.
    with open(dst, "rb") as f:
        head = f.read(64).lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError(
            f"Downloaded file looks like HTML (Google Drive interstitial), not a checkpoint: {dst}"
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_id", default=GDRIVE_FILE_ID)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "mixdq_qdiff_ckpt" / "ckpt.pth"),
        help="Output path for ckpt.pth (default: ../mixdq_qdiff_ckpt/ckpt.pth)",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite if the file already exists")
    ap.add_argument("--expected_sha256", default=EXPECTED_SHA256, help="Expected SHA256; pass empty string to skip")
    args = ap.parse_args()

    # NOTE: do NOT call .resolve() here; it follows symlinks. We want to overwrite the path itself.
    out = Path(args.out).expanduser().absolute()
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing file: {out} (pass --force)")

    # If it's a symlink, remove it first.
    if out.is_symlink() or out.exists():
        out.unlink()

    print(f"Downloading MixDQ PTQ checkpoint from Google Drive id={args.file_id}")
    print(f"-> {out}")
    _download_google_drive(args.file_id, out)
    size_mb = out.stat().st_size / (1024 * 1024)
    digest = _sha256(out)
    print(f"Done ({size_mb:.1f} MB)")
    print(f"sha256: {digest}")
    if args.expected_sha256 and digest != args.expected_sha256:
        raise RuntimeError(
            f"Checkpoint checksum mismatch: expected {args.expected_sha256}, got {digest}"
        )


if __name__ == "__main__":
    main()

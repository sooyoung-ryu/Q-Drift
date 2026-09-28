#!/usr/bin/env python3
"""Download pinned MJHQ-30K metadata and build the 5,000-image FID reference folder."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only', action='store_true', help='Download metadata without the image archive.')
    parser.add_argument('--cache-dir', default=None)
    args = parser.parse_args()
    spec = json.loads((ROOT/'manifests/mjhq_splits.json').read_text())
    options = dict(repo_id=spec['repo_id'], revision=spec['revision'], repo_type='dataset', cache_dir=args.cache_dir)
    metadata = Path(hf_hub_download(filename='meta_data.json', **options))
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != spec['metadata_sha256']:
        raise RuntimeError('MJHQ metadata does not match the reference manifest.')
    destination = ROOT/'data/meta_data.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(metadata.read_bytes())
    print('Pinned metadata:', destination, flush=True)
    if not args.metadata_only:
        archive = hf_hub_download(filename='mjhq30k_imgs.zip', **options)
        subprocess.run([sys.executable, str(ROOT/'scripts/build_mjhq_fid_reference.py'),
                        '--meta_path', str(destination), '--zip_path', archive,
                        '--seed', '42', '--num_samples', '5000',
                        '--output_dir', str(ROOT/'data/mjhq_fid_reference')], check=True)


if __name__ == '__main__':
    main()

# Third-Party Notices

This repository includes source code from the following projects. Each keeps its own license; this file is not a license grant for the repository as a whole.

## DeepCompressor (SVDQuant)

- Path: `third_party/deepcompressor/`
- License: Apache License 2.0 (`third_party/deepcompressor/LICENSE`)
- Upstream: https://github.com/mit-han-lab/deepcompressor

## MixDQ

- Path: `third_party/mixdq/quant_utils/`, `third_party/mixdq/mixed_precision_scripts/`
- Upstream: https://github.com/A-suozhang/MixDQ (built on Q-Diffusion and Diffusers)
- The upstream repository has no LICENSE file; check its terms before redistribution.
- The MixDQ W4A8 checkpoint is installed by `scripts/prepare_paper_artifacts.py` (sha256 `a26a12ae57883ca8b470c6a22f35c535974d345800e4dfbad1712cedeb57cfbc`); `experiments/mixdq/sdxl-turbo_w4a8/scripts/download_official_mixdq_ckpt.py` fetches the same file from the link in the upstream README.

## Models, datasets, and Python packages

Base models and MJHQ-30K are downloaded from Hugging Face under their providers' terms. Python packages are installed from `environments/requirements-svdquant.txt` and `environments/requirements-mixdq.txt` under their own licenses.

# Third-Party Notices

This repository includes source code from the following projects. Each keeps its own license; this file is not a license grant for the repository as a whole.

## Diffusers

- Paths include `schedulers/`, `experiments/svdquant/sdxl_w3a4_ptqd/scripts/euler_ptqd.py`, and Diffusers-derived portions of `third_party/mixdq/`.
- License: Apache License 2.0 (see the root `LICENSE`). Existing copyright and license headers remain in place.
- Upstream: https://github.com/huggingface/diffusers
- Q-Drift modifies the scheduler implementations to support quantization-aware sampling corrections. These modifications do not replace the upstream copyright notices.

## k-diffusion

- The Euler scheduler implementations contain functions attributed to k-diffusion.
- Upstream: https://github.com/crowsonkb/k-diffusion
- License: MIT. The upstream notice is reproduced below.

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

## k-diffusion MIT license text

```text
Copyright (c) 2022 Katherine Crowson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

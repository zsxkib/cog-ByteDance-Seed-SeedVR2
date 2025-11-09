# SeedVR2 Cog (3B & 7B)

[![Replicate](https://replicate.com/zsxkib/seedvr2/badge)](https://replicate.com/zsxkib/seedvr2)

# Overview

SeedVR2 Cog packages ByteDance-Seed's one-step diffusion transformer for restoring soft,
noisy, or artifact-heavy footage. This repository packages the official release
as a [Cog](https://github.com/replicate/cog) project so you can run the model on
Replicate or locally with the same configuration.

> **Status (Nov 9 2025):** The predictor now hot-swaps between the SeedVR2 **3B**
> and **7B** checkpoints on a single GPU. The build targets 80 GB accelerators
> (A100) and larger cards (H100/H200+) without relying on multi-GPU parallelism.

## Highlights
- One-step restoration for both videos and single-frame images with optional
  wavelet color correction to match the official Gradio demo.
- Audio passthrough keeps the original soundtrack when producing MP4 outputs.
- Dual-model cache: 3B loads by default, and 7B can be selected at runtime.
- All large assets download from Replicate's CDN via `pget` for reproducible builds.
- CUDA 12.4 / PyTorch 2.4.0 environment mirrors Replicate's production base image.

## What this repository includes
- `predict.py` wraps the original inference code for both video and single-frame
  inputs, returning WebP (by default) or PNG/JPG stills and MP4 video with the
  source audio stream passed through when available.
- `cog.yaml` defines the CUDA 12.4 build with the dependencies Replicate expects,
  including pre-installation of `flash-attn` and runtime checks for Apex.
- `cache_manager.py` helps bundle large weights into tarballs and publish them to
  the Replicate CDN, keeping `cog predict` downloads fast and deterministic.

## Try the model

- **In the browser:** open [replicate.com/zsxkib/seedvr2](https://replicate.com/zsxkib/seedvr2)
- **From the CLI:** install Cog and run:

  ```bash
  cog predict \
    -i media="https://replicate.delivery/pbxt/O0Iv6e7e6eXVjeuEy6bai5SYhkU2EEBxAgYQHp6Lc9kWUHBf/q6e4gnsd5xrma0ct4qt9zt4vd8.mp4" \
    -i sample_steps=1 \
    -i cfg_scale=1 \
    -i sp_size=1 \
    -i fps=24 \
    -i output_format="png" \
    -i output_quality=90
  ```

  The call detects whether `media` is a video or image and selects the correct
  preprocessing and output branch automatically.

## Inputs

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `media` | file / URL | Video (`.mp4`, `.mov`) or image (`.png`, `.jpg`, `.webp`). | – |
| `sample_steps` | integer | Diffusion steps (1 matches the public demo). | `1` |
| `cfg_scale` | float | Guidance strength; raise for stronger sharpening. | `1.0` |
| `sp_size` | integer | Keep at `1` for single-GPU runs; higher values only adjust padding heuristics (no multi-GPU). | `1` |
| `fps` | integer | Frame rate for video outputs. | `24` |
| `seed` | integer? | Optional random seed (auto-randomized when omitted). | random |
| `output_format` | string | Image output format when `media` is an image. | `webp` |
| `output_quality` | integer | JPEG/WebP quality when using lossy formats. | `90` |
| `model_variant` | string | Choose between the 3B and 7B checkpoints (`"3b"` or `"7b"`). | `3b` |
| `apply_color_fix` | boolean | Apply wavelet color correction (matches the official demo). | `false` |

### GPU expectations

- The predictor is optimized for a **single GPU**: 1× A100 80 GB, 1× H100 80/94 GB,
  1× H200 141 GB, or future single-card equivalents.
- `sp_size` remains fixed at `1`; sequence-parallel sharding is not used in this
  deployment. Longer clips are padded internally to satisfy the sampler layout.

## Outputs

- Video inputs return an MP4 file whose audio track is copied from the source
  when `ffmpeg` is present (the Cog image installs it by default).
- Image inputs return a single restored frame in the requested format.

## Run locally

```bash
pip install cog
git clone https://github.com/zsxkib/cog-ByteDance-Seed-SeedVR2.git
cd cog-ByteDance-Seed-SeedVR2
cog predict -i media=@test_videos/01.mp4
```

The first run downloads the cached weights, Apex wheel, and flash-attn build
from the Replicate CDN into `model_cache/`. Subsequent runs reuse those files.

## Weights and caching

- Weight archives live at `https://weights.replicate.delivery/default/seedvr2/`.
- `cache_manager.py` can regenerate the tarballs and upload replacements if the
  underlying checkpoints change.
- Environment variables (`HF_HOME`, `TORCH_HOME`, etc.) are set to `model_cache/`
  before any downloads so cached files persist across runs.

## Development notes

- The predictor removes the need for Hugging Face downloads by mirroring the
  release assets at `https://weights.replicate.delivery/default/seedvr2/`.
- `mux_audio_stream` keeps audio in sync by copying the source stream onto the
  generated video without re-encoding.
- CUDA 12.4 + PyTorch 2.4.0 matches Replicate's `cuda12.4-python3.10-torch2.4.0`
  base image, so no extra drivers are needed for deployment.

## Deploy to Replicate

Create or update a Replicate model pointing at `r8.im/zsxkib/seedvr2`, then:

```bash
cog push r8.im/zsxkib/seedvr2
```

The push builds the Docker image, validates the schema, and publishes the artifacts.

## Citation

Please reference the original SeedVR/SeedVR2 work if you build on these models:

```bibtex
@article{wang2025seedvr2,
  title   = {SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training},
  author  = {Wang, Jianyi and Lin, Shanchuan and Lin, Zhijie and Ren, Yuxi and Wei, Meng and Yue, Zongsheng and Zhou, Shangchen and Chen, Hao and Zhao, Yang and Yang, Ceyuan and Xiao, Xuefeng and Loy, Chen Change and Jiang, Lu},
  journal = {arXiv preprint arXiv:2506.05301},
  year    = {2025}
}

@inproceedings{wang2025seedvr,
  title     = {SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration},
  author    = {Wang, Jianyi and Lin, Zhijie and Wei, Meng and Zhao, Yang and Yang, Ceyuan and Loy, Chen Change and Jiang, Lu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```
## License

- All code in this repository is released under the MIT License (see `LICENSE`).
- Upstream SeedVR/SeedVR2 weights and research assets remain under Apache 2.0 per
  ByteDance-Seed’s terms. Bring your own weights or follow their license when you
  redistribute the checkpoints.

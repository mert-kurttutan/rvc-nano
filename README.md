# RVC Nano

A minimal, dependency-light subset of the original RVC (Retrieval-based Voice Conversion) project.
The goal is to keep core voice-conversion functionality while making the codebase easier to
port to other platforms (e.g., Tenstorrent) and easier to embed in constrained environments.

## Goals
- Minimal dependencies and small runtime footprint
- Keep core RVC inference functionality
- Clean, portable architecture for non-Python or accelerator-backed ports

## What is included
- Core conversion pipeline (inference-focused)
- Lightweight configuration and CLI surface
- Clear module boundaries for future ports

## What is not included
- Heavy training pipelines and experimental tooling
- Large pretrained models or indices (use external assets)

## Getting started
This have similar functionaities to the original RVC. To get started, install the package and run inference:

Example run:
```bash
uv run rvc infer -i speech-sample-01.wav -o output.wav
```

If you need the full-featured project, use the upstream repository:
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion

## Repository structure (high level)
- `rvc/`: core Python package
- `rvc/modules/`: VC pipeline and model-related modules
- `rvc/lib/`: shared utilities
- `rvc/configs/`: model configs and defaults
- `rvc/wrapper/`: CLI and API wrappers

## Status
Early and intentionally minimal. Expect missing features and breaking changes while the
interface is refined.

## Attributions
Heavily inspired by RVC (Retrieval-based Voice Conversion) and please see the original repo
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion

## License
See `LICENSE`.

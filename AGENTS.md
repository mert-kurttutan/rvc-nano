# Repository Guidelines

## Project Structure & Module Organization
- `rvc/` contains the Python package. Core logic lives under `rvc/modules/` (VC pipeline, UVR, ONNX) and shared utilities in `rvc/lib/`.
- `rvc/configs/` holds model configs (`v1/`, `v2/`) and project defaults.
- `rvc/wrapper/cli/` and `rvc/wrapper/api/` define the CLI and FastAPI server.
- `docs/` stores localized documentation.
- Runtime artifacts are created by `rvc init`: `assets/` for models and `result/` for outputs. Do not commit large model files.

## Build, Test, and Development Commands
- `pip install git+https://github.com/RVC-Project/Retrieval-based-Voice-Conversion` installs the library from source.
- `uv pip install -e .` installs development dependencies from `pyproject.toml`.
- `rvc init` creates `assets/` and `.env` in your working directory.
- `rvc infer -m {model.pth} -i {input.wav} -o {output.wav}` runs a local inference.
- `uv run rvc-api` starts the FastAPI server at `http://127.0.0.1:8000`.
- `./docker-run.sh` builds and runs the Docker image (see `Dockerfile`).

## Coding Style & Naming Conventions
- Python uses 4-space indentation, PEP 8–style naming (`snake_case` functions, `PascalCase` classes).
- Keep CLI options consistent with existing `rvc` commands and reuse config keys from `rvc/configs/`.
- Prefer descriptive module names (e.g., `pipeline.py`, `modules.py`) and avoid large monolithic files.

## Testing Guidelines
- There is no dedicated test suite in this repository.
- Validate changes by running a small inference via `rvc infer` and, for API changes, a `curl` request from `README.md`.
- If you add tests, place them in a top-level `tests/` directory and name them `test_*.py` (pytest).

## Commit & Pull Request Guidelines
- Recent commits favor conventional-style subjects (e.g., `fix(deps): ...`) and include PR/issue references.
- Keep commits scoped and descriptive; prefer one logical change per commit.
- PRs should include: a brief summary, reproduction/validation steps, and any new assets or config updates.

## Security & Configuration Tips
- Store local paths and model locations in `.env`; avoid hardcoding file system paths.
- Keep `assets/weights`, `assets/indices`, and `assets/audios` out of version control unless explicitly requested.

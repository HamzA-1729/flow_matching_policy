# Contributing to `flow_matching_policy`

Contributions, bug reports, and mathematical verification improvements are welcome.

## Development Setup

1. Clone the repository and sync the isolated `uv` environment:
   ```bash
   git clone https://github.com/HamzA-1729/flow_matching_policy.git
   cd flow_matching_policy
   uv sync
   ```
2. Run the automated test suite before submitting changes:
   ```bash
   uv run pytest -v -m "not slow"
   uv run pytest -v -m slow
   ```
3. Format and lint with `ruff`:
   ```bash
   uv run ruff check src tests
   ```

## Standards
- Keep implementations pure PyTorch without high-level policy/RL wrappers.
- Every new mathematical invariant, schedule bound, or environment rule must include a deterministic unit test in `tests/`.

# Contributing to the Omni-Retail Customer Analysis Platform

Thank you for your interest in contributing to this project!

## Development Setup
1. Clone the repository.
2. Install dependencies via `conda env create -f environment.yml` or `pip install -r requirements.txt`.
3. Ensure you have a CUDA-compatible GPU if you plan to run the RAPIDS cuML pipelines.

## Pull Request Process
1. Create a feature branch (`git checkout -b feature/your-feature-name`).
2. Ensure your code does not introduce target leakage.
3. Update the `CHANGELOG.md`.
4. Submit a Pull Request.

## Code Style
- Follow PEP 8 guidelines.
- Use explicit random seeds for reproducibility.
- Add docstrings to all custom classes and functions in the `src/` directory.

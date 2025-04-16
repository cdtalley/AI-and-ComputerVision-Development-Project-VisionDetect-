# Contributing to VisionDetect

Thank you for considering contributing to VisionDetect! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How to Contribute

There are many ways to contribute to VisionDetect:

1. Reporting bugs
2. Suggesting enhancements
3. Writing documentation
4. Submitting code changes
5. Reviewing pull requests

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any relevant logs or screenshots
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

If you have an idea for an enhancement, please create an issue with:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples or mockups
- Why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. Clone the repository
2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```
4. Install pre-commit hooks
   ```bash
   pre-commit install
   ```

## Coding Standards

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions, classes, and modules
- Include unit tests for new features
- Ensure all tests pass before submitting a pull request

## Testing

Run tests with pytest:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=src tests/
```

## Linting and Formatting

We use the following tools for code quality:

- flake8 for linting
- black for code formatting
- isort for import sorting

Run these tools with:

```bash
flake8 .
black .
isort .
```

## Documentation

- Update documentation for any changes to the API
- Add examples for new features
- Ensure documentation builds without errors

## Commit Messages

- Use clear, descriptive commit messages
- Reference issue numbers when applicable
- Follow the conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `style:` for formatting changes
  - `refactor:` for code refactoring
  - `test:` for adding or modifying tests
  - `chore:` for maintenance tasks

## Versioning

We follow [Semantic Versioning](https://semver.org/).

## License

By contributing to VisionDetect, you agree that your contributions will be licensed under the project's MIT License.

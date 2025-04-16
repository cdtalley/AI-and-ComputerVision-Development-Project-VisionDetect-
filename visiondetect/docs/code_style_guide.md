# VisionDetect Code Style Guide

This document outlines the coding standards and style guidelines for the VisionDetect project.

## Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a few project-specific additions.

### Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length of 88 characters (Black default)
- Use blank lines to separate logical sections of code
- End files with a single newline

### Naming Conventions

- `snake_case` for variables, functions, and methods
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Prefix private attributes with a single underscore (`_private_var`)

### Imports

- Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
- Use absolute imports when possible
- Sort imports alphabetically within each group

Example:
```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
import numpy as np
import torch
import torch.nn as nn

# Local
from src.data.preprocessing import DataProcessor
from src.utils.metrics import calculate_map
```

### Docstrings

We use Google-style docstrings:

```python
def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    
    Returns:
        bool: The return value. True for success, False otherwise.
    
    Raises:
        ValueError: If param1 is negative.
    """
    if param1 < 0:
        raise ValueError("param1 must be positive")
    return True
```

### Type Hints

Use type hints for function signatures:

```python
def greeting(name: str) -> str:
    return f"Hello {name}"
```

### Comments

- Use comments sparingly - prefer self-documenting code
- Comments should explain "why", not "what"
- Keep comments up-to-date with code changes

### Error Handling

- Use specific exception types
- Provide informative error messages
- Use context managers (`with` statements) for resource management

## Testing

- Write unit tests for all functions and classes
- Aim for high test coverage (>80%)
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert pattern

## Code Organization

- Keep functions and methods short and focused
- Follow the Single Responsibility Principle
- Use classes to encapsulate related functionality
- Separate interface from implementation

## Version Control

- Write clear, descriptive commit messages
- Keep commits focused on a single change
- Reference issue numbers in commit messages when applicable

## Tools

We use the following tools to enforce code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Static type checking
- **pytest**: Testing

## Pre-commit Hooks

We use pre-commit hooks to automatically check code quality before commits:

```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        args: ["--profile", "black"]

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]
```

## Conclusion

Following these guidelines will help maintain a consistent, high-quality codebase that is easy to read, understand, and maintain.

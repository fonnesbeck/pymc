# CLAUDE.md - PyMC AI Assistant Guide

This document provides comprehensive guidance for AI assistants working with the PyMC codebase.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
- [Testing Strategy](#testing-strategy)
- [Key Architecture Concepts](#key-architecture-concepts)
- [Common Tasks](#common-tasks)
- [Important Conventions](#important-conventions)
- [Resources](#resources)

---

## Project Overview

**PyMC** (formerly PyMC3) is a Python package for Bayesian statistical modeling focusing on advanced Markov chain Monte Carlo (MCMC) and variational inference (VI) algorithms.

### Core Features
- Intuitive model specification syntax (e.g., `x ~ N(0,1)` → `x = Normal('x',0,1)`)
- Powerful sampling algorithms (No U-Turn Sampler, ADVI, SMC)
- Built on **PyTensor** (computational backend)
- Integrated with **ArviZ** (plotting and diagnostics)
- Support for complex models with thousands of parameters

### Key Dependencies
- **PyTensor** (≥2.30.2, <2.31): Gradient computation, random number generation, tensor operations
- **ArviZ** (≥0.13.0): Plotting, MCMC diagnostics, model comparison
- **NumPy** (≥1.25.0), **SciPy** (≥1.4.1), **Pandas** (≥0.24.0): Numerical computing
- **Python 3.10-3.13**: Minimum Python 3.10 required

### Project Links
- Documentation: https://docs.pymc.io
- Examples: https://www.pymc.io/projects/examples/en/latest/gallery.html
- Discourse Forum: https://discourse.pymc.io
- GitHub: https://github.com/pymc-devs/pymc

---

## Repository Structure

```
pymc/
├── .github/
│   └── workflows/          # CI/CD pipelines (tests, mypy, docker, release)
├── benchmarks/             # Performance benchmarks
├── conda-envs/             # Conda environment specifications
├── docs/
│   ├── source/            # Sphinx documentation source
│   └── logos/             # Brand assets
├── pymc/                  # Main package directory
│   ├── backends/          # Storage backends for sampling results
│   ├── distributions/     # Probability distributions
│   │   ├── distribution.py      # Base distribution classes
│   │   ├── logprob.py          # Log probability calculations
│   │   └── dist_math.py        # Distribution math utilities
│   ├── gp/               # Gaussian processes
│   ├── logprob/          # Log probability transformations
│   ├── model/            # Model definition and core functionality
│   │   └── core.py       # Model class, ContextMeta, Factor, ValueGrad
│   ├── ode/              # Ordinary differential equations
│   ├── plots/            # Plotting utilities (defers to ArviZ)
│   ├── sampling/         # MCMC and sampling infrastructure
│   ├── smc/              # Sequential Monte Carlo
│   ├── stats/            # Statistical utilities
│   ├── step_methods/     # MCMC step methods (HMC, NUTS, Metropolis, etc.)
│   ├── tuning/           # Sampler tuning algorithms
│   ├── variational/      # Variational inference (ADVI)
│   ├── blocking.py       # Variable blocking for samplers
│   ├── data.py           # Data containers (Mutable, ConstantData)
│   ├── exceptions.py     # Custom exceptions
│   ├── initial_point.py  # Initial point finding for samplers
│   ├── math.py           # Mathematical utilities
│   ├── model_graph.py    # Model visualization (GraphViz, NetworkX)
│   ├── printing.py       # Model printing and repr
│   ├── pytensorf.py      # PyTensor interface functions
│   ├── testing.py        # Testing utilities and helpers
│   ├── util.py           # General utilities
│   └── __init__.py       # Package exports
├── scripts/              # Utility scripts
├── tests/                # Test suite (mirrors pymc/ structure)
│   ├── conftest.py       # Pytest configuration and fixtures
│   ├── helpers.py        # Test helpers
│   └── [mirrors pymc/]   # Test modules parallel to source
├── ARCHITECTURE.md       # High-level architecture documentation
├── CONTRIBUTING.md       # Contribution guidelines (points to docs)
├── CODE_OF_CONDUCT.md    # Community guidelines
├── GOVERNANCE.md         # Project governance structure
├── pyproject.toml        # Build config, tool settings (ruff, mypy, pytest)
├── setup.py              # Package installation
├── requirements.txt      # Runtime dependencies
└── requirements-dev.txt  # Development dependencies
```

### Key Files
- **pyproject.toml**: Central configuration for build, linting (ruff), type checking (mypy), testing (pytest)
- **.pre-commit-config.yaml**: Pre-commit hooks for code quality
- **ARCHITECTURE.md**: Detailed architecture documentation
- **CONTRIBUTING.md**: Links to comprehensive contribution docs

---

## Development Workflow

### Setting Up Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/pymc-devs/pymc.git
   cd pymc
   ```

2. **Install dependencies**
   ```bash
   # Using conda (recommended)
   conda env create -f conda-envs/environment-dev.yml
   conda activate pymc-dev

   # Using pip
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Branch Strategy
- **main**: Primary development branch
- Feature branches: Use descriptive names (e.g., `feature/improve-nuts-sampler`)
- Branches for Claude: Format `claude/claude-md-<session-id>`

### Git Commit Conventions
- Use clear, descriptive commit messages
- Format: Start with verb in present tense (e.g., "Add feature", "Fix bug", "Update docs")
- Reference issues when applicable (e.g., "Fix #1234: Issue description")

### Pull Request Process
1. Ensure all tests pass locally
2. Run pre-commit hooks
3. Update documentation if needed
4. Follow the PR checklist: https://docs.pymc.io/en/latest/contributing/pr_checklist.html
5. Link related issues

---

## Code Style and Standards

### Python Style
PyMC uses **Ruff** for linting and formatting (configured in `pyproject.toml`).

#### Key Settings
- **Line length**: 100 characters
- **Target Python**: 3.10+
- **Docstring convention**: NumPy style
- **Import sorting**: Enabled with line separation between types

#### Ruff Rules (Selected)
- `C4`: Comprehensions
- `D`: Docstrings (numpy convention, many checks disabled for brevity)
- `E/W`: PEP 8 style
- `F`: Pyflakes
- `I`: Import sorting
- `UP`: Modern Python upgrades
- `RUF`: Ruff-specific rules
- `T20`: No print statements (except in scripts/notebooks)

#### Ignored Rules
- `E501`: Line too long (handled by formatter)
- `F841`: Unused local variables
- `RUF001/002`: Ambiguous characters (Greek letters used in math)
- `D100-D103, D105`: Missing docstrings (relaxed for internal code)

### Running Formatters

```bash
# Auto-format code
ruff format .

# Auto-fix linting issues
ruff check --fix .

# Run pre-commit on all files
pre-commit run --all-files
```

### Type Hints
- **MyPy** is used for type checking (config in `pyproject.toml`)
- Type hints encouraged but not strictly enforced
- Run: `python scripts/run_mypy.py` or check `.github/workflows/mypy.yml`

### Pre-commit Hooks
The repository uses extensive pre-commit hooks:
- Ruff formatting and linting
- Trailing whitespace removal
- YAML/TOML validation
- Python checks (no eval, proper type annotations, etc.)
- Apache license headers
- Sphinx documentation linting
- No hard-coded URLs in docs (use cross-references)

---

## Testing Strategy

### Test Organization
Tests mirror the `pymc/` directory structure in `tests/`:
```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── helpers.py               # Common test utilities
├── models.py                # Reusable test models
├── sampler_fixtures.py      # Sampler test fixtures
├── test_*.py                # Test modules at root level
├── distributions/           # Distribution tests
├── step_methods/            # Step method tests
└── [other directories]      # Mirror pymc/ structure
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::test_model_creation

# Run with coverage
pytest --cov=pymc

# Run tests matching pattern
pytest -k "test_normal"
```

### Test Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
minversion = "6.0"
xfail_strict = true
addopts = ["--color=yes"]
```

### CI/CD Testing
Tests run on GitHub Actions (`.github/workflows/tests.yml`):
- **Split into multiple jobs** for parallel execution
- **Matrix strategy**: Different Python versions, OS (Ubuntu, Windows), float precision
- **Selective test subsets** to balance job duration
- **Change detection**: Tests only run if relevant files change

### Writing Tests
- Use pytest fixtures from `conftest.py` and `sampler_fixtures.py`
- Use helper functions from `tests/helpers.py`
- Follow existing patterns in similar test files
- Use `pymc.testing` utilities for model comparisons
- Ensure tests are deterministic (use random seeds)

### Test Coverage
- Target: High coverage for core modules
- Coverage reports generated via codecov
- Exclude patterns in `[tool.coverage.report]` section of `pyproject.toml`

---

## Key Architecture Concepts

### 1. Model Context Manager
The `with pm.Model()` syntax is enabled by `ContextMeta` in `pymc.model.core`:
```python
with pm.Model() as model:
    theta = pm.Beta("theta", alpha=1, beta=2)
    y = pm.Bernoulli("y", p=theta, observed=data)
```
- Variables are automatically registered to the model context
- Context can be nested
- `pm.modelcontext()` retrieves the current model

### 2. Distributions
Located in `pymc/distributions/`:
- **Base class**: `distribution.py` defines the distribution API
- **Log probability**: Calculated via PyTensor in `logprob.py`
- **Distribution types**:
  - Continuous: `continuous.py`
  - Discrete: `discrete.py`
  - Multivariate: `multivariate.py`
  - Time series: `timeseries.py`
  - Special: `mixture.py`, `censored.py`, `truncated.py`

#### Key Distribution Concepts
- `observed=` parameter: Differentiates likelihood from prior
- `shape=` and `dims=`: Control dimensionality
- `dist()` class method: Creates distributions without registration

### 3. Sampling Infrastructure
Located in `pymc/sampling/`:
- `pm.sample()`: Main entry point for MCMC sampling
- Automatically selects appropriate sampler (NUTS for continuous, Metropolis otherwise)
- Returns `InferenceData` (ArviZ format)

### 4. Step Methods
Located in `pymc/step_methods/`:
- **HMC/NUTS**: `hmc/` directory (Hamiltonian Monte Carlo)
- **Metropolis**: Various Metropolis variants
- **Compound steps**: Combine multiple step methods
- **Integrators**: Leapfrog integration for HMC

### 5. PyTensor Integration
- PyMC builds computational graphs using PyTensor
- Key module: `pytensorf.py` provides PyTensor utility functions
- `ValueGrad`: Connects PyMC to PyTensor's gradient computation
- JIT compilation: PyTensor can compile to C or JAX

### 6. Model Transformation
New in recent versions: `pm.do()` and `pm.observe()` for causal modeling
- `pm.do(model, interventions)`: Intervene on variables
- `pm.observe(model, observations)`: Condition on observations

### 7. Functionality Deferred to Other Libraries
- **PyTensor**: Gradients, RNG, low-level tensor ops, computational graphs
- **ArviZ**: Plotting, MCMC diagnostics (R-hat, ESS), model comparison, InferenceData

---

## Common Tasks

### Adding a New Distribution

1. **Choose the appropriate module** in `pymc/distributions/`:
   - `continuous.py`, `discrete.py`, `multivariate.py`, etc.

2. **Define the distribution class**:
   ```python
   class NewDistribution(Continuous):
       rv_op = new_distribution_op  # PyTensor RandomVariable op

       @classmethod
       def dist(cls, param1, param2, **kwargs):
           # Define distribution parameterization
           return super().dist([param1, param2], **kwargs)
   ```

3. **Implement logprob** in `logprob.py` (if custom logic needed)

4. **Add tests** in `tests/distributions/test_*.py`

5. **Add to exports** in `pymc/distributions/__init__.py`

6. **Document** in docstring (NumPy style)

### Modifying Sampling Behavior

1. **Step methods**: Edit in `pymc/step_methods/`
2. **Sampler selection**: Modify `pymc/sampling/mcmc.py`
3. **Initialization**: Edit `pymc/initial_point.py`

### Adding Model Functionality

1. **Edit `pymc/model/core.py`** for core model features
2. **Add to `Model` class** for new methods
3. **Update model graph** in `model_graph.py` if visualization affected

### Updating Documentation

1. **Edit source** in `docs/source/`
2. **Build locally**: `cd docs && make html`
3. **View**: Open `docs/build/html/index.html`
4. **Follow conventions**: Use MyST Markdown or RST

---

## Important Conventions

### For AI Assistants

#### 1. Read Before Writing
- Always read existing code before modifying
- Check `ARCHITECTURE.md` for design patterns
- Look at similar existing implementations

#### 2. Preserve PyMC Patterns
- Use context managers (`with pm.Model()`)
- Follow distribution API conventions
- Maintain PyTensor integration patterns
- Respect the separation: PyMC (model), PyTensor (computation), ArviZ (diagnostics)

#### 3. Testing is Mandatory
- Add tests for all new functionality
- Mirror source structure in tests
- Use existing test fixtures and helpers
- Ensure deterministic tests (random seeds)

#### 4. Documentation Standards
- NumPy-style docstrings
- Include mathematical notation where appropriate (LaTeX in docstrings)
- Provide examples in docstrings
- Update user-facing docs in `docs/source/`

#### 5. Code Quality
- Run `ruff format` and `ruff check --fix` before committing
- Fix any MyPy errors if touching type-sensitive code
- Pre-commit hooks will catch many issues
- Line length: 100 characters

#### 6. Performance Considerations
- PyMC models can have thousands of parameters
- Avoid Python loops where possible (use PyTensor operations)
- Profile before optimizing
- Benchmark changes in `benchmarks/`

#### 7. Backwards Compatibility
- PyMC is widely used in production
- Maintain API compatibility when possible
- Deprecation warnings for breaking changes
- Document migration paths

#### 8. Scientific Rigor
- Verify mathematical correctness
- Compare against reference implementations
- Check statistical properties in tests
- Cite relevant papers in docstrings

#### 9. Community Interaction
- Check Discourse forum for context on issues
- Reference relevant discussions in PRs
- Be respectful of contributor time
- Follow governance guidelines in `GOVERNANCE.md`

### Specific Code Patterns

#### Model Definition
```python
with pm.Model() as model:
    # Priors
    mu = pm.Normal("mu", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data)
```

#### Distribution Usage
```python
# Creating a prior
x = pm.Normal("x", mu=0, sigma=1)

# Creating a likelihood
y = pm.Normal("y", mu=x, sigma=1, observed=data)

# Creating unregistered distribution
dist = pm.Normal.dist(mu=0, sigma=1)
samples = pm.draw(dist, draws=1000)
```

#### Sampling
```python
with model:
    # Prior predictive
    prior_pred = pm.sample_prior_predictive()

    # Posterior
    trace = pm.sample(1000, tune=1000, chains=4)

    # Posterior predictive
    post_pred = pm.sample_posterior_predictive(trace)
```

#### Testing Pattern
```python
import pymc as pm
import numpy as np
import pytest

def test_new_feature():
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1)

    # Test model properties
    assert len(model.basic_RVs) == 1

    # Test sampling
    with model:
        trace = pm.sample(100, tune=100, random_seed=42)

    # Assertions on results
    assert trace.posterior["x"].shape == (4, 100)  # 4 chains, 100 draws
```

---

## Resources

### Documentation
- **Main docs**: https://docs.pymc.io
- **API reference**: https://docs.pymc.io/en/stable/api.html
- **Contributing guide**: https://docs.pymc.io/en/latest/contributing/index.html
- **PR tutorial**: https://docs.pymc.io/en/latest/contributing/pr_tutorial.html

### Examples and Learning
- **Examples gallery**: https://www.pymc.io/projects/examples/en/latest/gallery.html
- **PyMC overview**: https://docs.pymc.io/en/latest/learn/core_notebooks/pymc_overview.html
- **API quickstart**: https://www.pymc.io/projects/examples/en/latest/introductory/api_quickstart.html

### Community
- **Discourse forum**: https://discourse.pymc.io
- **GitHub issues**: https://github.com/pymc-devs/pymc/issues
- **X/Twitter**: @pymc_devs
- **LinkedIn**: @pymc
- **YouTube**: @PyMCDevelopers

### Related Projects
- **PyTensor**: https://pytensor.readthedocs.io (computational backend)
- **ArviZ**: https://python.arviz.org (diagnostics and plotting)
- **Bambi**: https://github.com/bambinos/bambi (high-level modeling interface)
- **CausalPy**: https://github.com/pymc-labs/CausalPy (causal inference)

### Papers
- **PyMC paper (2023)**: Abril-Pla et al., PeerJ Computer Science, DOI: 10.7717/peerj-cs.1516
- **NUTS sampler**: Hoffman & Gelman (2014), JMLR
- **ADVI**: Kucukelbir et al. (2017), JMLR

### Key Modules to Understand

For comprehensive understanding, focus on these modules in order:

1. **pymc/model/core.py**: Model definition, context management
2. **pymc/distributions/distribution.py**: Distribution base classes
3. **pymc/sampling/mcmc.py**: MCMC sampling interface
4. **pymc/step_methods/hmc/nuts.py**: NUTS implementation
5. **pymc/pytensorf.py**: PyTensor integration

### Debugging Tips

1. **Enable PyTensor warnings**: Set `pytensor.config.warn_float64='warn'`
2. **Visualize model**: Use `pm.model_to_graphviz(model)`
3. **Check log probability**: `model.compile_logp()(model.initial_point())`
4. **Profile sampling**: Use `pm.sample(..., return_inferencedata=True)` and examine warnings
5. **Test point evaluation**: `model.compile_fn(var)(test_point)`

---

## Version Information

This guide is current as of **November 2025**.

- **PyMC version**: 5.x (in active development)
- **Python support**: 3.10, 3.11, 3.12, 3.13
- **PyTensor**: ≥2.30.2, <2.31
- **License**: Apache License 2.0

For the latest information, always refer to:
- Repository: https://github.com/pymc-devs/pymc
- Documentation: https://docs.pymc.io

---

## Quick Reference Commands

```bash
# Development setup
conda env create -f conda-envs/environment-dev.yml
conda activate pymc-dev
pip install -e .
pre-commit install

# Code quality
ruff format .                    # Format code
ruff check --fix .               # Lint and fix
pre-commit run --all-files       # Run all pre-commit hooks
python scripts/run_mypy.py       # Type checking

# Testing
pytest                           # Run all tests
pytest tests/test_model.py       # Run specific test file
pytest --cov=pymc                # With coverage
pytest -k "normal"               # Run tests matching pattern

# Documentation
cd docs && make html             # Build docs locally

# Git workflow
git checkout -b feature/my-feature
git add .
git commit -m "Add feature X"
git push -u origin feature/my-feature
```

---

## Contact and Support

- **Questions**: Post on [Discourse](https://discourse.pymc.io) under "Questions" category
- **Bug reports**: Open an issue on [GitHub](https://github.com/pymc-devs/pymc/issues)
- **Feature requests**: Discuss on [Discourse](https://discourse.pymc.io) under "Development" category
- **Professional consulting**: [PyMC Labs](https://www.pymc-labs.io)

---

**Remember**: PyMC is a community-driven project. Be respectful, thorough, and scientifically rigorous in all contributions. When in doubt, ask on Discourse or reference existing patterns in the codebase.

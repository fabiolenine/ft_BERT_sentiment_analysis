# Contributing to BERT Sentiment Analysis

First off, thank you for considering contributing to this project! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- CUDA-compatible GPU (optional but recommended)

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/ft_BERT_sentiment_analysis.git
   cd ft_BERT_sentiment_analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install development dependencies**
   ```bash
   pip install pytest pytest-cov black flake8 isort
   ```

5. **Run tests to ensure everything works**
   ```bash
   pytest
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug fixes**
- ✨ **New features**
- 📚 **Documentation improvements**
- 🧪 **Tests**
- 🔧 **Code refactoring**
- 📊 **Performance improvements**

### Areas for Contribution

1. **Model Improvements**
   - Hyperparameter tuning
   - New model architectures
   - Ensemble methods

2. **Data Processing**
   - Data augmentation techniques
   - Better preprocessing methods
   - Support for new datasets

3. **API Enhancements**
   - New endpoints
   - Performance optimizations
   - Better error handling

4. **Infrastructure**
   - Docker improvements
   - CI/CD enhancements
   - Monitoring and logging

5. **Documentation**
   - README improvements
   - Code documentation
   - Tutorials and examples

## Development Setup

### Project Structure

```
ft_BERT_sentiment_analysis/
├── src/                        # Source code
│   └── bert_sentiment_trainer.py
├── tests/                      # Test files
├── logs/                       # Log files
├── models/                     # Model storage
├── data/                       # Data storage
├── notebooks/                  # Jupyter notebooks
├── .github/workflows/          # GitHub Actions
├── api.py                      # FastAPI application
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # Documentation
```

### Setting up Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will run code formatting and linting before each commit.

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use isort
- **Code formatting**: Use Black
- **Linting**: Use flake8

### Formatting Commands

```bash
# Format code
black .

# Sort imports
isort .

# Check linting
flake8 .
```

### Documentation Style

- Use Google-style docstrings
- Include type hints
- Add examples for complex functions
- Keep comments clear and concise

Example:
```python
def train_model(data: pd.DataFrame, epochs: int = 3) -> Dict[str, float]:
    """
    Train the BERT model on sentiment data.
    
    Args:
        data: DataFrame containing text and labels
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Example:
        >>> metrics = train_model(df, epochs=5)
        >>> print(metrics['accuracy'])
        0.85
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_trainer.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use fixtures for common setup

Example test structure:
```python
import pytest
from src.bert_sentiment_trainer import BERTSentimentTrainer

class TestBERTSentimentTrainer:
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        trainer = BERTSentimentTrainer()
        assert trainer.model is not None
        assert trainer.tokenizer is not None
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        trainer = BERTSentimentTrainer()
        with pytest.raises(ValueError):
            trainer.predict("")
```

## Pull Request Process

### Before Submitting

1. **Fork and clone** the repository
2. **Create a feature branch** from `main`
3. **Make your changes** with clear commit messages
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run the full test suite**
7. **Ensure code formatting** is correct

### Commit Messages

Use clear and descriptive commit messages:

```
feat: add support for custom tokenizers
fix: resolve memory leak in training loop
docs: update README with new examples
test: add unit tests for model validation
refactor: improve code structure in trainer
```

### Pull Request Template

When submitting a PR, include:

- **Description**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Changes**: List of key changes made
- **Testing**: How was this tested?
- **Screenshots**: If applicable

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation** review
5. **Merge** after approval

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, dependencies
- **Steps to reproduce**: Clear step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happened
- **Error messages**: Full stack traces
- **Additional context**: Any relevant information

### Feature Requests

For new features, please include:

- **Use case**: Why is this feature needed?
- **Description**: Detailed explanation of the feature
- **Implementation**: Suggested approach (if any)
- **Alternatives**: Other solutions considered

## Development Guidelines

### Best Practices

1. **Code Quality**
   - Write clean, readable code
   - Follow SOLID principles
   - Use meaningful variable names
   - Add comprehensive comments

2. **Performance**
   - Profile code for bottlenecks
   - Use efficient algorithms
   - Consider memory usage
   - Test with large datasets

3. **Error Handling**
   - Use appropriate exception types
   - Provide helpful error messages
   - Log errors appropriately
   - Fail gracefully

4. **Security**
   - Validate all inputs
   - Don't expose sensitive information
   - Use secure coding practices
   - Run security checks

### Git Workflow

1. **Branch naming**: `feature/description`, `bugfix/issue-number`, `docs/update-readme`
2. **Commit often**: Small, focused commits
3. **Rebase before merge**: Keep history clean
4. **Delete branches**: Clean up after merge

## Getting Help

### Resources

- **Documentation**: Check the README and code comments
- **Issues**: Search existing issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join our Discord/Slack (if applicable)

### Contact

- **Maintainer**: [Your Name](mailto:fabiolenine@gmail.com)
- **Issues**: Use GitHub Issues for bug reports
- **Security**: Report security issues privately

## Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **Releases**: Changelog mentions
- **Documentation**: Author credits

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Happy Contributing!** 🚀

Thank you for helping make this project better for everyone!
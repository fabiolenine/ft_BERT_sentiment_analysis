#!/usr/bin/env python3
"""
Setup script for BERT Sentiment Analysis project
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="bert-sentiment-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fine-tuning BERT for Portuguese sentiment analysis with professional logging and model versioning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ft_BERT_sentiment_analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.0.0",
            "python-multipart>=0.0.6",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bert-sentiment-train=src.bert_sentiment_trainer:main",
            "bert-sentiment-api=api:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "bert",
        "sentiment-analysis",
        "nlp",
        "pytorch",
        "transformers",
        "machine-learning",
        "deep-learning",
        "portuguese",
        "fine-tuning",
        "huggingface",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ft_BERT_sentiment_analysis/issues",
        "Source": "https://github.com/yourusername/ft_BERT_sentiment_analysis",
        "Documentation": "https://github.com/yourusername/ft_BERT_sentiment_analysis/blob/main/README.md",
    },
)
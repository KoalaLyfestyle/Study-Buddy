from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="study-buddy-rag",
    version="0.1.0",
    author="Study Buddy Team",
    description="A lightweight offline RAG study assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KoalaLyfestyle/Study-Buddy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "full": [
            "langchain>=0.1.0",
            "langchain-community>=0.0.10",
            "chromadb>=0.4.20",
            "sentence-transformers>=2.2.0",
            "pypdf2>=3.0.0",
            "ollama>=0.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "study-buddy=study_buddy.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md"],
    },
)
from setuptools import setup, find_packages

setup(
    name="circuit_subspace",
    version="0.1.0",
    description="SVD-based Transformer Circuit Discovery",
    author="Areeb",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "transformer-lens>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "ipywidgets",
            "plotly",
            "wandb",
        ]
    }
)

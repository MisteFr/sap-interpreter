from setuptools import setup, find_packages

setup(
    name="sap-interpreter",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "plotly>=5.0.0",
        "tqdm>=4.60.0",
        "transformers==4.45.2",
        "crlhf"
    ],
    entry_points={
        "console_scripts": [
            "compute-edge-violations=compute_edge_violations:main",
            "extract-sae-activations=extract_sae_activations:main"
        ],
    },
    author="Arthur Bigot",
    author_email="bigot.arthur@gmail.com",
    description="SaP Interpreter - A tool for interpreting SaP",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MisteFr/sap-interpreter",
    keywords="SaP, interpreter, polytope, safety, ai, ml, mechanistic interpretability",
    python_requires=">=3.7",
    classifiers=[],
) 
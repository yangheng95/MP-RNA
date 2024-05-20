








from pathlib import Path

from setuptools import setup, find_packages

from omnigenome import __name__, __version__

cwd = Path(__file__).parent
long_description = (cwd / "README.md").read_text(encoding="utf8")

extras = {}
extras["dev"] = [
    "dill",
    "pytest",
]


setup(
    name=__name__,
    version=__version__,
    description="OmniGenome: A comprehensive toolkit for genome analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(),
    include_package_data=True,
    exclude_package_date={"": [".gitignore"]},
    license="MIT",
    install_requires=[
        "findfile>=2.0.0",
        "autocuda>=0.16",
        "metric-visualizer>=0.9.6",
        "tqdm",
        "termcolor",
        "gitpython",  
        "transformers>=4.37.0",
        "torch>=1.0.0",
        "sentencepiece",
        "protobuf<4.0.0",
        "pandas",
        "viennarna",
    ],
    extras_require=extras,
)

from setuptools import setup, find_packages
import os

readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="qu-ssm",
    version="1.0.0",
    author="Prannessh K.V.A.",
    author_email="prannesshkva@gmail.com",
    description="Gated Quasi-Unitary Lie-Algebra Recurrent State Space Models with Sparse Mixture-of-Experts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prannesshkva/QU-SSM",
    packages=find_packages(),
    py_modules=["modeling_qu_ssm", "configuration_qu_ssm", "qu_ssm_cuda_ext"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.22.0",
    ],
)

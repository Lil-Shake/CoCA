from setuptools import setup, find_packages

setup(
    name="coca",
    version="0.0.1",
    packages=["coca"],
    python_requires=">=3.10",
    install_requires=[
        "ml-collections",
        "absl-py",
        "diffusers[torch]==0.17.1",
        "accelerate==0.17",
        "wandb",
        "torchvision",
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "transformers==4.30.2",
    ],
)

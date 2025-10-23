# setup.py
from setuptools import setup, find_packages

setup(
    name="llm-client",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.1",
        "openai>=1.51.0",
        "groq>=0.5.0",
        "ollama>=0.1.9"
    ],
)

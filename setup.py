from setuptools import setup, find_packages

setup(
    name="nust-bank-assistant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "transformers",
        "torch",
        "sentence-transformers",
        "faiss-cpu",
        "python-docx",
        "PyPDF2",
        "openpyxl",
        "peft",
        "trl",
        "wandb"
    ],
    python_requires=">=3.8",
) 
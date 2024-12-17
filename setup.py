from setuptools import setup, find_packages

setup(
    name="earnings_ai_demo",
    version="0.1",
    packages=find_packages("earnings_ai_demo"),
    package_dir={"": "earnings_ai_demo"},
    install_requires=[
        "pymongo",
        "fireworks-ai", 
        "python-docx",
        "PyMuPDF"
    ]
)
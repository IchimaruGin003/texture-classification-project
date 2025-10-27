from setuptools import setup, find_packages

# 读取requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="texture-classification",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
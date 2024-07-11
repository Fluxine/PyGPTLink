from setuptools import setup, find_packages

setup(
    name='PyGPTLink',
    version='0.1.0',
    packages=find_packages(),
    description='A Python framework for easily integrating with OpenAI\'s LLMs.',
    author='Your Name',
    author_email='your.email@example.com',
    install_requires=[
        "openai",
        "jsonlines",
        "tiktoken"
    ],
)

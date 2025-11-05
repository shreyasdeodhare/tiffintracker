from setuptools import setup, find_packages

setup(
    name="tiffin-tracker",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'Flask',
        'pandas',
        'gunicorn',
        'Werkzeug'
    ],
)

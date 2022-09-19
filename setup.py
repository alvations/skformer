from setuptools import setup, find_packages

setup(
  name = 'skformer',
  packages = find_packages(),
  version = '0.0.1',
  description = 'skformer',
  long_description = '',
  author = '',
  url = 'https://github.com/alvations/skformer',
  keywords = [],
  install_requires = ['transformers', 'numpy', 'scipy', 'scikit-learn', 'xgboost'],
  classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ]
)

"""Setup autoreject."""
import os
from setuptools import setup, find_packages

# get the version (don't import autoreject here to avoid dependency)
version = None
with open(os.path.join('autoreject', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

descr = """Automated rejection and repair of epochs in M/EEG."""

DISTNAME = 'autoreject'
DESCRIPTION = descr
MAINTAINER = 'Mainak Jas'
MAINTAINER_EMAIL = 'mainakjas@gmail.com'
LICENSE = 'BSD-3-Clause'
URL = 'http://autoreject.github.io/'
DOWNLOAD_URL = 'https://github.com/autoreject/autoreject.git'
VERSION = version

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          long_description_content_type='text/x-rst',
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3',
          ],
          platforms='any',
          keywords=('electroencephalography eeg magnetoencephalography '
                    'meg preprocessing analysis'),
          python_requires='~=3.7',
          install_requires=[
              'numpy >= 1.8',
              'scipy >= 0.16',
              'mne >= 0.14',
              'scikit-learn >= 0.18',
              'joblib',
              'matplotlib >= 1.3',
          ],
          extras_require={
              'full': [
                  'tqdm',
                  'h5py'
              ],
              'test': [
                  'pytest',
                  'pytest-cov',
                  'pytest-sugar',
                  'check-manifest',
                  'flake8',
                  'pooch'
              ],
              'doc': [
                  'sphinx',
                  'sphinx-gallery',
                  'sphinx_bootstrap_theme',
                  'sphinx-copybutton',
                  'numpydoc',
                  'cython',
                  'pillow',
                  'openneuro-py >= 2021.7',
                  'pooch'
              ]
          },
          packages=find_packages(),
          project_urls={
              'Documentation': 'http://autoreject.github.io/',
              'Bug Reports': 'https://github.com/autoreject/autoreject/issues',
              'Source': 'https://github.com/autoreject/autoreject'
          }
          )

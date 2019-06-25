#! /usr/bin/env python
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
LICENSE = 'BSD (3-clause)'
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
          ],
          platforms='any',
          keywords=('electroencephalography eeg magnetoencephalography '
                    'meg preprocessing analysis'),
          packages=find_packages(),
          project_urls={'Documentation': 'http://autoreject.github.io/',
                        'Bug Reports': 'https://github.com/autoreject/autoreject/issues',  # noqa: E501
                        'Source': 'https://github.com/autoreject/autoreject'
                        }
          )

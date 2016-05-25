.. autoreject documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

autoreject
==========

This is a library to automatically reject bad trials and repair bad sensors in M/EEG.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``autoreject``, you first need to install its dependencies::

	$ pip install matplotlib scipy scikit-learn mne joblib pandas

Then clone the repository::

	$ git clone http://github.com/autoreject/autoreject

and finally run `setup.py` to install the package::

	$ cd autoreject
	$ python setup.py install

Cite
====

[1] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, "`Automated rejection and repair of bad trials in MEG/EEG <https://hal.archives-ouvertes.fr/hal-01313458/document>`_."
In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI). 2016.
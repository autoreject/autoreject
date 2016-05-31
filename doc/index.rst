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

	$ pip install numpy matplotlib scipy scikit-learn mne joblib pandas

Then clone the repository::

	$ git clone http://github.com/autoreject/autoreject

and finally run `setup.py` to install the package::

	$ cd autoreject/
	$ python setup.py install

If you do not have admin privileges on the computer, use the ``--user`` flag
with `setup.py`.

To check if everything worked fine, you can do::

	$ python -c 'import autoreject'

and it should not give any error messages.

Quickstart
==========

The easiest way to get started is to copy the following three lines of code
in your script:

.. code:: python

	>>> from autoreject import LocalAutoRejectCV
	>>> ar = LocalAutoRejectCV()
	>>> epochs_clean = ar.fit_transform(epochs)

This will automatically clean an `epochs` object read in using MNE-Python. For more details check
out the :ref:`example to automatically detect and repair bad epochs <sphx_glr_auto_examples_plot_auto_repair.py>`.

Cite
====

[1] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, "`Automated rejection and repair of bad trials in MEG/EEG <https://hal.archives-ouvertes.fr/hal-01313458/document>`_."
In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI). 2016.
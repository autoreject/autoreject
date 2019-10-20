.. autoreject documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

autoreject
==========

This is a library to automatically reject bad trials and repair bad sensors in magneto-/electroencephalography (M/EEG) data.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_
and a **Python version >=3.5** To install ``autoreject``, you first need to
install its dependencies::

	$ conda install numpy matplotlib scipy scikit-learn joblib
	$ pip install -U mne

An optional dependency is `tqdm <https://tqdm.github.io/>`_ if you want to use the verbosity flags `'tqdm'` or `'tqdm_notebook'`
for nice progressbars. In case you want to be able to read and write `autoreject` objects using the HDF5 format,
you may also want to install `h5py <https://pypi.org/project/h5py/>`_.

Then install the latest release of autoreject use::

	$ pip install -U autoreject

If you want to install the latest version of the code (nightly) use::

	$ pip install https://api.github.com/repos/autoreject/autoreject/zipball/master

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import autoreject'

and it should not give any error messages.

Quickstart
==========

The easiest way to get started is to copy the following three lines of code
in your script:

.. code:: python

	>>> from autoreject import AutoReject
	>>> ar = AutoReject()
	>>> epochs_clean = ar.fit_transform(epochs)  # doctest: +SKIP

This will automatically clean an `epochs` object read in using MNE-Python. To get the
rejection dictionary, simply do:

.. code:: python

	>>> from autoreject import get_rejection_threshold
	>>> reject = get_rejection_threshold(epochs)  # doctest: +SKIP

We also implement RANSAC from the PREP pipeline.
The API is the same:

.. code:: python

	>>> from autoreject import Ransac
	>>> rsc = Ransac()
	>>> epochs_clean = rsc.fit_transform(epochs)  # doctest: +SKIP

For more details check out the :ref:`example to automatically detect and repair bad epochs <sphx_glr_auto_examples_plot_auto_repair.py>`.

Bug reports
===========

Use the `github issue tracker <https://github.com/autoreject/autoreject/issues>`_ to report bugs.

Cite
====

[1] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, "`Automated rejection and repair of bad trials in MEG/EEG <https://hal.archives-ouvertes.fr/hal-01313458/document>`_."
In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI), 2016.

[2] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017.
"`Autoreject: Automated artifact rejection for MEG and EEG data <http://www.sciencedirect.com/science/article/pii/S1053811917305013>`_".
NeuroImage, 159, 417-429.

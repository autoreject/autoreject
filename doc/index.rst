.. autoreject documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

autoreject
==========

This is a library to automatically reject bad trials and repair bad sensors in magneto-/electroencephalography (M/EEG) data.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``autoreject``, you first need to install its dependencies::

	$ pip install numpy matplotlib scipy mne scikit-learn scikit-optimize

An optional dependency is `tqdm <https://tqdm.github.io/>`_ if you want to use the verbosity flags `'tqdm'` or `'tqdm_notebook'` 
for nice progressbars.

Then install autoreject::

	$ pip install git+https://github.com/autoreject/autoreject.git#egg=autoreject

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

	>>> from autoreject import LocalAutoRejectCV
	>>> ar = LocalAutoRejectCV()
	>>> epochs_clean = ar.fit_transform(epochs)  # doctest: +SKIP

This will automatically clean an `epochs` object read in using MNE-Python. We also implement RANSAC from the PREP pipeline.
The API is the same:

.. code:: python

	>>> from autoreject import Ransac
	>>> rsc = Ransac()
	>>> epochs_clean = rsc.fit_transform(epochs)  # doctest: +SKIP

For more details check out the :ref:`example to automatically detect and repair bad epochs <sphx_glr_auto_examples_plot_auto_repair.py>`.

.. note::

	Fow now, we do not guarantee if autoreject will work for more than one channel type. We intend to support multiple channel
	types, but in the future. Contributions to make this happen are welcome.

Bug reports
===========

Use the `github issue tracker <https://github.com/autoreject/autoreject/issues>`_ to report bugs.

Cite
====

[1] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, "`Automated rejection and repair of bad trials in MEG/EEG <https://hal.archives-ouvertes.fr/hal-01313458/document>`_."
In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI), 2016.

[2] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort, "`Autoreject: Automated artifact rejection for MEG and EEG <https://arxiv.org/abs/1612.08194>`_."
arXiv preprint arXiv:1612.08194, 2016.
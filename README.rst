autoreject
==========

|CircleCI|_ |GitHub Actions|_ |Codecov|_ |PyPI|_ |Conda-Forge|_

.. |CircleCI| image:: https://circleci.com/gh/autoreject/autoreject/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/autoreject/autoreject

.. |GitHub Actions| image:: https://github.com/autoreject/autoreject/actions/workflows/test.yml/badge.svg
.. _GitHub Actions: https://github.com/autoreject/autoreject/actions/workflows/test.yml

.. |Codecov| image:: http://codecov.io/github/autoreject/autoreject/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/autoreject/autoreject?branch=master

.. |PyPI| image:: https://badge.fury.io/py/autoreject.svg
.. _PyPI: https://badge.fury.io/py/autoreject

.. |Conda-Forge| image:: https://img.shields.io/conda/vn/conda-forge/autoreject.svg
.. _Conda-Forge: https://anaconda.org/conda-forge/autoreject/

This is a library to automatically reject bad trials and repair bad sensors in magneto-/electroencephalography (M/EEG) data.

.. image:: https://autoreject.github.io/stable/_images/sphx_glr_plot_auto_repair_001.png
   :width: 400


The documentation can be found under the following links:

- for the `stable release <https://autoreject.github.io/stable/index.html>`_
- for the `latest (development) version <https://autoreject.github.io/dev/index.html>`_

.. docs_readme_include_label

Installation
------------

We recommend the `Anaconda Python distribution <https://www.anaconda.com/>`_
and a **Python version >= 3.7**.
To obtain the stable release of ``autoreject``, you can use ``pip``::

    pip install -U autoreject

Or ``conda``::

    conda install -c conda-forge autoreject

If you want the latest (development) version of ``autoreject``, use::

    pip install https://api.github.com/repos/autoreject/autoreject/zipball/master

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`.

To check if everything worked fine, you can do::

    python -c 'import autoreject'

and it should not give any error messages.

Below, we list the dependencies for ``autoreject``.
All required dependencies are installed automatically when you install ``autoreject``.

* ``mne`` (>=0.24)
* ``numpy`` (>=1.20)
* ``scipy`` (>=1.6)
* ``scikit-learn`` (>=0.24)
* ``joblib``
* ``matplotlib`` (>=3.3)

Optional dependencies are:

* ``tqdm`` (for nice progress-bars when setting ``verbose=True``)
* ``h5io`` (for writing ``autoreject`` objects using the HDF5 format)
* ``openneuro-py`` (>= 2021.7, for fetching data from OpenNeuro.org)

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

We also implement RANSAC from the `PREP pipeline <https://doi.org/10.3389/fninf.2015.00016>`_.
The API is the same:

.. code:: python

	>>> from autoreject import Ransac
	>>> rsc = Ransac()
	>>> epochs_clean = rsc.fit_transform(epochs)  # doctest: +SKIP

For more details check out the example to
`automatically detect and repair bad epochs <https://autoreject.github.io/stable/_images/sphx_glr_plot_auto_repair_001.png>`_.

Bug reports
===========

Please use the `GitHub issue tracker <https://github.com/autoreject/autoreject/issues>`_ to report bugs.

Cite
====

[1] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, "`Automated rejection and repair of bad trials in MEG/EEG <https://hal.archives-ouvertes.fr/hal-01313458/document>`_."
In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI), 2016.

[2] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017.
"`Autoreject: Automated artifact rejection for MEG and EEG data <http://www.sciencedirect.com/science/article/pii/S1053811917305013>`_".
NeuroImage, 159, 417-429.

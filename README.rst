autoreject
==========

|CircleCI|_ |GitHub Actions|_ |Codecov|_ |PyPI|_

.. |CircleCI| image:: https://circleci.com/gh/autoreject/autoreject/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/autoreject/autoreject

.. |GitHub Actions| image:: https://github.com/autoreject/autoreject/actions/workflows/test.yml/badge.svg
.. _GitHub Actions: https://github.com/autoreject/autoreject/actions/workflows/test.yml

.. |Codecov| image:: http://codecov.io/github/autoreject/autoreject/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/autoreject/autoreject?branch=master

.. |PyPI| image:: https://badge.fury.io/py/autoreject.svg
.. _PyPI: https://badge.fury.io/py/autoreject

This repository hosts code to automatically reject trials and repair sensors for M/EEG data.

.. image:: http://autoreject.github.io/_images/sphx_glr_plot_visualize_bad_epochs_002.png


The documentation can be found under the following links:

- for the `stable release <https://autoreject.github.io/>`_
- for the `latest (development) version <https://circleci.com/api/v1.1/project/github/autoreject/autoreject/latest/artifacts/0/html/index.html?branch=master>`_

Dependencies
------------

These are the dependencies to use ``autoreject``:

* ``Python`` (>=3.7)
* ``mne`` (>=0.14)
* ``numpy`` (>=1.8)
* ``scipy`` (>=0.16)
* ``scikit-learn`` (>=0.18)
* ``joblib``
* ``matplotlib`` (>=1.3)

Optional dependencies are:

* ``tqdm`` (for nice progressbars)
* ``h5py`` (for IO)
* ``openneuro-py`` (for fetching data from OpenNeuro.org)

Cite
----

If you use this code in your project, please cite::

    Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort,
    "Automated rejection and repair of bad trials in MEG/EEG." In 6th International Workshop on
    Pattern Recognition in Neuroimaging (PRNI). 2016.

    Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017.
    Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.

Auto Reject
===========

|CircleCI|_ |Travis|_ |Codecov|_

.. |CircleCI| image:: https://circleci.com/gh/autoreject/autoreject/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/autoreject/autoreject

.. |Travis| image:: https://api.travis-ci.org/autoreject/autoreject.svg?branch=master
.. _Travis: https://travis-ci.org/autoreject/autoreject

.. |Codecov| image:: http://codecov.io/github/autoreject/autoreject/coverage.svg?branch=master
.. _Codecov: http://codecov.io/github/autoreject/autoreject?branch=master

This repository hosts code to automatically reject trials and repair sensors for M/EEG data.

.. image:: http://autoreject.github.io/_images/sphx_glr_plot_visualize_bad_epochs_002.png


The documentation can be found under the following links:

- for the `stable release <https://autoreject.github.io/>`_
- for the `latest (development) version <https://circleci.com/api/v1.1/project/github/autoreject/autoreject/latest/artifacts/0/html/index.html?branch=master>`_

Dependencies
------------

These are the dependencies to use autoreject:

* Python (>=3.5)
* numpy (>=1.8)
* matplotlib (>=1.3)
* scipy (>=0.16)
* mne-python (>=0.14)
* scikit-learn (>=0.18)
* joblib

Two optional dependencies are `tqdm` (for nice progressbars) and `h5py` (for IO).

Cite
----

If you use this code in your project, please cite::

    Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort,
    "Automated rejection and repair of bad trials in MEG/EEG." In 6th International Workshop on
    Pattern Recognition in Neuroimaging (PRNI). 2016.

    Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017.
    Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.

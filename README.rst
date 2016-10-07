Auto Reject
===========

|CircleCI|_

.. |CircleCI| image:: https://circleci.com/gh/autoreject/autoreject/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/autoreject/autoreject

This repository hosts code to automatically reject trials and repair sensors for M/EEG data.

Dependencies
------------

We are actively trying to reduce the number of dependencies. However, as of now these are the dependencies for the examples
to run:

* numpy (>=1.8)
* matplotlib (>=1.3)
* scipy (>=0.16)
* mne-python
* scikit-learn (0.18)
* scikit-optimize (0.1)
* joblibs
* pandas

Cite
----

If you use this code in your project, please cite::

	Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort,
	"Automated rejection and repair of bad trials in MEG/EEG." In 6th International Workshop on
	Pattern Recognition in Neuroimaging (PRNI). 2016.
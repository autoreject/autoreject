:orphan:

.. _faq:

Frequently asked questions
==========================

This section of the documentation provides a *discussion-like* format, answering
"How-to" questions.

Should I apply ICA first or autoreject first?
---------------------------------------------

Please read :ref:`plot_autoreject_workflow`.

Is it dangerous to do source estimation with interpolated channels?
-------------------------------------------------------------------

Interpolated data is no different from measured data. It's what you would expect
to see if there were no artifacts in the data. Interpolation is nothing magical.
It simply takes a weighted average of the data in the neighboring good channels. Indeed, if the artifact was left in the data, it will bias the source estimate
far more than any potential harm that interpolation may pose.

How do I manually set the `n_interpolate` and `consensus` parameter?
--------------------------------------------------------------------------

If you do not want autoreject to select a parameter for you, simply pass it
as a list of a single element::

	>>> ar = AutoReject(n_interpolate=[1], consensus_percs=[0.6])

Note this will still run a cross-validation loop to generate the
validation score.

Is it possible to get only bad sensor annotations and not interpolate?
----------------------------------------------------------------------

Yes! Simply do::

	>>> ar.fit(epochs)
	>>> reject_log = ar.get_reject_log(epochs)

No need to run `ar.transform(epochs)` in this case.

Frequently asked questions
==========================

Should I apply ICA first or autoreject first?
---------------------------------------------

ICA solutions can be affected by high amplitude artifacts, therefore
we recommend first using autoreject to detect the bad segments, then applying
ICA, and finally interpolating the bad data.

To ignore bad segments using autoreject (local), we could do::

	>>> ar = AutoReject()
	>>> _, reject_log = ar.fit(epochs).transform(epochs, return_log=True)
	>>> ica.fit(epochs[~reject_log.bad_epochs])

or use autoreject (global)::

	>>> reject = get_rejection_threshold(epochs)
	>>> ica.fit(epochs, reject=reject)

This option can be more preferred if we would like to fit ICA on the raw
data, and not on the epochs. After this, we can apply ICA::

	>>> ica.exclude = [5, 7]  # exclude EOG components
	>>> ica.transform(epochs)

Finally, autoreject could be applied to clean the data::

	>>> ar = AutoReject()
	>>> epochs_clean = ar.fit_transform(epochs)

Autoreject is not meant for eyeblink artifacts since it affects neighboring
sensors. Indeed, a spatial filtering method like ICA is better suited for this.

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

Frequently asked questions
==========================

Should I apply ICA first or autoreject first?
---------------------------------------------

ICA solutions can be affected by high amplitude artifacts, therefore
we recommend first using autoreject to detect the bad segments, then applying
ICA, and finally interpolating the bad data.

To ignore bad segments using autoreject (local), we could do::

	>>> ar = Autoreject()
	>>> _, reject_log = ar.fit(epochs).transform(epochs)
	>>> ica.fit(epochs[~reject_log.bad_epochs_idx])

or use autoreject (global)::

	>>> reject = get_rejection_threshold(epochs)
	>>> ica.fit(epochs, reject=reject)

Then, we can apply ICA::

	>>> ica.exclude = [5, 7]  # exclude EOG components
	>>> ica.transform(epochs)

Finally, autoreject could be applied to clean the data::

	>>> ar = Autoreject()
	>>> epochs_clean = ar.fit(epochs).transform(epochs)

Autoreject is not meant for eyeblink artifacts since it affects neighboring
sensors. Indeed, a spatial filtering method like ICA is better suited for this.

How do I manually set the `n_interpolate` and `consensus` parameter?
--------------------------------------------------------------------------

If you do not want autoreject to select a parameter for you, simply pass it
as a list of a single element::

	>>> ar = Autoreject(n_interpolate=[1], consensus_percs=[0.6])

Note this will still run a cross-validation loop to generate the 
validation score.

Is it possible to get only bad sensor annotations and not interpolate?
----------------------------------------------------------------------

Yes! Simply do::

	>>> ar.fit(epochs)
	>>> ar.get_reject_log(epochs)

No need to run `ar.transform(epochs)` in this case.

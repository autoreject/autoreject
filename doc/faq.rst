:orphan:

Frequently asked questions
==========================

This section of the documentation provides a *discussion-like* format, answering
"How-to" questions.

Should I apply ICA first or autoreject first?
---------------------------------------------

ICA solutions can be affected by high amplitude artifacts, therefore
we recommend to determine a reasonable rejection threshold on which data
segments to ignore in the ICA. autoreject (global) can be used exactly for this
purpose::

	>>> reject = get_rejection_threshold(epochs)
	>>> ica.fit(epochs, reject=reject)

In case you want to fit your ICA on the raw data, you will need an intermediate
step, because autoreject only works on epoched data. ICA is ignoring the time
domain of the data, so we can simply turn the raw data into equally spaced
"fixed length" epochs using ::func::`mne.make_fixed_length_events`::

	>>> epo_duration = 1.0
	>>> events = mne.make_fixed_length_events(raw, duration=epo_duration)
	>>> epochs = mne.Epochs(raw, events, tmin=0.0, tmax=epo_duration)
	>>> reject = get_rejection_threshold(epochs)
	>>> ica.fit(epochs, reject=reject)

After the estimation step and all other processing that happened on the
components, the ICA results can be applied to the raw data::

	>>> ica.exclude = [5, 7]  # exclude EOG components
	>>> clean_raw = ica.apply(raw)

After obtaining the ICA cleaned raw data, you may consider making your own
specific epochs, and applying autoreject (local).

As an alternative to using autoreject (global) before the ICA, and autorejct
(local) as a second step later on, you can use autoreject (local) directly
on your epochs to detect the bad segments, then applying ICA, and finally
interpolating the bad data.

To ignore bad segments using autoreject (local), we could do::

	>>> ar = AutoReject()
	>>> _, reject_log = ar.fit(epochs).transform(epochs, return_log=True)
	>>> ica.fit(epochs[~reject_log.bad_epochs])

As a final note, consider that autoreject is not meant to "clean" eyeblink
artifacts since it affects neighboring sensors. Indeed, a spatial filtering
method like ICA is better suited for this.

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

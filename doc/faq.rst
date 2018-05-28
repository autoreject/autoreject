Frequently asked questions
==========================

Should I apply ICA first or autoreject first?
---------------------------------------------

ICA solutions can be affected by high amplitude artifacts, therefore
we recommend first using autoreject to detect the bad segments, then applying
ICA, and finally interpolating the bad data::

	>>> ar = LocalAutorejectCV()
	>>> ar.fit(epochs)
	>>> ica.fit(epochs[~ar.reject_log.bad_epochs_idx])
	>>> ica.exclude = [5, 7]  # exclude EOG components
	>>> ica.transform(epochs)
	>>> ar.transform(epochs)

Autoreject is not meant for eyeblink artifacts since it affects neighboring
sensors. Indeed, a spatial filtering method like ICA is better suited for this.

How do I manually set the `n_interpolate` and `consensus_percs` parameter?
--------------------------------------------------------------------------

If you do not want autoreject to select a parameter for you, simply pass it
as a list of a single element::

	>>> ar = LocalAutorejectCV(n_interpolate=[1], consensus_percs=[0.6])

Note this will still run a cross-validation loop to generate the 
validation score.

What if I do not know the channel locations?
--------------------------------------------

While autoreject will still work, the solution may not be optimal. The channel
locations are needed for generating augmented trials which is a necessary
ingredient of the algorithm. If you are working with EEG data, you can use MNE
to set a standard montage::

     # code for setting montage

Does autoreject also interpolate user-marked bad channels?
----------------------------------------------------------

No, autoreject ignores the bad channels in `epochs.info['bads']` by default.
However, it is possible to explicitly ask autoreject to work on all the channels
(including those marked as bad by the user) by using the `picks` argument.

Is it possible to get only bad sensor annotations and not interpolate?
----------------------------------------------------------------------

Yes! Simply do::

	>>> ar.fit(epochs)
	>>> ar.get_reject_log()

No need to run `ar.transform()` in this case.

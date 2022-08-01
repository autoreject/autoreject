:orphan:

.. _whats_new:

What's new?
===========

.. currentmodule:: autoreject

.. _0.4:

0.4 (unreleased)
----------------

Changelog
~~~~~~~~~

- ``compute_thresholds`` now works without channel location data with option
  ``augment=False`` by `Nikolai Chapochnikov`_ in :github:`#282`

- ``get_rejection_threshold`` now also accepts ECoG and SEEG data by `Nikolai Chapochnikov`_ in :github:`#281`

- RANSAC implementation was refactored, works now with `n_jobs>1` and produces consistent results across different number of jobs. Testing on simulated data added. by `Simon Kern`_ in :github:`#280`

- ``autoreject`` now requires ``mne >= 1.0``, by `Mainak Jas`_, `Alex Gramfort`_, and `Stefan Appelhoff`_ in :github:`#267` and :github:`#268`

- Add ``reject_log`` option to :meth:`autoreject.AutoReject.transform` to enable
  users to make corrections to ``reject_log`` estimated by autoreject, by `Alex Rockhill`
  in :github:`#270` 

- Add :meth:`autoreject.RejectLog.save` and :func:`autoreject.read_reject_log` to
  save and load reject logs, by `Alex Rockhill`_ in :github:`#270`

.. _0.3:

0.3 (2022-01-04)
----------------

Changelog
~~~~~~~~~

- Make plotting of reject log more flexible by allowing to change the orientation of plot,
  option to supress the immediate display of the plot and returning the plot figure for further
  modification or saving in :meth:`autoreject.RejectLog.plot`, by `Marcin Koculak`_ in :github:`#152`
- Add `show_names` option to :func:`autoreject.RejectLog.plot,` by `Mainak Jas`_ in :github:`#209`
- Enable support for fNIRS data types, by `Robert Luke`_ in :github:`#177`
- Add additional type support for argument `picks`, by `Mathieu Scheltienne`_ in :github:`#225`
- Use MNE progressbar, by `Patrick Stetz`_ in :github:`#227`
- :func:`autoreject.RejectLog.plot` now accepts an ``ax`` parameter to pass an existing Axes object, by `Stefan Appelhoff`_ in :github:`#301`

Bug
~~~

- Don't reset `epochs.info['bads']` within :func:`autoreject.compute_thresholds`, by `Mainak Jas`_ in :github:`#203`
- Adjust usage of a private MNE method used for interpolation that failed for newer MNE versions, by `Adina Wagner`_ in :github:`#212`.

API
~~~

.. _0.2:

0.2 (2019-06-24)
----------------

Changelog
~~~~~~~~~

- Introduced a new method :meth:`autoreject.AutoReject.save` and function :func:`autoreject.read_auto_reject`
  for IO of autoreject objects, by `Mainak Jas`_ in :github:`#120`
- Make MEG interpolation faster by precomputing dot products for the interpolation, by `Mainak Jas`_
  in :github:`#122`
- Add default option for `param_range` in :func:`autoreject.validation_curve`, by `Alex Gramfort`_
  in :github:`#129`

Bug
~~~

- Fixed bug in picking bad channels during interpolation. This bug only affects users who got an assertion
  error when running :class:`autoreject.Autoreject`, by `Mainak Jas`_ in :github:`#115`
- Added check for channel locations so that autoreject does not
  hang when the channel positions are nan, by `Mainak Jas`_ in :github:`#130`
- Fixed bug in random seed for the Ransac algorithm when n_jobs > 1, by `Legrand Nico`_ and `Mainak Jas`_ in :github:`#138`
- Fixed pickling of :class:`autoreject.AutoReject`, by `Hubert Banville`_ in :github:`#193`

API
~~~

- Added `ch_types` argument to :func:`autoreject.get_rejection_threshold` to find
  rejection thresholds for only subset of channel types in the data, by `Mainak Jas`_ in :github:`#140`

0.1 (2018-06-11)
----------------

Changelog
~~~~~~~~~

- Initial release

.. _Mainak Jas: https://perso.telecom-paristech.fr/mjas/
.. _Legrand Nico: https://legrandnico.github.io/
.. _Alex Gramfort: http://alexandre.gramfort.net
.. _Marcin Koculak: https://mkoculak.github.io/
.. _Hubert Banville: https://hubertjb.github.io/
.. _Adina Wagner: https://www.adina-wagner.com/
.. _Robert Luke: https://github.com/rob-luke/
.. _Mathieu Scheltienne: https://github.com/mscheltienne
.. _Patrick Stetz: https://patrickstetz.com
.. _Stefan Appelhoff: https://stefanappelhoff.com/
.. _Alex Rockhill: https://github.com/alexrockhill
.. _Nikolai Chapochnikov: https://github.com/chapochn
.. _Simon Kern: https://github.com/skjerns

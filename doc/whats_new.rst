:orphan:

.. _whats_new:

What's new?
===========

.. currentmodule:: autoreject

.. _current:

Current
-------

Changelog
~~~~~~~~~

- Make plotting of reject log more flexible by allowing to change the orientation of plot, option to supress the immediate display of the plot and returning the plot figure for further modification or saving in :meth:`autoreject.RejectLog.plot`, by `Marcin Koculak`_ in :github:`#152`

- Add `show_names` option to :func:`autoreject.RejectLog.plot` by `Mainak Jas`_ in :github:`#322`

- Enable support for fNIRS data types by `Robert Luke`_ in :github:`#177`

- Add additional type support for argument `picks` by `Mathieu Scheltienne`_ in :github:`#225`

- Use MNE progressbar by `Patrick Stetz`_ in :github:`#227`

Bug
~~~

- Don't reset `epochs.info['bads']` within :func:`autoreject.compute_thresholds` by `Mainak Jas`_ in :github:`#203`

- Adjust usage of a private MNE method used for interpolation that failed for newer MNE versions, by `Adina Wagner`_ in :github:`#212`.

API
~~~

.. _0.2:

0.2
---

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
  error when running :class:`autoreject.Autoreject`. Fixed by `Mainak Jas`_ in :github:`#115`
- Added check for channel locations so that autoreject does not
  hang when the channel positions are nan. Fixed by `Mainak Jas`_ in :github:`#130`
- Fixed bug in random seed for the Ransac algorithm when n_jobs > 1, by `Legrand Nico`_ and `Mainak Jas`_ in :github:`#138`
- Fixed pikcling of :class:`autoreject.AutoReject`. Fixed by `Hubert Banville`_ in :github:`#193`


API
~~~

- Added `ch_types` argument to :func:`autoreject.get_rejection_threshold` to find
  rejection thresholds for only subset of channel types in the data, by `Mainak Jas`_ in :github:`#140`

.. _Mainak Jas: https://perso.telecom-paristech.fr/mjas/
.. _Legrand Nico: https://legrandnico.github.io/
.. _Alex Gramfort: http://alexandre.gramfort.net
.. _Marcin Koculak: https://mkoculak.github.io/
.. _Hubert Banville: https://hubertjb.github.io/
.. _Adina Wagner: https://www.adina-wagner.com/
.. _Robert Luke: https://github.com/rob-luke/
.. _Mathieu Scheltienne: https://github.com/mscheltienne
.. _Patrick Stetz: https://patrickstetz.com

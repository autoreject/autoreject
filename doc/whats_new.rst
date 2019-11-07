:orphan:

.. _whats_new:

0.2
---

Changelog
~~~~~~~~~

- Introduced a new method :meth:`autoreject.AutoReject.save` and function :func:`autoreject.read_auto_reject`
  for IO of autoreject objects, by `Mainak Jas`_ in `#120 <https://github.com/autoreject/autoreject/pull/120>`_

- Make MEG interpolation faster by precomputing dot products for the interpolation, by `Mainak Jas`_
  in `#122 <https://github.com/autoreject/autoreject/pull/122>`_

- Add default option for `param_range` in :func:`autoreject.validation_curve`, by `Alex Gramfort`_
  in `#129 <https://github.com/autoreject/autoreject/pull/129>`_

Bug
~~~

- Fixed bug in picking bad channels during interpolation. This bug only affects users who got an assertion
  error when running :class:`autoreject.Autoreject`. Fixed by `Mainak Jas`_ in `#115 <https://github.com/autoreject/autoreject/pull/115>`_
- Added check for channel locations so that autoreject does not
  hang when the channel positions are nan. Fixed by `Mainak Jas`_ in `#130 <https://github.com/autoreject/autoreject/pull/130>`_
- Fixed bug in random seed for the Ransac algorithm when n_jobs > 1, by `Legrand Nico`_ and `Mainak Jas`_ in `#138 <https://github.com/autoreject/autoreject/pull/138>`_

API
~~~

- Added `ch_types` argument to :func:`autoreject.get_rejection_threshold` to find
  rejection thresholds for only subset of channel types in the data, by `Mainak Jas`_ in `#140 <https://github.com/autoreject/autoreject/pull/140>`_

.. _Mainak Jas: https://perso.telecom-paristech.fr/mjas/
.. _Legrand Nico: https://legrandnico.github.io/
.. _Alex Gramfort: http://alexandre.gramfort.net
.. _Marcin Koculak: https://mkoculak.github.io/

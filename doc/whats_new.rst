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

- Introduced a new method :meth:`autoreject.AutoReject.save` and function :func:`autoreject.read_auto_reject`
  for IO of autoreject objects, by `Mainak Jas`_ in `#120 <https://github.com/autoreject/autoreject/pull/120>`_

Bug
~~~

- Fixed bug in picking bad channels during interpolation. This bug only affects users who got an assertion
  error when running :class:`autoreject.Autoreject`. Fixed by `Mainak Jas`_ in `#115 <https://github.com/autoreject/autoreject/pull/115>`_
- Added check for channel locations so that autoreject does not
  hang when the channel positions are nan. Fixed by `Mainak Jas`_ in `#130 <https://github.com/autoreject/autoreject/pull/130>`_

API
~~~

.. _Mainak Jas: https://perso.telecom-paristech.fr/mjas/

:orphan:

.. _explanation:

Explanation
===========

This section of the documentation exists to illuminate how ``autoreject`` works.
The primary source for understanding should be the original publication [1]_,
however the sections in this guide can make the content of that primary source
more graspable.


Intuition on how the - *autoreject global* - algorithm works
------------------------------------------------------------

- Given some MEEG data :math:`X` with the dimensions
  :math:`trials(=epochs) \times sensors \times timepoints`

- We want to find a threshold :math:`\tau` in :math:`\mu V` that will reject
  noisy epochs and retain clean epochs

- Do the following for a set of possible candidate thresholds: :math:`\Phi`

- For each :math:`\tau_i \in \Phi` :

  - Split your data :math:`X` into :math:`K` folds (:math:`K` equal parts)
    along the trial dimension

    - Each of the :math:`K` parts will be a "test" set once, while the
      remaining :math:`K-1` parts will be combined to be the corresponding
      "train" set (see `k-fold crossvalidation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_)

  - Then for each fold :math:`K` (consisting of train and test trials) do:

    - apply threshold :math:`\tau_i` to reject trials in the train set

    - calculate the mean of the signal (for each sensor and timepoint) over
      the GOOD (=not rejected) trials in the train set

    - calculate the *median* of the signal (for each sensor and timepoint)
      over ALL trials in the test set

    - compare both of these signals and calculate the error :math:`e_k`
      (i.e., take the `Frobenius norm <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`_
      of their difference)

    - save that error :math:`e_k`

  - Now we have :math:`K` errors :math:`e_k  \in E`

  - Form the mean error :math:`\bar E` (over all :math:`K` errors) associated
    with our current threshold :math:`\tau_i` in :math:`\mu V`

  - Save the mapping of :math:`\tau_i` to its associated error :math:`\bar E`

- ... now each threshold candidate in the set :math:`\Phi` is mapped to a
  specific error value :math:`\bar E`

- the candidate threshold :math:`\tau_i` with the lowest error is the best
  rejection threshold for a global rejection

References
----------
.. [1] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and
   Alexandre Gramfort. 2017. Autoreject: Automated artifact rejection for MEG
   and EEG data. NeuroImage, 159, 417-429.

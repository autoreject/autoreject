Explanations
============

This section of the documentation exists to illuminate how ``autoreject`` works.
The primary source for understanding should be the original publication [1]_,
however the sections in this guide can make the content of that primary source
more graspable.


Intuition on how the - *autoreject global* - algorithm works
------------------------------------------------------------
- Given some data with multiple trials (epochs) that contain sensor X timepoints data
- We want to find a threshold T in microvolts that will reject noisy epochs and retain clean epochs
- Do the following for a set of possible candidate thresholds Ts
- For each T:
    - Split your data into K folds (=k equal parts) along the trial dimension
        - Each of the k parts will be a "test" set once, while the others will be combined to be a "train" set
    - Then for each fold k (consisting of train and test trials) do:
        - apply threshold T to reject trials in the train set
        - calculate the mean of the signal (for each sensor and timepoint) over the GOOD trials in the train set
        - calculate the *median* of the signal (for each sensor and timepoint) over ALL trials in the test set
        - compare both of these signals and calculate the error (frobenius norm of their difference)
        - save the error
    - Now we have k errors
    - Form the mean error (over all k errors) associated with our current threshold T in microvolts
    - Save the mapping of current T to its error

- ... now we have a set of Ts all mapped to their specific error
- the T with the lowest error is the best rejection threshold for a global rejection

References
----------
.. [1] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and
   Alexandre Gramfort. 2017. Autoreject: Automated artifact rejection for MEG
   and EEG data. NeuroImage, 159, 417-429.

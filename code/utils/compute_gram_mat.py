import pdb

import numpy as np
from neu_ml.code.utils.compute_kernel_func_vals \
    import compute_kernel_func_vals

def compute_gram_mat(data, kernel):
    """Compute the Gram matrix corresponding to input data using the specified
    kernel

    Parameters
    ----------
    data : array, shape ( n_instances, dimension )
        The input data over which to compute the Gram matrix

    kernel : list
        A list with two elements. The first element is a string indicating the
        type of kernel. The second element is an array indicating the
        parameter values of the kernel

    Returns
    -------
    gram_mat : array, shape ( n_instances, n_instances )
        The Gram matrix

    See also
    --------
    Refer to 'compute_kernel_func_vals.py' for supported kernel types
    """

    n_instances = data.shape[0]
    gram_mat = np.zeros((n_instances, n_instances), dtype=float)

    for i in np.arange(0, n_instances):
        for j in np.arange(i, n_instances):
            # Gram matrices are symmetric
            gram_mat[i,j] = gram_mat[j,i] = \
                compute_kernel_func_vals(np.atleast_2d(data[i,:]),
                                         np.atleast_2d(data[j,:]), kernel)

    return gram_mat

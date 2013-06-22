import pdb

import numpy as np

def compute_kernel_func_vals(data1, data2, kernel):
    """Computes the kernel function value between the two input vectors using
    the specified kernel.

    Parameters
    ----------
    data1 : array, shape ( n_samples, dimension )
        'n_samples' is the number of data points and 'dimension' is the
        dimension.

    data2 : array, shape ( n_samples, dimension )
        'n_samples' is the number of data points and 'dimension' is the
        dimension

    kernel : list
        A list with two elements. The first element is a string indicating the
        name of the kernel (see the 'Notes' below for currently supported
        kernels). The second element is an array of length 'params', where
        'params' are the scalar parameter values controlling the behavior of
        the kernel

    Returns
    -------
    vals : array, shape ( n_samples, 1 )
        The scalar kernel function values

    Notes
    -----
    Currently supported kernels:
    1) Gaussian kernel: kernel[0] = 'gaussian'. The equation of the kernel is
    $\theta_{1}^{2}\exp\{-\frac{\theta_{2}}{2}||x_{n}-x_{m}||^{2}\}$. The
    parameters for this kernel are: kernel[1] = array([$\theta_{1}$,
    $\theta_{2}$])
    2) Constant kernel: kernel[0] = 'constant'. All kernel values are set to
    the value specified by kernel[1]
    """
    if data1.shape[0] != data2.shape[0]:
        raise ValueError("Number of samples must be same")

    n_samples = data1.shape[0]
    params = kernel[1]
    vals = np.zeros((n_samples,1))

    # Gaussian kernel case
    if kernel[0] == 'gaussian':
        diff = data1 - data2
        tmp  = (diff*diff).sum(axis = 1)
        vals = (params[0]**2)*np.exp(-params[1]*tmp/2.0)
    # Constant kernel case
    elif kernel[0] == 'constant':
        vals = params[0]
    else:
        vals = None

    return vals

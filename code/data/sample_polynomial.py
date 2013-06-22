import numpy as np

def sample_polynomial(n_points=1, cos=None, domain_range=None, var=0,
                      domain_locs=None):
    """Randomly sample a (noisy) polynomial

    This function is used to generate random noisy samples from a polynomial.
    The user can specify the coefficients of the desired polynomial, the
    number of desired points, the range over which the samples are drawn, and
    the amount of noise added to the target values.
    
    Parameters
    ----------
    n_points : int
        Number of desired data points

    cos : array, shape ( 'degree' )
        The coefficients of the polynomial in descending powers. The length of
        the vector indicates the degree of the desired polynomial

    domain_range : array, shape ( 2 )
        This is a 2D vector indicating the minimum and maximum of
        the input range of the 'n_points'. The points will be randomly
        (uniformly) sampled within this range, and the target values will be
        computed at those input locations}

    var : float
        Each computed target value will corrupted with Gaussian noise with the
        specified variance

    domain_locs : array, shape ( n_instances ), optional
        A specific list of domain locations at which to evaluate the
        polynomial. If specified, the input variables 'n_points' and
        'domain_range' will be ignored.

    Returns
    -------
    data : array, shape ( 2, 'n_points')
        A 'n_points'x2 array where the first column represents the the randomly
        sampled inputs and the second column corresponds to the noisy target
        values.
    """
    
    # First generate the domain location values. They should be distributed
    # randomly and uniformly across the range specified
    if domain_locs is None:
        domain_locs = (domain_range[1] - \
                       domain_range[0])*np.random.rand(n_points) + \
                       domain_range[0]
    
    # Now generate the noisy target values
    targets = np.polyval(cos, domain_locs) + np.sqrt(var)*np.random.randn(n_points)

    # Construct the data for output
    data = np.array([domain_locs, targets])

    return data.T

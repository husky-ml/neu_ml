import pdb

import numpy as np

def sample_helix(n_points, z_range, var=0.0, frequency=1.0, x_amplitude=1.0,
                 y_amplitude=1.0, x_phase=0.0, y_phase=0.0, z_locs=None):
    """Randomly generate samples from a helix function.

    This function is used to generate random noisy samples from a helix.
    The user can specify the parameters of the helix, the number of desired
    points, the range over which the samples are drawn, and
    the amount of noise added to the x- and y-locations.
    
    Parameters
    ----------
    n_points : int
        Number of desired data points

    z_range : array, shape ( 2 )
        This is a 2D vector indicating the minimum and maximum of
        the input range of the 'n_points'. The z-locations will be randomly
        (uniformly) sampled within this range, and the x- and y- locations
        will be computed at those input locations

    var : float        
        The x- and y-locations will be corrupted with Gaussian noise with the
        specified variance

    frequency : float, optional
        The frequency governing the sine and cosine terms in the helix
        equation. This term controls the "tightness" of the helix

    x_amplitude : float, optional
        The helix amplitude in the x-direction

    y_amplitude : float, optional
        The helix amplitude in the y-direction

    x_phase : float, optional
        The phase of the cosine in the x-direction

    y_phase : float, optional
        The phase of the cosine in the y-direction

    z_locs : array, shape ( n_instances ), optional
        A specific list of z-locations at which to evaluate the helix. If
        specified, the input variables 'n_points' and 'z_range' will be
        ignored.

    Returns
    -------
    data : array, shape ( n_points, 3 )
        The first column corresponds to the x-axis, the second the y-axis, and
        the third the z-axis
    """
    
    # First generate the z location values. They should be distributed
    # randomly and uniformly across the range specified
    if z_locs is None:
        z_locs = (z_range[1] - z_range[0])*np.random.rand(n_points) + z_range[0]

    x_locs = x_amplitude*np.cos(frequency*z_locs + x_phase) + \
        np.sqrt(var)*np.random.randn(n_points)
    y_locs = y_amplitude*np.sin(frequency*z_locs + y_phase) + \
        np.sqrt(var)*np.random.randn(n_points)

    data = np.array([x_locs, y_locs, z_locs])    

    return data.T

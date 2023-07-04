def calculate_parameters(mean, var, prev_p, prev_r):
    """
    Calculate the parameters p and r using the mean and variance.

    Args:
        mean (float): Mean of the distribution.
        var (float): Variance of the distribution.
        prev_p (float): Previous p value.
        prev_r (float): Previous r value.

    Returns:
        tuple: The calculated parameters p and r.
    """
    if mean is not None and var is not None and var > mean and var != 0:
        p = mean / var
        r = mean ** 2 / (var - mean)
    else:
        p, r = prev_p, prev_r

    return p, r



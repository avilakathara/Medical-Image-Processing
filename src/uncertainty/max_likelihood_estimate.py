import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def max_likelihood_estimate(data):
    # Define initial parameter values
    initial_params = [0, 1]

    # Minimize the negative log-likelihood function using the Nelder-Mead method
    result = minimize(neg_log_likelihood, initial_params, args=(data,), method='Nelder-Mead')

    # Get the estimated parameters
    mu_estimated, sigma_estimated = result.x

    return mu_estimated, sigma_estimated

# Define the negative log-likelihood function for a normal distribution
def neg_log_likelihood(params, data):
    mu, sigma = params
    ll = np.sum(norm.logpdf(data, loc=mu, scale=sigma))
    return -ll



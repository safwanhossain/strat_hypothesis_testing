import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

mu_b = 0.5

def unif_cdf(mu_val, left=0.4, right=0.7):
    return (min(mu_val, right) - left) / (right - left)

def unif_pdf(mu_0, cond=None, left=0.4, right=0.7):
    if mu_0 < left or mu_0 > right:
        return 0

    if cond is None:
        return 1 / (right - left) 
    
    if cond and cond[0] == "interval":
        # compute P[mu_0 | left_int < mu_0 < right_int]
        left_int, right_int = cond[1], cond[2]
        if not left_int < mu_0 < right_int:
            return 0
        return 1 / (mu_b - max(left_int, left))
    
    if cond and cond[0] == "max":
        # compute P[mu_0 | mu_0 >= max(val1, val2)]
        max_val = max(cond[1], cond[2])
        if max_val > right or mu_0 < max_val:
            return 0
        return 1 / (right - max_val)
    
def truc_gaussian_cdf(mu_val, left=0.4, right=0.7, sigma=0.04, mu=0.67):
    # Define the truncated normal distribution
    # a and b are the lower and upper bounds (in standard deviations)
    # mu is the mean, sigma is the standard deviation
    lower, upper = (left - mu) / sigma, (right - mu) / sigma  # Standardize bounds

    # Calculate the CDF for the truncated normal distribution
    cdf = truncnorm.cdf(mu_val, lower, upper, loc=mu, scale=sigma)
    return cdf

def truc_gaussian_pdf(mu_0, cond=None, left=0.4, right=0.7, sigma=0.04, mu=0.67):
    lower, upper = (left - mu) / sigma, (right - mu) / sigma  # Standardize bounds

    if mu_0 < left or mu_0 > right:
        return 0

    if cond is None:
        pdf = truncnorm.pdf(mu_0, lower, upper, loc=mu, scale=sigma)
        return pdf
    
    if cond and cond[0] == "interval":
        # compute P[mu_0 | left_int < mu_0 < right_int]
        left_int, right_int = cond[1], cond[2]
        if not left_int < mu_0 < right_int:
            return 0
        pdf = truncnorm.pdf(mu_0, lower, upper, loc=mu, scale=sigma)
        return pdf / (truc_gaussian_cdf(right_int) - truc_gaussian_cdf(left_int))
        
    if cond and cond[0] == "max":
        # compute P[mu_0 | mu_0 >= max(val1, val2)]
        max_val = max(cond[1], cond[2])
        if max_val > right or mu_0 <= max_val:
            return 0
        pdf = truncnorm.pdf(mu_0, lower, upper, loc=mu, scale=sigma)
        return pdf / (1 - truc_gaussian_cdf(max_val))
    
def plot_distribution(pdf_func, cdf_func, name):
    x_range = np.linspace(0.4, 0.7, 100)
    pdf_vals = []
    cdf_vals = []
    for i in x_range:
        pdf_vals.append(pdf_func(i))
        cdf_vals.append(cdf_func(i))
        
    plt.figure()  # Adjusted figure size for two plots
    plt.plot(x_range, pdf_vals, color='red', label=f'PDF')
    plt.axvline(x=0.5, color='black', linestyle='--', label=r'$\mu_b$')
    ax1 = plt.gca()  # Get current axis
    ax1.legend(loc="upper left")
    ax1.set_ylabel("PDF Values")
    ax2 = ax1.twinx()  # Create a twin axis
    ax2.plot(x_range, cdf_vals, color='green', label=f'CDF')
    ax2.set_ylabel('CDF Values')
    plt.title(f"{name}")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    plot_distribution(truc_gaussian_pdf, truc_gaussian_cdf, "Trunc Gaussian [0.4, 0.7]")

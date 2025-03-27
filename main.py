import numpy as np
import math
from scipy import stats

import scipy.integrate as integrate
import copy
import matplotlib.pyplot as plt  # Add this import at the top

# CONSTANTS
VERBOSE = False
R = 100
A_0, A = 5, 0.25
NMAX = 300

def get_utility(inst_dict, pass_prob, n):
    return pass_prob*inst_dict["r"] - (n*inst_dict["a"] + inst_dict["a_0"])

def get_p_value(inst_dict, n, verbose=False, exact=False):
    # compute the p value for a baseline effectiveness, alternate hypothesis (current)
    # and number of samples n.
    mu_b, mu_0 = inst_dict["mu_b"], inst_dict["mu_0"]

    # Explicit and direct calculation
    if exact:
        run_sum = 0
        for i in range(int(n*mu_0), n+1):
            run_sum += math.comb(n, i) * mu_b**i * (1-mu_b)**(n-i)
        exact_p_val = run_sum
        if verbose or VERBOSE:
            print(f"Exact p value: {exact_p_val}")
        return exact_p_val
    
    # Implicitly using scipy
    k = int(n * mu_0)  # number of successes observed
    p_value_binom = 1 - stats.binom.cdf(k - 1, n, mu_b)
    est_p_value = p_value_binom

    if verbose or VERBOSE:
        print(f"Approx p value: {est_p_value}")
    return est_p_value


def get_pass_probability(inst_dict, alpha, n, verbose=False, exact=False):
    # For a released p-value, compute the probability of passing when using n samples in the trial
    # This is the probability that the evidence be enough to reject the null hypothesis
    mu_b, mu_0 = inst_dict["mu_b"], inst_dict["mu_0"]
    n = int(n)

    # Exact calculations  
    if exact:
        # First find the critical region, the set of observations that pass the p value
        z_exact = n+1
        for z in range(1, n+1):
            run_sum = 0
            for i in range(z, n+1):
                run_sum += math.comb(n, i) * mu_b**i * (1-mu_b)**(n-i) 
            if run_sum <= alpha:
                z_exact = z
                break

        pass_prob_exact = 0
        for z in range(z_exact, n+1):
            pass_prob_exact += math.comb(n, z) * mu_0**z * (1-mu_0)**(n-z)
        
        if verbose or VERBOSE:
            print(f"The exact pass probability is: {pass_prob_exact}") 
        return pass_prob_exact

    # Normal approximation to the binomial calculation
    z_norm = n*mu_b + stats.norm.ppf(1-alpha, loc=0, scale=1)*np.sqrt(n*mu_b*(1-mu_b)) + 0.5
    cdf_arg = (z_norm - n*mu_0)/np.sqrt(n*mu_0*(1-mu_0))
    pass_prob_norm = 1 - stats.norm.cdf(cdf_arg, loc=0, scale=1)
    
    if verbose or VERBOSE:
        print(f"The approx pass probability is: {pass_prob_norm}")

    return pass_prob_norm

def get_best_response(inst_dict, alpha, verbose=False):
    """ Get the optimal number of samples and the whether the agent should participate or not in the trials.
    We are gonna stick to a quick and dirty linear search, although this can be done quickly in log(n) time
    """
    mu_b, mu_0 = inst_dict["mu_b"], inst_dict["mu_0"]

    ut_opt, n_opt = -1, 0
    for i in range(1, NMAX):
        pass_prob = get_pass_probability(inst_dict, alpha, i, verbose=False, exact=False)
        utility = get_utility(inst_dict, pass_prob, i)
        if utility >= 0 and utility > ut_opt:
            n_opt = i
            ut_opt = utility

    if ut_opt < 0:
        if verbose or VERBOSE:
            print(f"Agent with effect size {mu_0 - mu_b} will not particiapte")
        return False, 0
    else:
        if verbose or VERBOSE:
            print(f"Agent with effect size {mu_0 - mu_b} will particiapte at optimal n: {n_opt}")
        return True, n_opt


def get_threshold_belief(inst_dict, alpha, eps=0.005, verbose=False):
    """Get the threshold for a given baseline and p-value threshold
    """
    # discretize the belief space
    beliefs = np.linspace(0, 1, int(1/eps) + 1)

    # also just do a linear search for now
    left, right = 0, len(beliefs)
    while left <= right:
        mid = left + math.ceil((right - left) // 2)
        inst_dict["mu_0"] = beliefs[mid]
        participate, n = get_best_response(inst_dict, alpha)
        if left == right:
            break
        if not participate:
            left = mid + 1
        else:
            right = mid
   
    threshold = beliefs[mid]
    inst_dict["mu_0"] = beliefs[mid]
    participate, n_tau = get_best_response(inst_dict, alpha)
    pass_prob = get_pass_probability(inst_dict, alpha, n_tau, verbose=False)
    utility = get_utility(inst_dict, pass_prob, n_tau)
    if verbose or VERBOSE:
        print(f"The threshold belief is: {threshold}")
        print(f"The utility at threshold belief is: {utility}") 
    return threshold, n_tau

def get_alpha_star(inst_dict):
    """ Compute the value of alpha such that mu_tau(alpha) = mu_b. Since mu_tau decreases as 
        alpha increases, we can use binary search to find this value.
    """
    # discretize the p value space
    alphas = np.linspace(0.001, 0.25, 200)

    # also just do a linear search for now
    left, right = 0, len(alphas)
    mu_b = inst_dict["mu_b"]
    while left <= right:
        mid = left + math.ceil((right - left) // 2)
        alpha_val = alphas[mid]
        mu_tau, n_tau = get_threshold_belief(inst_dict, alpha_val)
        if left == right:
            break
        if mu_tau > mu_b: 
            left = mid + 1
        else:
            right = mid
   
    alpha_star = alphas[mid]
    return alpha_star

def get_loss(inst_dict, alpha):
    """ Plot the total loss as a function of the p-value alpha
    """
    # We shall consider the belief distribution to uniform between 
    # mu_b - 0.1 and mu_b + 0.2
    mu_b = inst_dict["mu_b"] 
    left, right = 0.4, 0.7

    def unif_cdf(mu_val):
        return (min(mu_val, right) - left) / (right - left)

    def unif_pdf(mu_0, cond=None):
        if mu_0 <= left or mu_0 >= right:
            return 0

        if cond is None:
            return 1 / (right - left) 
        
        if cond and cond[0] == "interval":
            # compute P[mu_0 | left_int < mu_0 < right_int]
            left_int, right_int = cond[1], cond[2]
            if not left_int <= mu_0 <= right_int:
                return 0
            return 1 / (mu_b - max(left_int, left))
        
        if cond and cond[0] == "max":
            # compute P[mu_0 | mu_0 >= max(val1, val2)]
            max_val = max(cond[1], cond[2])
            if max_val >= right or mu_0 <= max_val:
                return 0
            return 1 / (right - max_val)  

    def conditional_prob(mu_0, mu_tau, cond, arg):
        inst_dict_copy = copy.deepcopy(inst_dict)
        inst_dict_copy["mu_0"] = mu_0
        if unif_pdf(mu_0) == 0:
            return 0
       
        if cond == "interval": 
            val = unif_pdf(mu_0, ("interval", mu_tau, mu_b))
        if cond == "max":
            val = unif_pdf(mu_0, ("max", mu_tau, mu_b)) 
        
        participate, n_opt = get_best_response(inst_dict_copy, alpha)
        if not participate:
            return 0

        pass_prob = get_pass_probability(inst_dict_copy, alpha, n_opt)
        if arg == "pass":
            return pass_prob*val
        else:
            return (1-pass_prob)*val
    
    # First get the threshold belief for this
    mu_tau, n_tau = get_threshold_belief(inst_dict, alpha, eps=0.005)
    
    # compute the fp conditional on participation
    if mu_tau >= mu_b:
        fp_part = 0
        total_fp = 0
        approx_fp_part = 0
    else:
        fp_part, err = integrate.quad(
            conditional_prob, 
            left, 
            right, 
            args=(mu_tau, "interval", "pass"),
            epsabs=1e-8, epsrel=1e-8
        )
        total_fp = fp_part*(unif_cdf(mu_b) - unif_cdf(mu_tau))/unif_cdf(mu_b) 
        approx_fp_part = alpha*(unif_cdf(mu_b) - unif_cdf(mu_tau))/unif_cdf(mu_b)
    
    # compute the fn conditional on participation
    if mu_tau >= right:
        fn_part = 0
        total_fn = 0
        approx_fn_part = 0
    else:
        fn_part, err = integrate.quad(
            conditional_prob,
            left, 
            right, 
            args=(mu_tau, "max", "fail"),
            epsabs=1e-8, epsrel=1e-8
        )
        approx_fn_part, err = integrate.quad(
            conditional_prob, 
            left, 
            right, 
            args=(mu_tau, "max", "pass"),
            epsabs=1e-8, epsrel=1e-8
        )  
        total_fn = fn_part*(1-unif_cdf(max(mu_tau, mu_b)))/(1 - unif_cdf(mu_b))
        approx_fn_part *= (1-unif_cdf(max(mu_tau, mu_b)))/(1 - unif_cdf(mu_b)) 
     
    # compute the worthy non-participation probabilities
    if mu_tau <= mu_b:
        worthy_part_loss = 0
    else:
        worthy_part_loss = (unif_cdf(mu_tau) - unif_cdf(mu_b)) / (1 - unif_cdf(mu_b))
    
    loss = total_fp + total_fn + worthy_part_loss
    ret_dict = {
        "approx_loss" : 1 + approx_fp_part - approx_fn_part,
        "loss" : loss,
        "fp_part" : fp_part,
        "total_fp" : total_fp,
        "fn_part" : fn_part,
        "total_fn" : total_fn,
        "worthy_part_loss" : worthy_part_loss,
    }
    print(f"Computed loss for alpha: {alpha}")
    return ret_dict

if __name__ == "__main__":
    inst_dict = {
        "mu_b" : 0.5,
        "mu_0" : 0.6,
        "r" : 100,
        "a_0" : 5,
        "a" : 0.25
    }
    
    alpha = 0.05
    #get_p_value(inst_dict, 100, verbose=True)
    #get_pass_probability(inst_dict, 0.05, 50, verbose=True)
    #get_best_response(inst_dict, alpha, verbose=True)
    #get_threshold_belief(inst_dict, 0.05, eps=0.005, verbose=True)
    get_loss(inst_dict, alpha)



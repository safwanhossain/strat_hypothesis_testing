import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json  # Import the json module
from multiprocess import Pool

from main import (
    get_utility,
    get_best_response,
    get_pass_probability,
    get_threshold_belief,
    get_loss,
    get_alpha_star
)

def plot_exact_vs_estimate_pass_prob():
    inst_dict = {
        "mu_b" : 0.5,
        "mu_0" : 0.6,
        "r" : 100,
        "a_0" : 5,
        "a" : 0.25
    }
    
    # Vary alpha for fixed n
    alpha_range = np.linspace(0.01, 0.1, 21)
    exact_pass_prob_alpha, est_pass_prob_alpha = [], []
    for alpha in alpha_range:
        est = get_pass_probability(inst_dict, alpha, 100, verbose=False, exact=False)
        exact = get_pass_probability(inst_dict, alpha, 100, verbose=False, exact=True)
        est_pass_prob_alpha.append(est)
        exact_pass_prob_alpha.append(exact)

    # Vary n for fixed alpha
    n_range = np.linspace(10, 200, 21)
    exact_pass_prob_n, est_pass_prob_n = [], []
    for n in n_range:
        est = get_pass_probability(inst_dict, 0.05, n, verbose=False, exact=False)
        exact = get_pass_probability(inst_dict, 0.05, n, verbose=False, exact=True)
        est_pass_prob_n.append(est)
        exact_pass_prob_n.append(exact)

     # Vary effect size for fixed alpha, n
    mu_range = np.linspace(0.5, 0.8, 21)
    exact_pass_prob_mu, est_pass_prob_mu = [], []
    for mu in mu_range:
        inst_dict["mu_0"] = mu
        est = get_pass_probability(inst_dict, 0.05, 100, verbose=False, exact=False)
        exact = get_pass_probability(inst_dict, 0.05, 100, verbose=False, exact=True)
        est_pass_prob_mu.append(est)
        exact_pass_prob_mu.append(exact)

    # Plotting
    plt.figure(figsize=(18, 6))  # Adjusted figure size for three plots

    # Plot for varying alpha
    plt.subplot(1, 3, 1)
    plt.plot(alpha_range, est_pass_prob_alpha, color='red', label='Estimated', marker='o')
    plt.plot(alpha_range, exact_pass_prob_alpha, color='blue', label='Exact', marker='x')
    plt.title('Pass Probability vs Alpha (n=100, mu_0=0.06)')
    plt.xlabel('Alpha')
    plt.ylabel('Pass Probability')
    plt.legend()

    # Plot for varying n
    plt.subplot(1, 3, 2)
    plt.plot(n_range, est_pass_prob_n, color='red', label='Estimated', marker='o')
    plt.plot(n_range, exact_pass_prob_n, color='blue', label='Exact', marker='x')
    plt.title('Pass Probability vs n (Alpha=0.05, mu_0=0.06)')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Pass Probability')
    plt.legend()

    # Plot for varying effect size
    plt.subplot(1, 3, 3)
    plt.plot(mu_range, est_pass_prob_mu, color='red', label='Estimated', marker='o')
    plt.plot(mu_range, exact_pass_prob_mu, color='blue', label='Exact', marker='x')
    plt.title('Pass Probability vs Effect Size (Alpha=0.05, n=100)')
    plt.xlabel('Effectiveness (mu_0)')
    plt.ylabel('Pass Probability')
    plt.legend()

    plt.tight_layout()
    plt.show()  # Display the plots

def plot_utility():
    inst_dict = {
        "mu_b" : 0.5,
        "mu_0" : 0.6,
        "r" : 100,
        "a_0" : 5,
        "a" : 0.25
    }

    # Vary n for fixed alpha
    n_range = np.linspace(10, 300, 21)
    utility_n = []
    for n in n_range:
        pass_prob = get_pass_probability(inst_dict, 0.05, n, verbose=False, exact=False)
        ut = get_utility(inst_dict, pass_prob, n)
        utility_n.append(ut)

     # Plot for varying n
    plt.plot(n_range, utility_n, color='red', label='Utility', marker='o')
    plt.title('Utility vs n (Alpha=0.05, mu_0=0.06)')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Utility')
    plt.legend() 
    plt.show()


def plot_best_response():
    inst_dict = {
        "mu_b" : 0.5,
        "mu_0" : 0.6,
        "r" : 100,
        "a_0" : 5,
        "a" : 0.25
    }

    # Vary alpha for fixed effect
    alpha_range = np.linspace(0.01, 0.1, 21)
    n_opts_alpha = []
    ut_opts_alpha = []
    pass_prob_opts_alpha = []
    for alpha in alpha_range:
        _, n_opt = get_best_response(inst_dict, alpha, verbose=False) 
        pass_prob = get_pass_probability(inst_dict, alpha, n_opt, verbose=False, exact=False)
        ut = get_utility(inst_dict, pass_prob, n_opt)
        pass_prob_opts_alpha.append(pass_prob)
        ut_opts_alpha.append(ut) 
        n_opts_alpha.append(n_opt)

    # Vary effect for fixed alpha
    mu_range = np.linspace(0.5, 0.8, 21)
    n_opts_mu = []
    ut_opts_mu = []
    pass_prob_opts_mu = []
    for mu in mu_range:
        inst_dict["mu_0"] = mu
        _, n_opt = get_best_response(inst_dict, 0.05, verbose=False)
        pass_prob = get_pass_probability(inst_dict, 0.05, n_opt, verbose=False, exact=False)
        pass_prob_opts_mu.append(pass_prob)
        ut = get_utility(inst_dict, pass_prob, n_opt)
        ut_opts_mu.append(ut)  
        n_opts_mu.append(n_opt)

    # Plotting
    plt.figure(figsize=(12, 6))  # Adjusted figure size for two plots

    # Plot for varying alpha
    plt.subplot(1, 2, 1)
    plt.plot(alpha_range, pass_prob_opts_alpha, color='red', label='Pass Probability', marker='o')
    plt.title('Pass Probability vs Alpha (mu_0 = 0.6)')
    plt.xlabel('Alpha')
    plt.ylabel('Optimal Sample Size (n)')
    plt.legend(loc='upper left')

    # Add a secondary y-axis for utility
    ax1 = plt.gca()  # Get current axis
    ax2 = ax1.twinx()  # Create a twin axis
    ax2.plot(alpha_range, ut_opts_alpha, color='green', label='Utility', marker='x')
    #ax2.plot(alpha_range, ut_opts_alpha_200, alpha=0.4, color='green', label='Utility', marker='x')
    ax2.set_ylabel('Utility')
    ax2.legend(loc='upper right')

    # Plot for varying effect size
    plt.subplot(1, 2, 2)
    plt.plot(mu_range, n_opts_mu, color='red', label='Optimal n', marker='o')
    plt.title('Optimal Sample Size vs Effect Size (Alpha=0.05)')
    plt.xlabel('Effectiveness mu_0 (Baseline is 0.5)')
    plt.ylabel('Optimal Sample Size (n)')
    plt.legend(loc='upper left')

    # Add a secondary y-axis for utility
    ax3 = plt.gca()  # Get current axis
    ax4 = ax3.twinx()  # Create a twin axis
    ax4.plot(mu_range, ut_opts_mu, color='green', label='Utility', marker='x')
    ax4.set_ylabel('Utility')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_threshold():
    inst_dict = {
        "mu_b" : 0.5,
        "mu_0" : 0.6,
        "r" : 100,
        "a_0" : 5,
        "a" : 0.25
    }

    # Vary alpha for fixed effect
    alpha_range = np.linspace(0.005, 0.1, 21)
    r_range = [80, 100, 120]
    n_taus = np.zeros((3, len(alpha_range)))
    mu_taus = np.zeros((3, len(alpha_range)))
    for i, r in enumerate(r_range):
        for j, alpha in enumerate(alpha_range):
            inst_dict['r'] = r
            threshold, n_tau = get_threshold_belief(inst_dict, alpha, eps=0.0005) 
            mu_taus[i,j] = threshold
            n_taus[i,j] = n_tau

    # Plotting
    plt.figure(figsize=(12, 6))  # Adjusted figure size for two plots

    # Plot for varying alpha
    plt.subplot(1, 2, 1)
    plt.plot(alpha_range, mu_taus[0], color='red', label='Thres Belief (R=80)', marker='o')
    plt.plot(alpha_range, mu_taus[1], color='red', alpha=0.6, label='Thres Belief (R=100)', marker='o')
    plt.plot(alpha_range, mu_taus[2], color='red', alpha= 0.3, label='Thres Belief (R=120)', marker='o')
    plt.title('Threshold Belief vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Threshold Belief (mu_tau)')
    plt.legend(loc='upper right')

    # Plot for varying effect size
    plt.subplot(1, 2, 2)
    plt.plot(alpha_range, n_taus[0], color='blue', label='n_tau (R=80)', marker='o')
    plt.plot(alpha_range, n_taus[1], color='blue', alpha=0.6, label='n_tau (R=100)', marker='o')
    plt.plot(alpha_range, n_taus[2], color='blue', alpha = 0.3, label='n_tau (R=120)', marker='o')
    plt.title('Optimal Sample Size vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Sample Size at threshold (n_tau)')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_loss():
    inst_dict = {
        "mu_b" : 0.5,
        "mu_0" : 0.6,
        "r" : 100,
        "a_0" : 5,
        "a" : 0.25
    }

    # Vary alpha for fixed effect
    alpha_range = np.linspace(0.001, 0.2, 50)
    alpha_star = get_alpha_star(inst_dict)

    fp_loss, fn_loss, worth_loss, cum_loss, approx_loss = [], [], [], [], []
    total_fp, total_fn = [], []
    
    args = [(inst_dict, alpha) for alpha in alpha_range]
    with Pool(processes=6) as pool:
        results = pool.starmap_async(get_loss, args)
        results = results.get()

    for result in results:
        loss_dict = result
        fp_loss.append(loss_dict["fp_part"])
        fn_loss.append(loss_dict["fn_part"])
        total_fp.append(loss_dict["total_fp"])
        total_fn.append(loss_dict["total_fn"])
        worth_loss.append(loss_dict["worthy_part_loss"])
        cum_loss.append(loss_dict["loss"])
        approx_loss.append(loss_dict["approx_loss"]) 

    # If you don't want to use multiprocessing
    # for alpha in tqdm(alpha_range):
    #     loss_dict = get_loss(inst_dict, alpha)
    #     fp_loss.append(loss_dict["fp_part"])
    #     fn_loss.append(loss_dict["fn_part"])
    #     total_fp.append(loss_dict["total_fp"])
    #     total_fn.append(loss_dict["total_fn"])
    #     worth_loss.append(loss_dict["worthy_part_loss"])
    #     cum_loss.append(loss_dict["loss"])
    #     approx_loss.append(loss_dict["approx_loss"])

    # Save results to JSON
    results_dict = {
        "fp_loss": fp_loss,
        "fn_loss": fn_loss,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "worth_loss": worth_loss,
        "cum_loss": cum_loss,
        "approx_loss": approx_loss
    }
    
    with open('loss_results.json', 'w') as json_file:  # Specify the filename
        json.dump(results_dict, json_file)  # Write the dictionary to the JSON file
    
    # Plotting
    plt.figure(figsize=(12, 6))  # Adjusted figure size for two plots

    # First plot for fp_loss, fn_loss, and worth_loss
    plt.subplot(1, 2, 1)
    plt.plot(alpha_range, total_fp, color='blue', label='FP Loss')
    plt.plot(alpha_range, total_fn, color='orange', label='FN Loss')
    plt.plot(alpha_range, worth_loss, color='green', label='Worth Loss')
    plt.axvline(x=alpha_star, color='black', linestyle='--', label='alpha*')
    plt.title('Loss Components vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Loss')
    plt.legend()

    # Second plot for components of the false negative loss
    plt.subplot(1, 2, 2)
    plt.plot(alpha_range, fn_loss, color='blue', label='Exact FN Loss')
    plt.plot(alpha_range, np.array(total_fn)/np.array(fn_loss), color='green', label='P[mu_0 > mu_tau|mu_0 > mu_b]')
    plt.plot(alpha_range, total_fn, color='orange', label='FN Loss * Prob')
    plt.axvline(x=alpha_star, color='black', linestyle='--', label='alpha*')
    plt.title('FN Loss Components vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()  # Display the plots

    # Second plot for cum_loss
    plt.figure()  # Adjusted figure size for two plots
    plt.plot(alpha_range, cum_loss, color='red', label='Cumulative Loss')
    plt.plot(alpha_range, approx_loss, color='red', alpha=0.4, label='Cumulative Loss (Upper Bound)')
    plt.axvline(x=alpha_star, color='black', linestyle='--', label='alpha*')
    plt.title('Cumulative Loss vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Cumulative Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()  # Display the plots
   
    
if __name__ == "__main__":
    plot_loss()
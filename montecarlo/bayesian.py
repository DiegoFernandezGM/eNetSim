import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from network import Network
from sampletechs import Sampler
import emcee

class BayesianAnalysis:
    def __init__(self, config, network):
        self.config = config
        self.network = network
        self.energy_consumption_results = self.network.calculate_energy_consumption()
        self.sample_techs = Sampler(self.config)
        self.prior_distribution = self.define_prior_distribution()
        self.subsystems = self.identify_subsystems()

    def identify_subsystems(self):
        subsystems = []
        for subsystem, subsystem_config in self.config.items():
            if subsystem not in ["electricity_emission_factor", "iterations"]:
                subsystems.append(subsystem)
        return subsystems
    
    def define_prior_distribution(self):
        # Define the prior distribution for the parameters
        # Here, we assume a uniform distribution as the prior
        prior_distribution = stats.uniform(loc=0, scale=1)
        return prior_distribution

    def calculate_true_energy(self, subsystem_name):
        # Use the calculate_energy_consumption function to get the energy consumption estimate
        if subsystem_name in self.energy_consumption_results:
            true_energy = self.energy_consumption_results[subsystem_name]
        else:
            raise ValueError(f"Subsystem '{subsystem_name}' not found in energy consumption results")
        return true_energy

    def calculate_energy_uncertainty(self, subsystem_name):
        # Calculate the energy uncertainty based on the results
        energy_uncertainty = np.std(self.energy_consumption_results[subsystem_name])
        # Add a small jitter value to avoid dividing by zero
        energy_uncertainty = np.maximum(energy_uncertainty, 1e-6)
        return energy_uncertainty

    def calculate_likelihood(self, theta, subsystem_name):
        # Calculate the likelihood function
        # Here, we assume a Gaussian distribution as the likelihood
        true_energy = self.calculate_true_energy(subsystem_name)
        energy_uncertainty = self.calculate_energy_uncertainty(subsystem_name)
        likelihood = stats.norm(loc=true_energy, scale=energy_uncertainty)
        log_likelihood = np.log(likelihood.pdf(theta))
        print(log_likelihood)
        return log_likelihood

    def calculate_posterior(self, theta, subsystem_name):
        # Calculate the posterior distribution
        log_prior = np.log(self.prior_distribution.pdf(theta))
        log_likelihood = self.calculate_likelihood(theta, subsystem_name)
        log_posterior = log_prior + log_likelihood
        return log_posterior

    def sample_posterior(self, subsystem_name, num_samples):
        # Sample the posterior distribution using MCMC
        #slice sampling method
        #theta = np.random.uniform(0,1)
        #posterior_samples = []
        #for _ in range(num_samples):
        #    log_posterior = np.log(self.calculate_posterior(theta, subsystem_name))
        #    theta = self.slice_sampling(theta, log_posterior, subsystem_name)
        #    posterior_samples.append(theta)
        #return np.array(posterior_samples)
        # Sample the posterior distribution using MCMC
        ndim = 1
        nwalkers = 50
        pos = np.random.uniform(0, 1, size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.calculate_posterior, args=[subsystem_name])
        state = sampler.run_mcmc(pos, num_samples, progress=True)
        posterior_samples = sampler.flatchain
        return posterior_samples
    
    def slice_sampling(self, theta, log_posterior, subsystem_name):
        # Implement the slice sampling algorithm
        # You can replace this with the appropriate slice sampling implementation for your problem
        u = np.random.uniform()
        lower = theta - np.abs(theta * u)
        upper = theta + np.abs(theta * u)
        while True:
            theta_proposal = np.random.uniform(lower, upper)
            log_posterior_proposal = np.log(self.calculate_posterior(theta_proposal, subsystem_name))
            if np.log(np.random.uniform()) < log_posterior_proposal - log_posterior:
                theta = theta_proposal
                break
        return theta

    def perform_bayesian_analysis(self):
        # Perform the Bayesian analysis for all subsystems
        posterior_samples = {}
        for subsystem_name in self.subsystems:
            posterior_samples[subsystem_name] = self.sample_posterior(subsystem_name, 100)
        return posterior_samples

    def plot_posterior_samples(self, posterior_samples):
        # Plot histograms of posterior samples
        for subsystem_name, samples in posterior_samples.items():
            plt.hist(samples, bins=50, alpha=0.5, label=subsystem_name)
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.title('Posterior Samples')
        plt.legend()
        plt.show()

    def plot_posterior_distributions(self, posterior_samples):
        # Plot kernel density estimates of posterior distributions
        for subsystem_name, samples in posterior_samples.items():
            kde = stats.gaussian_kde(samples.T)
            x = np.linspace(samples.min(), samples.max(), 100)
            plt.plot(x, kde(x), label=subsystem_name)
        plt.xlabel('Parameter Value')
        plt.ylabel('Density')
        plt.title('Posterior Distributions')
        plt.legend()
        plt.show()

    def plot_uncertainty_intervals(self, posterior_samples):
        # Plot uncertainty intervals
        for subsystem_name, samples in posterior_samples.items():
            quantiles = np.quantile(samples, [0.025, 0.975])
            plt.plot([subsystem_name, subsystem_name], quantiles, 'k-')
        plt.xlabel('Subsystem')
        plt.ylabel('Parameter Value')
        plt.title('Uncertainty Intervals')
        plt.show()

    def analyze_uncertainty(self, posterior_samples):
        # Analyze uncertainty in model parameters and predictions
        for subsystem_name, samples in posterior_samples.items():
            print(f"Subsystem: {subsystem_name}")
            print(f"Mean: {np.mean(samples)}")
            print(f"Standard Deviation: {np.std(samples)}")
            print(f"95% Credible Interval: {np.quantile(samples, [0.025, 0.975])}")
            print()
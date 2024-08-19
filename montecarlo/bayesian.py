from importer import *

from network import Network
from sampletechs import Sampler

class BayesianAnalysis:
    def __init__(self, config, network):
        self.config = config
        self.network = network
        self.energy_consumption_results = self.network.calculate_energy_consumption()
        self.sample_techs = Sampler(self.config)
        self.prior_distribution = self.define_prior_distribution()
        self.subsystems = self.identify_subsystems()

    def identify_subsystems(self):

        ##  Identifies all subsystems within the configuration subject to analysis,
        ##  excluding data within certain predefined keys (emission factor and iteration nº)

        subsystems = []
        for subsystem, subsystem_config in self.config.items():
            if subsystem not in ["electricity_emission_factor", "iterations"]:
                subsystems.append(subsystem)
        
        return subsystems
    
    def define_prior_distribution(self):

        ##  Define the prior distribution for the parameters
        ##  Here, we assume a uniform distribution as the non-informative prior
        
        prior_distribution = stats.uniform(loc=0, scale=1)
        
        return prior_distribution

    def calculate_true_energy(self, subsystem_name):

        ##  Retrieves the true energy consumption estimate for a subsystem from network
        
        if subsystem_name in self.energy_consumption_results:
            true_energy = self.energy_consumption_results[subsystem_name]
        else:
            raise ValueError(f"Subsystem '{subsystem_name}' not found in energy consumption results")
        
        return true_energy

    def calculate_energy_uncertainty(self, subsystem_name):

        ##  Estimates uncertainty by calculating the stddev of a given subsystem energy consumption 
        ##  Adds a small jitter to avoid division by zero

        energy_uncertainty = np.std(self.energy_consumption_results[subsystem_name])
        energy_uncertainty = np.maximum(energy_uncertainty, 1e-6)
        
        return energy_uncertainty

    def calculate_likelihood(self, theta, subsystem_name):
        
        ##  Computes the likelihood of the observed data given the parameters (theta) using a Gaussian distribution
        ##  Returns the log-likelihood

        true_energy = self.calculate_true_energy(subsystem_name)
        energy_uncertainty = self.calculate_energy_uncertainty(subsystem_name)
        likelihood = stats.norm(loc=true_energy, scale=energy_uncertainty)
        log_likelihood = np.log(likelihood.pdf(theta))
        
        return log_likelihood

    def calculate_posterior(self, theta, subsystem_name):
        
        ##  Calculates the posterior probability distribution using Bayes' theorem, combining the prior and likelihood

        log_prior = np.log(self.prior_distribution.pdf(theta))
        log_likelihood = self.calculate_likelihood(theta, subsystem_name)
        log_posterior = log_prior + log_likelihood
        
        return log_posterior

    def sample_posterior(self, subsystem_name, num_samples):
        
        ##  Samples from the posterior distribution using the MCMC method
        
        ndim = 1
        nwalkers = 50
        pos = np.random.uniform(0, 1, size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.calculate_posterior, args=[subsystem_name])
        state = sampler.run_mcmc(pos, num_samples, progress=True)
        posterior_samples = sampler.flatchain
        
        return posterior_samples
    
    def slice_sampling(self, theta, log_posterior, subsystem_name):
    
        ##  Implements a slice sampling algorithm to generate samples from the posterior distribution
        ##  This method can be used as alternative to MCMC sampling
        
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
        
        ##  Executes Bayesian analysis for all identified subsystems
        ##  Generates posterior samples for each subsystem
        
        posterior_samples = {}
        for subsystem_name in self.subsystems:
            posterior_samples[subsystem_name] = self.sample_posterior(subsystem_name, 100)
        
        return posterior_samples

    def plot_posterior_samples(self, posterior_samples):
        
        ##  Plots histograms of posterior samples for each subsystem
        
        plt.figure(figsize=(12, 6))

        color_palette = sns.color_palette("muted", len(posterior_samples))

        for idx, (subsystem_name, samples) in enumerate(posterior_samples.items()):
            plt.hist(samples, bins=50, alpha=0.7, edgecolor='black', color=color_palette[idx], label=subsystem_name)

        # Set chart labels, title, legend and grid
        plt.xlabel('Parameter Value', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        plt.title('Posterior Samples', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left') # Position outside the plot
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

        # Style spines
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    def plot_posterior_distributions(self, posterior_samples):
        
        ##  Plots each subsystem's posterior distributions using Kernel Density Estimation
        ##  Provides smooth estimate of the distribution
        
        plt.figure(figsize=(12, 6))
        
        for subsystem_name, samples in posterior_samples.items():
            kde = stats.gaussian_kde(samples.T)
            x = np.linspace(samples.min(), samples.max(), 100)
            plt.plot(x, kde(x), label=subsystem_name, linewidth=2)
        
        # Set chart labels, title, legend and grid
        plt.xlabel('Parameter Value', fontsize=14, fontweight='bold')
        plt.ylabel('Density', fontsize=14, fontweight='bold')
        plt.title('Posterior Distributions', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, frameon=False)
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        
        # Style spines
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        plt.show()

    def plot_uncertainty_intervals(self, posterior_samples):
        
        ##  Plots credibility intervals for each subsystem
        ##  Represents range within which true parameter value are expected to lie (95% probability)

        plt.figure(figsize=(12, 6))
        
        for subsystem_name, samples in posterior_samples.items():
            quantiles = np.quantile(samples, [0.025, 0.975])
            plt.plot([subsystem_name, subsystem_name], quantiles, 'o-', markersize=8, linewidth=2)
        
        # Set chart labels, title, legend and grid
        plt.xlabel('Subsystem', fontsize=14, fontweight='bold')
        plt.ylabel('Parameter Value', fontsize=14, fontweight='bold')
        plt.title('Uncertainty Intervals', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        
        # Style spines
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        plt.show()

    def analyse_uncertainty(self, posterior_samples):

        ##  Model parameters and predictions UA for each unit by calculating key statistics (mean, stddev and credible intervals)

       for subsystem_name, samples in posterior_samples.items():
            print(f"Subsystem: {subsystem_name}")
            print(f"Mean: {np.mean(samples)}")
            print(f"Standard Deviation: {np.std(samples)}")
            print(f"95% Credible Interval: {np.quantile(samples, [0.025, 0.975])}")
            print()
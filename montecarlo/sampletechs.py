import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
import seaborn as sns
from network import Network
import sobol_seq

class Sampler:
    def __init__(self, config):
        self.config = config
        self.input_params = self.identify_input_params()
        self.param_distributions = self.assign_probability_distributions()
    
    def identify_input_params(self):
        input_params = []
        for subsystem, subsystem_config in self.config.items():
            if subsystem not in ["electricity_emission_factor", "iterations"]:
                for key, value in subsystem_config.items():
                    if key not in ["type", "next_subsystem"]:
                        input_params.append(f"{subsystem}.{key}")
        return input_params
    
    def assign_probability_distributions(self):
        param_distributions = {}
        for param in self.input_params:
            if "flow_rate" in param:
                loc = self.config[param.split(".")[0]][param.split(".")[1]]
                scale = 10
                param_distributions[param] = norm(loc=loc, scale=scale)
            elif "temperature" in param:
                low = self.config[param.split(".")[0]][param.split(".")[1]] - 10
                high = self.config[param.split(".")[0]][param.split(".")[1]] + 10
                param_distributions[param] = uniform(loc=low, scale=high - low)
            elif "pressure" in param:
                low = self.config[param.split(".")[0]][param.split(".")[1]] - 1000
                high = self.config[param.split(".")[0]][param.split(".")[1]] + 1000
                param_distributions[param] = uniform(loc=low, scale=high - low)
            elif "efficiency" in param:
                low = 0.5
                high = 0.9
                param_distributions[param] = uniform(loc=low, scale=high - low)
            elif "electricity_conversion" in param:
                low = 2
                high = 5
                param_distributions[param] = uniform(loc=low, scale=high - low)
            else:
                low = 0
                high = 1
                param_distributions[param] = uniform(loc=low, scale=high - low)
        return param_distributions
    
    def generate_quasi_samples(self, num_samples):
        samples = {}
        sobol_sequence = sobol_seq.i4_sobol_generate(len(self.input_params), num_samples)
        for i, param in enumerate(self.input_params):
            distribution = self.param_distributions[param]
            samples[param] = distribution.ppf(sobol_sequence[:, i])
        return samples
    
    def generate_latin_samples(self, num_samples):
        samples = {}
        lhs_samples = np.random.uniform(0, 1, size=(num_samples, len(self.input_params)))
        for i, param in enumerate(self.input_params):
            distribution = self.param_distributions[param]
            lhs_samples[:, i] = distribution.ppf(lhs_samples[:, i])
        for i, param in enumerate(self.input_params):
            samples[param] = lhs_samples[:, i]
        return samples
    
    def run_quasi_simulation(self, num_samples):
        # Run the simulation
        results = []
        total_emissions_results = []
        for i in range(num_samples):
            config_sample = self.config.copy()
            for param, value in self.generate_quasi_samples(num_samples).items():
                subsystem, key = param.split(".")
                if key == "flow_rate":
                    config_sample[subsystem]["input_stream"]["flow_rate"] = value[i]
                elif key == "temperature":
                    config_sample[subsystem]["input_stream"]["temperature"] = value[i]
                elif key == "pressure":
                    config_sample[subsystem]["input_stream"]["pressure"] = value[i]
                elif key == "efficiency":
                    config_sample[subsystem]["efficiency"] = value[i]
                elif key == "electricity_conversion":
                    config_sample[subsystem]["electricity_conversion"] = value[i]
            network = Network(config_sample)
            network.build_network()
            network.simulate()
            subsystem_emissions, total_emissions = network.calculate_emissions()
            results.append(subsystem_emissions)
            if subsystem_emissions is not None:
                results.append(subsystem_emissions)
            total_emissions_results.append(total_emissions)
        
        if total_emissions_results:
            all_emissions = total_emissions_results
            mean_all_emissions = np.mean(total_emissions_results)
            std_all_emissions = np.std(total_emissions_results)

            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.hist(all_emissions, bins=100, density=True, alpha=0.6, label='All Emissions')
            lower_bound = np.percentile(all_emissions, 5)
            upper_bound = np.percentile(all_emissions, 95)
            ax.set_xlim(lower_bound, upper_bound)
            # PDF
            x = np.linspace(lower_bound, upper_bound, 100)
            pdf = norm.pdf(x, loc=mean_all_emissions, scale=std_all_emissions)
            ax.plot(x, pdf, 'k', linewidth=2, label='Normal Distribution')
            sns.kdeplot(all_emissions, fill=True, label='Kernel Density Estimate of Emissions', ax=ax)

            ax.set_title('Histogram of Total Emissions with Normal Distribution')
            ax.set_xlabel('Emissions (kg CO2)')
            ax.set_ylabel('Probability Density')
            ax.legend()
            plt.show()

        return results

    def run_latin_simulation(self, num_samples):
        # Run the simulation
        results = []
        total_emissions_results = []
        for i in range(num_samples):
            config_sample = self.config.copy()
            for param, value in self.generate_latin_samples(num_samples).items():
                subsystem, key = param.split(".")
                if key == "flow_rate":
                    config_sample[subsystem]["input_stream"]["flow_rate"] = value[i]
                elif key == "temperature":
                    config_sample[subsystem]["input_stream"]["temperature"] = value[i]
                elif key == "pressure":
                    config_sample[subsystem]["input_stream"]["pressure"] = value[i]
                elif key == "efficiency":
                    config_sample[subsystem]["efficiency"] = value[i]
                elif key == "electricity_conversion":
                    config_sample[subsystem]["electricity_conversion"] = value[i]
            network = Network(config_sample)
            network.build_network()
            network.simulate()
            subsystem_emissions, total_emissions = network.calculate_emissions()
            results.append(subsystem_emissions)
            if subsystem_emissions is not None:
                results.append(subsystem_emissions)
            total_emissions_results.append(total_emissions)
        
        if total_emissions_results:
            all_emissions = total_emissions_results
            mean_all_emissions = np.mean(total_emissions_results)
            std_all_emissions = np.std(total_emissions_results)

            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.hist(all_emissions, bins=100, density=True, alpha=0.6, label='All Emissions')
            lower_bound = np.percentile(all_emissions, 5)
            upper_bound = np.percentile(all_emissions, 95)
            ax.set_xlim(lower_bound, upper_bound)
            # PDF
            x = np.linspace(lower_bound, upper_bound, 100)
            pdf = norm.pdf(x, loc=mean_all_emissions, scale=std_all_emissions)
            ax.plot(x, pdf, 'k', linewidth=2, label='Normal Distribution')
            sns.kdeplot(all_emissions, fill=True, label='Kernel Density Estimate of Emissions', ax=ax)

            ax.set_title('Histogram of Total Emissions with Normal Distribution')
            ax.set_xlabel('Emissions (kg CO2)')
            ax.set_ylabel('Probability Density')
            ax.legend()
            plt.show()

        return results

    def plot_subsystems(self, results):
        if results:

            subsystem_emissions_mean = {}
            subsystem_emissions_std = {}

            for key in results[0].keys():
                values = [result[key] for result in results]
                subsystem_emissions_mean[key] = np.mean(values)
                subsystem_emissions_std[key] = np.std(values)
            print(f"Mean emissions: {subsystem_emissions_mean}")
            print(f"Standard deviation of emissions: {subsystem_emissions_std}")
            
            # Histogram and error-bars for each subsystem
            for subsystem_name in subsystem_emissions_mean.keys():
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

                subsystem_emissions = [result[subsystem_name] for result in results]
                if subsystem_emissions:
                    #Histograms
                    axs[0].hist(subsystem_emissions, bins=100, density=True, alpha=0.6, label=subsystem_name, range=(0, subsystem_emissions_mean[subsystem_name]*2))
                    lower_bound = np.percentile(subsystem_emissions, 5)
                    upper_bound = np.percentile(subsystem_emissions, 95)
                    axs[0].set_xlim(lower_bound, upper_bound)
            
                    # PDF
                    x = np.linspace(lower_bound, upper_bound, 100)
                    pdf = norm.pdf(x, loc=np.mean(subsystem_emissions), scale=np.std(subsystem_emissions))
                    axs[0].plot(x, pdf, 'k', linewidth=2, label='Normal Distribution')
                    sns.kdeplot(subsystem_emissions, fill=True, label='Kernel Density Estimate of Emissions', ax=axs[0])
                    
                    axs[0].set_title('Histogram of Emissions with Normal Distribution')
                    axs[0].set_xlabel('Emissions (kg CO2)')
                    axs[0].set_ylabel('Probability Density')
                    axs[0].legend()

                    #ErrorBars
                    mean_emissions = np.mean(subsystem_emissions)
                    std_emissions = np.std(subsystem_emissions)
                    axs[1].errorbar(subsystem_name, mean_emissions, yerr=std_emissions, fmt='o', capsize=5, label=subsystem_name)
                    axs[1].fill_between([subsystem_name], mean_emissions - 2*std_emissions, mean_emissions + 2*std_emissions, color='gray', alpha=0.2, label='95% Confidence Interval')
                    axs[1].set_xlim(-0.5, 0.5)
                    axs[1].set_xticks([0])
                    axs[1].set_xticklabels([subsystem_name])
                    axs[1].set_xlabel('Subsystem')
                    axs[1].set_ylabel('Emissions (kg CO2)')
                    axs[1].set_title('Error Bars for Subsystem Emissions')
                    axs[1].legend()
                    
                    plt.tight_layout()
                    plt.show()

        else:
            print("No valid emissions results found.")

        
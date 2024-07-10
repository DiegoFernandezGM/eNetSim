import numpy as np
import matplotlib.pyplot as plt
from network import Network 

class UncertaintyAnalysis:
    def __init__(self, config):
        self.config = config
        
    def calculate_emission_uncertainty(self, iteration=1):
        total_emissions_results = []
        total_pert_emissions_results = []
        parameter_labels = []
        uncertainty = []

        perturbation = 1e-6
  
        for param, details in self.config.items():
            if isinstance(details, dict):
                config_sample = self.config.copy()
                config_pert = self.config.copy()
                for key, value in details.items():
                    if key == "input_stream":
                        for sub_key, sub_value in value.items():
                            if sub_key in ["flow_rate", "temperature", "pressure"]:
                                config_sample[param]["input_stream"][sub_key] = sub_value
                                config_pert[param]["input_stream"][sub_key] = sub_value + perturbation
                                parameter_labels.append(f"{param}.input_stream.{sub_key}")
                                
                                mean_total_emissions, _ = self.calculate_mean_emissions(config_sample, iteration)
                                mean_pert_emissions, _ = self.calculate_mean_emissions(config_pert, iteration)

                                if mean_total_emissions is not None and mean_pert_emissions is not None:
                                    total_emissions_results.append(mean_total_emissions)
                                    total_pert_emissions_results.append(mean_pert_emissions)
                    elif key == "output_streams":
                        for i, output_stream in enumerate(value):
                            for sub_key, sub_value in output_stream.items():
                                if sub_key in ["temperature", "pressure"]:
                                    config_sample[param]["output_streams"][i][sub_key] = sub_value
                                    config_pert[param]["output_streams"][i][sub_key] = sub_value + perturbation
                                    parameter_labels.append(f"{param}.output_streams.{i}.{sub_key}")
                                    
                                    mean_total_emissions, _ = self.calculate_mean_emissions(config_sample, iteration)
                                    mean_pert_emissions, _ = self.calculate_mean_emissions(config_pert, iteration)

                                    if mean_total_emissions is not None and mean_pert_emissions is not None:
                                        total_emissions_results.append(mean_total_emissions)
                                        total_pert_emissions_results.append(mean_pert_emissions)
                    elif key in ["efficiency", "electricity_conversion"]:
                        config_sample[param][key] = value
                        config_pert[param][key] = value + perturbation
                        parameter_labels.append(f"{param}.{key}")
                        mean_total_emissions, _ = self.calculate_mean_emissions(config_sample, iteration)
                        mean_pert_emissions, _ = self.calculate_mean_emissions(config_pert, iteration)

                        if mean_total_emissions is not None and mean_pert_emissions is not None:
                            total_emissions_results.append(mean_total_emissions)
                            total_pert_emissions_results.append(mean_pert_emissions)
           
        mean_total_emissions = np.mean(total_emissions_results)
        
        derivatives, variances = self.calculate_derivatives_and_variances(total_pert_emissions_results, mean_total_emissions, perturbation)
        
        uncertainty = np.sqrt(np.dot(derivatives ** 2, variances))

        return mean_total_emissions, parameter_labels, uncertainty
    
    def calculate_derivatives_and_variances(self, total_pert_emissions_results, mean_total_emissions, perturbation):
        num_params = len(total_pert_emissions_results)
        derivatives = np.zeros(num_params)
        
        for i in range(num_params):
            derivatives[i] = (total_pert_emissions_results[i] - mean_total_emissions) / perturbation
        
        variances = np.var(total_pert_emissions_results, axis=0)
        
        return derivatives, variances
    
    def calculate_mean_emissions(self, config_sample, iteration):
        network = Network(config_sample)
        network.build_network()
        network.simulate()
        
        subsystem_emissions, total_emissions = network.calculate_emissions()
        
        return total_emissions, subsystem_emissions
    
    def visualize_emission_uncertainty(self, parameter_labels, uncertainties):
        plt.figure(figsize=(16, 10))

        bars = plt.bar(parameter_labels, uncertainties, color='lightcoral')
        plt.xlabel('Parameters')
        plt.ylabel('Parameter Uncertainty')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(uncertainties) * 1.5)
        plt.tight_layout()

        for bar, unc in zip(bars, uncertainties):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.000005, f'{unc:.2e}', ha='center', va='bottom', rotation=90)

        plt.show()

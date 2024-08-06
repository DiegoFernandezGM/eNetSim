import numpy as np
import matplotlib.pyplot as plt
from network import Network 
from distribution import DistributionAnalyzer
import scipy.stats as stats
import textwrap
import copy

class UncertaintyAnalysis:
    def __init__(self, config):
        self.config = config
        self.input_params, self.output_params = self.identify_input_params()
        self.param_distributions = self.assign_probability_distributions()

    def compute_partial_derivatives(self):
        partial_derivatives = {}
        for param, details in self.config.items():
            if isinstance(details, dict):
                for key, value in details.items():
                    if key == "input_stream":
                        for sub_key, sub_value in value.items():
                            if sub_key in ["flow_rate", "temperature", "pressure"]:
                                partial_derivative = self.compute_partial_derivative(param, key, sub_key)
                                partial_derivatives[(param, key, sub_key)] = partial_derivative
                    elif key == "output_streams":
                        for i, output_stream in enumerate(value):
                            for sub_key, sub_value in output_stream.items():
                                if sub_key in ["temperature", "pressure"]:
                                    partial_derivative = self.compute_partial_derivative(param, key, sub_key, i)
                                    partial_derivatives[(param, key, i, sub_key)] = partial_derivative
                    elif key in ["efficiency", "electricity_conversion"]:
                        partial_derivative = self.compute_partial_derivative(param, key)
                        partial_derivatives[(param, key)] = partial_derivative
        return partial_derivatives

    def compute_partial_derivative(self, param, key, sub_key=None, index=None):
        perturbation_factor = 0.95 # Smaller perturbation size
        if sub_key:
            if key == "output_streams" and index is not None:
                original_value = self.config[param][key][index][sub_key]
            else:
                original_value = self.config[param][key][sub_key]
        else:
            original_value = self.config[param][key]
        
        perturbed_value = original_value*perturbation_factor
        perturbed_config = copy.deepcopy(self.config)
        
        if sub_key:
            if key == "output_streams" and index is not None:
                perturbed_config[param][key][index][sub_key] *= perturbation_factor
            else:
                perturbed_config[param][key][sub_key] *= perturbation_factor
        else:
            perturbed_config[param][key] *= perturbation_factor
        
        perturbed_network = Network(perturbed_config)
        perturbed_network.build_network()
        perturbed_network.simulate()
        perturbed_emissions, _ = perturbed_network.calculate_emissions()
        
        network = Network(self.config)
        network.build_network()
        network.simulate()
        original_emissions, _ = network.calculate_emissions()
        print(f"perturbed:  {perturbed_value}")
        print(f"original:  {original_value}")
        partial_derivatives = {}
        for emission_key in perturbed_emissions:
            print(f"Perturbed:{perturbed_emissions[emission_key]}")
            print(f"original:{original_emissions[emission_key]}")
            partial_derivative = (perturbed_emissions[emission_key] - original_emissions[emission_key]) / perturbation_factor
            partial_derivatives[emission_key] = partial_derivative

        # Debugging print statement to check partial derivatives
        print(f"Partial derivative for {param}.{key}{'.' + sub_key if sub_key else ''}: {partial_derivative}")

        return partial_derivatives

    def compute_uncertainties(self):
        uncertainties = {}
        for param, details in self.config.items():
            if isinstance(details, dict):
                for key, value in details.items():
                    if key == "input_stream":
                        for sub_key, sub_value in value.items():
                            if sub_key in ["flow_rate", "temperature", "pressure"]:
                                full_param = f"{param}.input_stream.{sub_key}"
                                uncertainties[full_param] = self.compute_uncertainty(full_param, param, key)
                    elif key == "output_streams":
                        for i, output_stream in enumerate(value):
                            for sub_key, sub_value in output_stream.items():
                                if sub_key in ["temperature", "pressure"]:
                                    full_param = f"{param}.output_streams.{i}.{sub_key}"
                                    uncertainties[full_param] = self.compute_uncertainty(full_param, param, key)
                    elif key in ["efficiency", "electricity_conversion"]:
                        full_param = f"{param}.{key}"
                        uncertainties[full_param] = self.compute_uncertainty(full_param, param)
        return uncertainties

    def compute_uncertainty(self, param, module_name, sub_key=None):
        if param in self.param_distributions:
            return self.param_distributions[param]
        else:
            raise ValueError(f"No probability distribution found for parameter {param} in module {module_name}")

    def identify_input_params(self):
        input_params = []
        output_params = []

        def recursive_identify(subsystem_config):
            for key, value in subsystem_config.items():
                if key == "input_stream":
                    for sub_key, sub_value in value.items():
                        input_params.append(f"{subsystem_config['type']}.input_stream.{sub_key}")
                elif key == "output_streams":
                    for i, output_stream in enumerate(value):
                        for sub_key, sub_value in output_stream.items():
                            output_params.append(f"{subsystem_config['type']}.output_streams.{i}.{sub_key}")
                elif key in ["efficiency", "electricity_conversion"]:
                    input_params.append(f"{subsystem_config['type']}.{key}")
                elif isinstance(value, dict):
                    recursive_identify(value)

        for subsystem, subsystem_config in self.config.items():
            if subsystem not in ["electricity_emission_factor", "iterations"]:
                recursive_identify(subsystem_config)

        return input_params, output_params

    def assign_probability_distributions(self):
        param_distributions = {}
        for param in self.input_params + self.output_params:
            parts = param.split(".")
            if len(parts) == 4:
                module_name, output_streams_name, stream_index, sub_param_name = parts
                stream_index = int(stream_index)
                output_streams = self.config[module_name][output_streams_name]
                data = output_streams[stream_index][sub_param_name]
                # Assign probability distribution based on data
                if sub_param_name == "temperature":
                    param_distributions[param] = stats.uniform(loc=data - 10, scale=20)
                elif sub_param_name == "pressure":
                    param_distributions[param] = stats.uniform(loc=data - 100, scale=200)
            elif len(parts) == 3:
                module_name, input_stream_name, sub_param_name = parts
                input_stream = self.config[module_name][input_stream_name]
                if sub_param_name in input_stream:
                    data = input_stream[sub_param_name]
                    if sub_param_name == "flow_rate":
                        param_distributions[param] = stats.uniform(loc=input_stream[sub_param_name] * 0.9, scale=input_stream[sub_param_name] * 0.2)
                    elif sub_param_name == "temperature":
                        param_distributions[param] = stats.uniform(loc=input_stream[sub_param_name] - 10, scale=20)
                    elif sub_param_name == "pressure":
                        param_distributions[param] = stats.uniform(loc=input_stream[sub_param_name] - 100, scale=200)
            elif len(parts) == 2:
                module_name, param_name = parts
                if param_name == "efficiency":
                    param_distributions[param] = stats.uniform(loc=self.config[module_name][param_name] * 0.9, scale=0.1)
                elif param_name == "electricity_conversion":
                    param_distributions[param] = stats.uniform(loc=self.config[module_name][param_name] * 0.99, scale=0.02)
        return param_distributions

    def estimate_uncertainties(self, partial_derivatives, uncertainties):
        uncertainty_emissions = {}
        for key, partial_derivative in partial_derivatives.items():
            if len(key) == 2:
                module_name, param_name = key
                full_param = f"{module_name}.{param_name}"
            elif len(key) == 3:
                module_name, stream_name, sub_param_name = key
                full_param = f"{module_name}.{stream_name}.{sub_param_name}"
            elif len(key) == 4:
                module_name, stream_name, index, sub_param_name = key
                full_param = f"{module_name}.{stream_name}.{index}.{sub_param_name}"
            else:
                raise ValueError("Invalid key length")
            if full_param not in uncertainties:
                raise KeyError(f"Missing uncertainty for parameter: {full_param}")
            distribution = uncertainties[full_param]
            samples = distribution.rvs(size=1000)  # Sample from the distribution 1000 times
            uncertainty_emission_samples = np.zeros(1000)
            for value in partial_derivative.values():
                partial_derivative_samples = np.random.normal(value, 0.01, size=1000)  # Sample from a normal distribution with mean=value and std=0.01
                uncertainty_emission_samples += partial_derivative_samples * samples
            uncertainty_emission = np.std(uncertainty_emission_samples)  # Calculate the standard deviation of the samples
            uncertainty_emissions[key] = uncertainty_emission

            # Debugging print statement to check uncertainty estimation
            print(f"Uncertainty for {key}: {uncertainty_emission}")

        return uncertainty_emissions

    def run(self):
        partial_derivatives = self.compute_partial_derivatives()
        uncertainties = self.compute_uncertainties()
        uncertainty_emissions = self.estimate_uncertainties(partial_derivatives, uncertainties)
        return uncertainty_emissions

    def plot_uncertainties(self, uncertainty_emissions):
        plt.figure(figsize=(16, 10))
        keys = list(uncertainty_emissions.keys())
        values = list(uncertainty_emissions.values())
        bars = plt.bar(range(len(values)), values)
        # Use the full parameter names as labels without wrapping
        labels = [f"{key[0]}.{key[1]}" + (f".{key[2]}" if len(key) > 2 else "") + (f".{key[3]}" if len(key) > 3 else "") for key in keys]
        plt.xticks(range(len(values)), labels, rotation=45, ha='right')
        plt.xlabel("Input Parameters")
        plt.ylabel("Uncertainty in Emissions")
        plt.title("Uncertainties in Emissions Calculations")
        for bar, unc in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.000005, f'{unc:.2e}', ha='center', va='bottom', rotation=90)

        plt.tight_layout()  # Adjust the layout to prevent label cutoff
        plt.show()
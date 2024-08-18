from importer import *

from network import Network 
from distribution import DistributionAnalyzer

class UncertaintyAnalysis:
    def __init__(self, config):
        self.config = config
        self.input_params, self.output_params = self.identify_input_params()
        self.param_distributions = self.assign_probability_distributions()

    def compute_partial_derivatives(self):

        ##   Computes partial derivatives for each identified system' parameter
        ##   These are used to estimate how small input changes affect the output emissions
             
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
        
        ##  Computes the partial derivative of the emission output w.r.t the given parameter
        ##  The derivative is estimated via a perturbation approach
        
        perturbation_factor = 0.95 
        if sub_key:
            if key == "output_streams" and index is not None:
                original_value = self.config[param][key][index][sub_key]
            else:
                original_value = self.config[param][key][sub_key]
        else:
            original_value = self.config[param][key]
        
        # Apply perturbation to the parameter value
        perturbed_value = original_value*perturbation_factor
        perturbed_config = copy.deepcopy(self.config)
        
        if sub_key:
            if key == "output_streams" and index is not None:
                perturbed_config[param][key][index][sub_key] *= perturbation_factor
            else:
                perturbed_config[param][key][sub_key] *= perturbation_factor
        else:
            perturbed_config[param][key] *= perturbation_factor
        
        # Compute emissions with perturbed configuration
        perturbed_network = Network(perturbed_config)
        perturbed_network.build_network()
        perturbed_network.simulate()
        perturbed_emissions, _ = perturbed_network.calculate_emissions()
        
        # Compute original emissions for comparison
        network = Network(self.config)
        network.build_network()
        network.simulate()
        original_emissions, _ = network.calculate_emissions()
        
        # Calculate partial derivatives for each subsystem emission
        partial_derivatives = {}
        for emission_key in perturbed_emissions:
            partial_derivative = (perturbed_emissions[emission_key] - original_emissions[emission_key]) / perturbation_factor
            partial_derivatives[emission_key] = partial_derivative

        # Debugging output for partial derivatives
        # print(f"Partial derivative for {param}.{key}{'.' + sub_key if sub_key else ''}: {partial_derivative}")

        return partial_derivatives

    def compute_uncertainties(self):

        ##  Computes the uncertainty associated with each in/output parameter

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
        
        ##  Retrieves the probability distribution associated with a specific parameter
        
        if param in self.param_distributions:
            return self.param_distributions[param]
        else:
            raise ValueError(f"No probability distribution found for parameter {param} in module {module_name}")

    def identify_input_params(self):

        ##  Identifies all in/output parameters in the configuration
        ##  Necessary for assigning probability distributions to these identified parameters


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
        
        ##  Assigns probability distributions to each identified in/output parameter
        ##  Distributions used to model the uncertainties in the parameters
        ##  Assigns distributions based on the parameter type

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
        
        ##  Estimates emission uncertainties by combining partial derivatives
        ##  with the uncertainties in the input parameters

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
             
            # Sample from distribution and compute emission uncertainty
            distribution = uncertainties[full_param]
            samples = distribution.rvs(size=1000) # Generate 1000 samples
            uncertainty_emission_samples = np.zeros(1000)
            for value in partial_derivative.values():
                # Sample from a normal distribution with mean=value and std=0.01
                partial_derivative_samples = np.random.normal(value, 0.01, size=1000)  
                uncertainty_emission_samples += partial_derivative_samples * samples
            
            # Calculate stddev of the emission samples as the uncertainty
            uncertainty_emission = np.std(uncertainty_emission_samples)
            uncertainty_emissions[key] = uncertainty_emission

            # Debugging output for uncertainty estimation
            print(f"Uncertainty for {key}: {uncertainty_emission}")

        return uncertainty_emissions

    def run(self):

        ## Main method to run the UA

        logging.info("Starting Probabilistic (1st Taylor Series) Uncertainty Analysis")
        start_time = time.time()
        
        # Compute partial derivatives and uncertainties
        partial_derivatives = self.compute_partial_derivatives()
        uncertainties = self.compute_uncertainties()
        
        # Estimate the uncertainties in emissions
        uncertainty_emissions = self.estimate_uncertainties(partial_derivatives, uncertainties)
        
        end_time = time.time()
        logging.info("Simulation took {:.2f} seconds".format(end_time - start_time))
        return uncertainty_emissions

    def plot_uncertainties(self, uncertainty_emissions):
        
        ##  Plots  estimated emissions uncertainties
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        keys = list(uncertainty_emissions.keys())
        values = list(uncertainty_emissions.values())
        
        # Generate parameter labels 
        labels = [
            f"{key[0]}.{key[1]}" + (f".{key[2]}" if len(key) > 2 else "") + (f".{key[3]}" if len(key) > 3 else "")
            for key in keys
        ]

        # Create horizontal bar chart
        bars = ax.barh(range(len(values)), values, color='#A0522D', edgecolor='black', height=0.6)
        
        # Set chart labels and title
        ax.set_xlabel('Uncertainty in Emissions [kg CO2/kg NH3]', fontsize=14, fontweight='bold', family='Arial')
        ax.set_ylabel('Input Parameters', fontsize=14, fontweight='bold', family='Arial')
        ax.set_title('Parameter Uncertainty - Probabilistic Route - 1st Order Taylor', fontsize=16, fontweight='bold', family='Arial', pad=20)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels, fontsize=12, family='Arial')
        ax.tick_params(axis='x', labelsize=12, labelcolor='black', direction='out')

        # Adjust x-axis limits and add grid lines
        ax.set_xlim(0, max(values) * 1.1)
        ax.grid(True, which='major', axis='x', linestyle='-', linewidth=0.5, color='lightgray')
        
        # Invert y-axis for better readability
        ax.invert_yaxis()  

        # Annotate bars with their respective uncertainty values
        for bar, unc in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, f'{unc:.2e}', 
                    va='center', ha='left', fontsize=10, family='Arial')

        plt.tight_layout(pad=3)
        plt.show()
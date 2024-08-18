from importer import *

from network import Network
from distribution import DistributionAnalyzer

class Sampler:
    def __init__(self, config):

        ##  Initialises the Sampler class (for QMCS & LHS) with the given configuration
        ##  Identifies in/output parameters and assigns probability distributions

        self.config = config
        self.input_params, self.output_params = self.identify_input_params()
        self.param_distributions = self.assign_probability_distributions()

    def identify_input_params(self):

        ##  Identifies in/output params from the config, and returns them in lists: input_params, output_params

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
        
        ##  Assigns probability distributions to each identified param based on its characteristics (config)
        ##  Provides dictionary mapping parameters to their respective distributions
        ##  If params are provided in ranges of possible values: DistributionAnalyzer finds best-fit distribution 

        param_distributions = {}
        for param in self.input_params + self.output_params:
            parts = param.split(".")
            if len(parts) == 4:
                # Handling of output streams
                module_name, output_streams_name, stream_index, sub_param_name = parts
                stream_index = int(stream_index)
                output_streams = self.config[module_name][output_streams_name]
                data = output_streams[stream_index][sub_param_name]
                # Assign probability distribution based on data type
                if sub_param_name == "temperature":
                    if isinstance(data, list) or isinstance(data, tuple):
                        if len(data) == 1:
                            param_distributions[param] = stats.uniform(loc=data[0] - 5, scale=20)
                        else:
                            handler = DistributionAnalyzer(data)
                            param_distributions[param] = handler.analyze_and_sample()
                    else:
                        param_distributions[param] = stats.uniform(loc=data - 10, scale=20)
                elif sub_param_name == "pressure":
                    if isinstance(data, list) or isinstance(data, tuple):
                        if len(data) == 1:
                            param_distributions[param] = stats.uniform(loc=data[0] - 100, scale=200)
                        else:
                            handler = DistributionAnalyzer(data)
                            param_distributions[param] = handler.analyze_and_sample()
                    else:
                        param_distributions[param] = stats.uniform(loc=data - 100, scale=200)
            elif len(parts) == 3:
                # Handling of input streams
                module_name, input_stream_name, sub_param_name = parts
                input_stream = self.config[module_name][input_stream_name]
                if sub_param_name in input_stream:
                    data = input_stream[sub_param_name]
                    if sub_param_name == "flow_rate":
                        if isinstance(data, list) or isinstance(data, tuple):
                            if len(data) == 1:
                                param_distributions[param] = stats.uniform(loc=data[0]*0.9, scale=data[0]*1.1 - data[0]*0.9)
                            else:
                                handler = DistributionAnalyzer(data)
                                param_distributions[param] = handler.analyze_and_sample()
                        else:
                            param_distributions[param] = stats.uniform(loc=data*0.9, scale=data*1.1 - data*0.9)
                    elif sub_param_name == "temperature":
                        if isinstance(data, list) or isinstance(data, tuple):
                            if len(data) == 1:
                                param_distributions[param] = stats.uniform(loc=data[0] - 5, scale=20)
                            else:
                                handler = DistributionAnalyzer(data)
                                param_distributions[param] = handler.analyze_and_sample()
                        else:
                            param_distributions[param] = stats.uniform(loc=data - 10, scale=20)
                    elif sub_param_name == "pressure":
                        if isinstance(data, list) or isinstance(data, tuple):
                            if len(data) == 1:
                                param_distributions[param] = stats.uniform(loc=data[0] - 100, scale=200)
                            else:
                                handler = DistributionAnalyzer(data)
                                param_distributions[param] = handler.analyze_and_sample()
                        else:
                            param_distributions[param] = stats.uniform(loc=data - 100, scale=200)
            elif len(parts) == 2:
                # Handling of efficiency and electricity conversion
                module_name, param_name = parts
                if param_name == "efficiency":
                    data = self.config[module_name][param_name]
                    param_distributions[param] = stats.uniform(loc=data*0.9, scale=0.9 - data*0.9)
                elif param_name == "electricity_conversion":
                    data = self.config[module_name][param_name]
                    param_distributions[param] = stats.uniform(loc=data*0.99, scale=data*1.01 - data*0.99)
        return param_distributions
    
    def generate_quasi_samples(self, num_samples):
        
        ##  Generates quasi-random samples using Sobol sequence.
        
        samples = {}
        sobol_sequence = sobol_seq.i4_sobol_generate(len(self.param_distributions), num_samples)
        
        for i, (param, distribution) in enumerate(self.param_distributions.items()):
            samples[param] = [distribution.ppf(sobol_sequence[j, i]) for j in range(num_samples)]
        
        return samples

    def generate_latin_samples(self, num_samples):
        
        ##  Generates Latin Hypercube samples

        samples = {}
        lhs_samples = np.random.uniform(0, 1, size=(num_samples, len(self.param_distributions)))
        
        for i, (param, distribution) in enumerate(self.param_distributions.items()):
            sampled_values = distribution.ppf(lhs_samples[:, i])
            samples[param] = sampled_values
        
        return samples

    def run_quasi_simulation(self, num_samples, technique, output_file=None, sensitivity = True):
        
        ##  Runs a QMCS, collects emissions results and performs SA if specified

        logging.info("Starting simulation with {} samples and technique {}".format(num_samples, technique))
        try:
            start_time = time.time()
            results = []
            total_emissions_results = []
            self.flow_rate_results = []
            self.temperature_results = []
            self.pressure_results = []
            self.efficiency_results = []
            self.electricity_conversion_results = []
            parameter_emissions_results = {}

            samples = self.generate_quasi_samples(num_samples)
            for param, value in samples.items():
                parameter_emissions_results[param] = []

            for i in range(num_samples):
                config_sample = copy.deepcopy(self.config)
                for param, value in samples.items():
                    logging.debug(f"Handling parameter: {param}, with value: {value[i]}")

                    keys = param.split(".")
                    if len(keys) == 4:
                        if keys[1] == "output_streams":
                            module_name, output_streams_name, stream_index, sub_param_name = keys
                            stream_index = int(stream_index)
                            output_streams = config_sample[module_name][output_streams_name]
                            output_streams[stream_index][sub_param_name] = value[i]
                            if sub_param_name == "temperature":
                                self.temperature_results.append(value[i])
                            elif sub_param_name == "pressure":
                                self.pressure_results.append(value[i])
                    elif len(keys) == 3:
                        module_name, input_stream_name, sub_param_name = keys
                        input_stream = config_sample[module_name][input_stream_name]
                        if isinstance(input_stream, dict):
                            input_stream[sub_param_name] = value[i]
                        if sub_param_name == "flow_rate":
                            self.flow_rate_results.append(value[i])
                        elif sub_param_name == "temperature":
                            self.temperature_results.append(value[i])
                        elif sub_param_name == "pressure":
                            self.pressure_results.append(value[i])
                    elif len(keys) == 2:
                        module_name, param_name = keys
                        config_sample[module_name][param_name] = value[i]
                        if param_name == "efficiency":
                            self.efficiency_results.append(value[i])
                        elif param_name == "electricity_conversion":
                            self.electricity_conversion_results.append(value[i])
                
                # Build and simulate the network for the current sample
                network = Network(config_sample)
                network.build_network()
                network.simulate()
                
                # Record total and unit emission estimates
                subsystem_emissions, total_emissions = network.calculate_emissions()
                results.append(subsystem_emissions)
                if subsystem_emissions is not None:
                    results.append(subsystem_emissions)
                total_emissions_results.append(total_emissions)
                
                # Classify emissiones estimated based on parameter-type sampled
                for param, value in samples.items():
                    keys = param.split(".")
                    
                    if len(keys) == 4:
                        if keys[1] == "output_streams":
                            module_name, output_streams_name, stream_index, sub_param_name = keys
                            stream_index = int(stream_index)
                            subsystem_name = f"{module_name}.{output_streams_name}"
                            if module_name in subsystem_emissions:
                                if sub_param_name == "temperature":
                                    parameter_emissions_results[param].append(subsystem_emissions[module_name])
                                elif sub_param_name == "pressure":
                                    parameter_emissions_results[param].append(subsystem_emissions[module_name])
                    elif len(keys) == 3:
                        module_name, input_stream_name, sub_param_name = keys
                        if module_name in subsystem_emissions:
                            if sub_param_name == "flow_rate":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
                            elif sub_param_name == "temperature":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
                            elif sub_param_name == "pressure":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
                    elif len(keys) == 2:
                        module_name, param_name = keys
                        if module_name in subsystem_emissions:
                            if param_name == "efficiency":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
                            elif param_name == "electricity_conversion":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
            
            # Calculate parameter-type uncertainty in emission calculation based on stddev of all results
            parameter_uncertainties = {}
            for param, emissions in parameter_emissions_results.items():
                parameter_uncertainties[param] = np.std(emissions)
            
            end_time = time.time()
            logging.info("Simulation took {:.2f} seconds".format(end_time - start_time))

            # Plot the total system emission results if available
            if total_emissions_results:
                all_emissions = total_emissions_results
                mean_all_emissions = np.mean(total_emissions_results)
                std_all_emissions = np.std(total_emissions_results)
                
                sns.set_style('whitegrid')
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.hist(all_emissions, bins=100, density=True, alpha=0.6, label='All Emissions')
                lower_bound = np.percentile(all_emissions, 5)
                upper_bound = np.percentile(all_emissions, 95)
                ax.set_xlim(lower_bound, upper_bound)
                
                # Plot normal distribution PDF
                x = np.linspace(lower_bound, upper_bound, 100)
                pdf = norm.pdf(x, loc=mean_all_emissions, scale=std_all_emissions)
                ax.plot(x, pdf, 'k', linewidth=2, label='Normal Distribution')
                sns.kdeplot(all_emissions, fill=True, label='Kernel Density Estimate of Emissions', ax=ax)

                # Set chart labels, title and legend
                ax.set_title('Histogram of Total Emissions with Normal Distribution - {}'.format(technique), fontsize=16, fontweight='bold', family='Arial')
                ax.set_xlabel('Emissions [kg CO2/kg NH3]', fontsize=14, fontweight='bold', family='Arial')
                ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold', family='Arial')
                ax.legend()
                
                plt.show()
            
            print("---------------")
            print(f"Technique {technique}")
            print("---------------")
            print("All Emissions - Simulation Results:")
            print("Mean: {:.2f}".format(mean_all_emissions))
            print("Standard Deviation: {:.2f}".format(std_all_emissions))
            print("---------------")

            # Write results to the output file if specified
            if output_file is not None:
                mode = 'w'
            else:
                mode = 'a'

            with open(output_file, mode, newline='') as f:
                writer = csv.writer(f)
                if mode == 'w':
                    writer.writerow(["Subsystem", "Emissions"])
                for result in results:
                    for subsystem, emissions in result.items():
                        writer.writerow([subsystem, emissions])        
            
            # Call Sobol sensitivity analysis function if requested
            if sensitivity:
                self.sensitivity_sobol(total_emissions_results)
                
            return results, parameter_uncertainties
        
        except Exception as e:
            logging.error("Error in simulation: {}".format(e))
            return None, None
        logging.info("Simulation completed")

    def run_latin_simulation(self, num_samples, technique, output_file= None):
        
        ##  Runs a LHS simulation: collects emissions results
        
        logging.info("Starting simulation with {} samples and technique {}".format(num_samples, technique))
        try:
            start_time = time.time()
            results = []
            total_emissions_results = []
            self.flow_rate_results = []
            self.temperature_results = []
            self.pressure_results = []
            self.efficiency_results = []
            self.electricity_conversion_results = []
            parameter_emissions_results = {}
            
            samples = self.generate_latin_samples(num_samples)
            
            for param, value in samples.items():
                parameter_emissions_results[param] = []

            for i in range(num_samples):
                config_sample = copy.deepcopy(self.config)
                for param, value in samples.items():
                    keys = param.split(".")
                    if len(keys) == 4:
                        if keys[1] == "output_streams":
                            module_name, output_streams_name, stream_index, sub_param_name = keys
                            stream_index = int(stream_index)
                            output_streams = config_sample[module_name][output_streams_name]
                            output_streams[stream_index][sub_param_name] = value[i]
                            if sub_param_name == "temperature":
                                self.temperature_results.append(value[i])
                            elif sub_param_name == "pressure":
                                self.pressure_results.append(value[i])
                    elif len(keys) == 3:
                        module_name, input_stream_name, sub_param_name = keys
                        input_stream = config_sample[module_name][input_stream_name]
                        if isinstance(input_stream, dict):
                            input_stream[sub_param_name] = value[i]
                        if sub_param_name == "flow_rate":
                            self.flow_rate_results.append(value[i])
                        elif sub_param_name == "temperature":
                            self.temperature_results.append(value[i])
                        elif sub_param_name == "pressure":
                            self.pressure_results.append(value[i])
                    elif len(keys) == 2:
                        module_name, param_name = keys
                        config_sample[module_name][param_name] = value[i]
                        if param_name == "efficiency":
                            self.efficiency_results.append(value[i])
                        elif param_name == "electricity_conversion":
                            self.electricity_conversion_results.append(value[i])
                
                # Build and simulate the network for the current sample
                network = Network(config_sample)
                network.build_network()
                network.simulate()
                
                # Record total and unit emission estimates
                subsystem_emissions, total_emissions = network.calculate_emissions()
                results.append(subsystem_emissions)
                if subsystem_emissions is not None:
                    results.append(subsystem_emissions)
                total_emissions_results.append(total_emissions)

                # Classify emissiones estimated based on parameter-type sampled
                for param, value in samples.items():
                    keys = param.split(".")
                    if len(keys) == 4:
                        if keys[1] == "output_streams":
                            module_name, output_streams_name, stream_index, sub_param_name = keys
                            stream_index = int(stream_index)
                            subsystem_name = f"{module_name}.{output_streams_name}"
                            if module_name in subsystem_emissions:
                                if sub_param_name == "temperature":
                                    parameter_emissions_results[param].append(subsystem_emissions[module_name])
                                elif sub_param_name == "pressure":
                                    parameter_emissions_results[param].append(subsystem_emissions[module_name])
                    elif len(keys) == 3:
                        module_name, input_stream_name, sub_param_name = keys
                        if module_name in subsystem_emissions:
                            if sub_param_name == "flow_rate":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
                            elif sub_param_name == "temperature":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
                            elif sub_param_name == "pressure":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
                    elif len(keys) == 2:
                        module_name, param_name = keys
                        if module_name in subsystem_emissions:
                            if param_name == "efficiency":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
                            elif param_name == "electricity_conversion":
                                parameter_emissions_results[param].append(subsystem_emissions[module_name])
            
            # Calculate parameter-type uncertainty in emission calculation based on stddev of all results
            parameter_uncertainties = {}
            for param, emissions in parameter_emissions_results.items():
                parameter_uncertainties[param] = np.std(emissions)

            end_time = time.time()
            logging.info("Simulation took {:.2f} seconds".format(end_time - start_time))

            # Plot the total system emission results if available
            if total_emissions_results:
                all_emissions = total_emissions_results
                mean_all_emissions = np.mean(total_emissions_results)
                std_all_emissions = np.std(total_emissions_results)

                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.hist(all_emissions, bins=100, density=True, alpha=0.6, label='All Emissions')
                lower_bound = np.percentile(all_emissions, 5)
                upper_bound = np.percentile(all_emissions, 95)
                ax.set_xlim(lower_bound, upper_bound)

                # Plot normal distribution PDF
                x = np.linspace(lower_bound, upper_bound, 100)
                pdf = norm.pdf(x, loc=mean_all_emissions, scale=std_all_emissions)
                ax.plot(x, pdf, 'k', linewidth=2, label='Normal Distribution')
                sns.kdeplot(all_emissions, fill=True, label='Kernel Density Estimate of Emissions', ax=ax)
                
                # Set chart labels, title and legend
                ax.set_title('Histogram of Total Emissions with Normal Distribution - {}'.format(technique), fontsize=16, fontweight='bold', family='Arial', pad=20)
                ax.set_xlabel('Emissions [kg CO2/kg NH3]', fontsize=14, fontweight='bold', family='Arial')
                ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold', family='Arial')
                ax.legend()
                
                plt.show()

            print("---------------")
            print(f"Technique {technique}")
            print("---------------")
            print("All Emissions - Simulation Results:")
            print("Mean: {:.2f}".format(mean_all_emissions))
            print("Standard Deviation: {:.2f}".format(std_all_emissions))
            print("---------------")

            # Write results to the output file if specified
            if output_file is not None:
                mode = 'w'
            else:
                mode = 'a'

            with open(output_file, mode, newline='') as f:
                writer = csv.writer(f)
                if mode == 'w':
                    writer.writerow(["Subsystem", "Emissions"])
                for result in results:
                    for subsystem, emissions in result.items():
                        writer.writerow([subsystem, emissions])        
            
            return results, parameter_uncertainties
        
        except Exception as e:
            logging.error("Error in simulation: {}".format(e))
        logging.info("Simulation completed")

    def plot_subsystems(self, technique, results):
        
        ##  Plots the emissions results for each subsystem
        ##  Comprehensive simulation-result visualisation results: Includes histograms, error bars, violin plots, and CDFs 

        if results:
            subsystem_emissions_mean = {}
            subsystem_emissions_std = {}

            for key in results[0].keys():
                values = [result[key] for result in results]
                subsystem_emissions_mean[key] = np.mean(values)
                subsystem_emissions_std[key] = np.std(values)

            # Iterate through each subsystem to generate the plots
            for subsystem_name in subsystem_emissions_mean.keys():
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))  # Adjusted to match the other function

                subsystem_emissions = [result[subsystem_name] for result in results]

                print(f"{subsystem_name} Subsystem Emissions - Simulation Results:")
                print("Mean: {:.2f}".format(np.mean(subsystem_emissions)))
                print("Standard Deviation: {:.2f}".format(np.std(subsystem_emissions)))
                print("---------------")

                if subsystem_emissions:
                    # Histogram
                    axs[0, 0].hist(subsystem_emissions, bins=100, density=True, alpha=0.6, label=subsystem_name)
                    axs[0, 0].set_xlim(np.min(subsystem_emissions), np.max(subsystem_emissions))

                    # PDF
                    x = np.linspace(np.min(subsystem_emissions), np.max(subsystem_emissions), 100)
                    pdf = norm.pdf(x, loc=np.mean(subsystem_emissions), scale=np.std(subsystem_emissions))
                    axs[0, 0].plot(x, pdf, 'k', linewidth=2, label='Normal Distribution')
                    sns.kdeplot(subsystem_emissions, fill=True, label='Kernel Density Estimate of Emissions', ax=axs[0, 0])

                    axs[0, 0].set_title('Emissions Histogram - Normal Distribution', fontsize=10, fontweight='bold', family='Arial')
                    axs[0, 0].set_xlabel('Emissions [kg CO2/kg NH3]', fontsize=10, fontweight='bold', family='Arial')
                    axs[0, 0].set_ylabel('Probability Density', fontsize=10, fontweight='bold', family='Arial')
                    axs[0, 0].legend()

                    # Error Bars
                    mean_emissions = np.mean(subsystem_emissions)
                    std_emissions = np.std(subsystem_emissions)
                    axs[0, 1].errorbar(subsystem_name, mean_emissions, yerr=std_emissions, fmt='o', capsize=5, label=subsystem_name)
                    axs[0, 1].fill_between([subsystem_name], mean_emissions - 2*std_emissions, mean_emissions + 2*std_emissions, color='gray', alpha=0.2, label='95% Confidence Interval')
                    axs[0, 1].set_xlim(-0.5, 0.5)
                    axs[0, 1].set_xticks([0])
                    axs[0, 1].set_xticklabels([subsystem_name])
                    axs[0, 1].set_xlabel('Subsystem', fontsize=10, fontweight='bold', family='Arial')
                    axs[0, 1].set_ylabel('Emissions [kg CO2/kg NH3]', fontsize=10, fontweight='bold', family='Arial')
                    axs[0, 1].set_title('Emissions Error Bars', fontsize=10, fontweight='bold', family='Arial')
                    axs[0, 1].legend()

                    # Violin plot
                    axs[1, 0].violinplot([subsystem_emissions], showmeans=True, showmedians=True)
                    axs[1, 0].set_title(f'Violin Plot', fontsize=10, fontweight='bold', family='Arial')
                    axs[1, 0].set_xlabel('Probability Density', fontsize=10, fontweight='bold', family='Arial')
                    axs[1, 0].set_ylabel('Emissions [kg CO2/kg NH3]', fontsize=10, fontweight='bold', family='Arial')

                    # CDF
                    axs[1, 1].plot(np.sort(subsystem_emissions), np.linspace(0, 1, len(subsystem_emissions), endpoint=False))
                    axs[1, 1].set_title(f'CPD Plot', fontsize=10, fontweight='bold', family='Arial')
                    axs[1, 1].set_xlabel('Emissions [kg CO2/kg NH3]', fontsize=10, fontweight='bold', family='Arial')
                    axs[1, 1].set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold', family='Arial')
                    
                    fig.suptitle(f'{subsystem_name} Subsystem Analysis - {technique}', fontsize=14, fontweight='bold')
                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
                    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2.5)
                    plt.show()

        else:
            print("No valid emissions results found.")

    def get_param_bounds(self):

        ##  Returns parameter bounds based on the assigned distributions
        ##  Assumes all have 'ppf' (percent-point function)
        
        param_bounds = {}
        for param, distribution in self.param_distributions.items():
            lower_bound = distribution.ppf(0)
            upper_bound = distribution.ppf(1)
            param_bounds[param] = (lower_bound, upper_bound)
        return param_bounds

    def sensitivity_sobol(self, total_emissions_results):
        
        ##  Performs Sobol SA on total emissions results

        param_bounds_dict = self.get_param_bounds()
        relevant_params = list(self.param_distributions.keys())

        # Problem definition for Sobol analysis using the filtered parameters
        problem = {
            'num_vars': len(relevant_params),
            'names': relevant_params,
            'bounds': [param_bounds_dict[param] for param in relevant_params]
        }
        num_saltelli_samples = len(total_emissions_results) // (2 * len(relevant_params))
        param_values = saltelli.sample(problem, num_saltelli_samples, calc_second_order=True)

        # Evaluate the model at each sample point
        total_emissions_results = []
        for i in range(param_values.shape[0]):
            config_sample = copy.deepcopy(self.config)
            for j, param in enumerate(relevant_params):
                keys = param.split(".")
                if len(keys) == 4:
                    if keys[1] == "output_streams":
                        module_name, output_streams_name, stream_index, sub_param_name = keys
                        stream_index = int(stream_index)
                        output_streams = config_sample[module_name][output_streams_name]
                        output_streams[stream_index][sub_param_name] = param_values[i, j]
                elif len(keys) == 3:
                    module_name, input_stream_name, sub_param_name = keys
                    input_stream = config_sample[module_name][input_stream_name]
                    if isinstance(input_stream, dict):
                        input_stream[sub_param_name] = param_values[i, j]
                elif len(keys) == 2:
                    module_name, param_name = keys
                    config_sample[module_name][param_name] = param_values[i, j]

            # Build and simulate network for sample
            network = Network(config_sample)
            network.build_network()
            network.simulate()
            
            # Record all emission results
            subsystem_emissions, total_emissions = network.calculate_emissions()
            total_emissions_results.append(total_emissions)

        # Perform the SA
        Si = sobol.analyze(problem, np.array(total_emissions_results), conf_level=0.95, print_to_console=False)
        
        # Plot SA results
        fig, ax = plt.subplots(figsize=(12, 8))  # Slightly taller figure for better spacing
        sns.barplot(x=Si['S1'], y=relevant_params, ax=ax, color='#FF7F7F', edgecolor='#8B0000', linewidth=1.2)
        
        # Set chart labels and title
        ax.set_title('Sobol Sensitivity Indices', fontsize=16, fontweight='bold', family='Arial', pad=20)
        ax.set_xlabel('Sensitivity Index', fontsize=14, fontweight='bold', family='Arial', labelpad=15)
        ax.set_ylabel('Parameter', fontsize=14, fontweight='bold', family='Arial', labelpad=15)

        # Adjust tick and grid parameters
        ax.tick_params(axis='x', labelsize=12, pad=5)
        ax.tick_params(axis='y', labelsize=12, pad=5)
        ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.grid(True, which='both', axis='y', linestyle=':', linewidth=0.5, alpha=0.7)

        # Set X-axis limits
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim(0, 1)

        # Style spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        plt.tight_layout(pad=3)
        plt.show()
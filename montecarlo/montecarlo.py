from network import Network
from distribution import DistributionAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, gamma
import seaborn as sns
import csv
import logging
import time
import scipy.stats as stats

class MonteCarloSampler:
    def __init__(self, config):
        self.config = config
        self.input_params, self.output_params = self.identify_input_params()
        self.param_distributions = self.assign_probability_distributions()

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
                    if isinstance(data, list) or isinstance(data, tuple):
                        if len(data) == 1:
                            param_distributions[param] = stats.uniform(loc=data[0] - 10, scale=20)
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
                                param_distributions[param] = stats.uniform(loc=data[0] - 10, scale=20)
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
                module_name, param_name = parts
                if param_name == "efficiency":
                    data = self.config[module_name][param_name]
                    param_distributions[param] = stats.uniform(loc=data*0.9, scale=0.9 - data*0.9)
                elif param_name == "electricity_conversion":
                    data = self.config[module_name][param_name]
                    param_distributions[param] = stats.uniform(loc=data*0.99, scale=data*1.01 - data*0.99)
        return param_distributions  

    def generate_samples(self, num_samples):
        samples = {}
        for param, distribution in self.param_distributions.items():
            samples[param] = [distribution.rvs() for _ in range(num_samples)]
        return samples
    
    def run_simulation(self, num_samples, technique, output_file=None):
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
                      
            samples = self.generate_samples(num_samples)
            
            for param, value in samples.items():
                parameter_emissions_results[param] = []
            
            for i in range(num_samples):
                config_sample = self.config.copy()
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
                network = Network(config_sample)
                network.build_network()
                network.simulate()
                subsystem_emissions, total_emissions = network.calculate_emissions()
                results.append(subsystem_emissions)
                
                if subsystem_emissions is not None:
                    results.append(subsystem_emissions)
                total_emissions_results.append(total_emissions)
                
                for param, value in samples.items():
                    keys = param.split(".")
                    #print(f"Param: {param}, Keys: {keys}")
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
            
            parameter_uncertainties = {}

            for param, emissions in parameter_emissions_results.items():
                parameter_uncertainties[param] = np.std(emissions)

            if total_emissions_results:
                all_emissions = total_emissions_results
                mean_all_emissions = np.mean(total_emissions_results)
                std_all_emissions = np.std(total_emissions_results)
                
                sns.set_style('whitegrid')
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.hist(all_emissions, bins=100, density=True, alpha=0.6, label='All Emissions')
                ax.set_xlim(np.min(all_emissions), np.max(all_emissions))

                # PDF
                x = np.linspace(np.min(all_emissions), np.max(all_emissions), 100)
                pdf = norm.pdf(x, loc=mean_all_emissions, scale=std_all_emissions)
                ax.plot(x, pdf, 'k', linewidth=2, label='Normal Distribution')
                sns.kdeplot(all_emissions, fill=True, label='Kernel Density Estimate of Emissions', ax=ax)

                ax.set_title('Histogram of Total Emissions with Normal Distribution')
                ax.set_xlabel('Emissions (kg CO2)')
                ax.set_ylabel('Probability Density')
                ax.legend()
                plt.show()
            
            end_time = time.time()
            logging.info("Simulation took {:.2f} seconds".format(end_time - start_time))

            print("---------------")
            print(f"Technique {technique}")
            print("---------------")
            print("All Emissions - Simulation Results:")
            print("Mean: {:.2f}".format(mean_all_emissions))
            print("Standard Deviation: {:.2f}".format(std_all_emissions))
            print("---------------")

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
            return None, None
        logging.info("Simulation completed")
        
        
    def plot_subsystems(self, results):
        if results:
            subsystem_emissions_mean = {}
            subsystem_emissions_std = {}

            for key in results[0].keys():
                values = [result[key] for result in results]
                subsystem_emissions_mean[key] = np.mean(values)
                subsystem_emissions_std[key] = np.std(values)

            # Histogram and error-bars for each subsystem
            for subsystem_name in subsystem_emissions_mean.keys():
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

                subsystem_emissions = [result[subsystem_name] for result in results]

                print(f"{subsystem_name} Subsystem Emissions - Simulation Results:")
                print("Mean: {:.2f}".format(np.mean(subsystem_emissions)))
                print("Standard Deviation: {:.2f}".format(np.std(subsystem_emissions)))
                print("---------------")

                if subsystem_emissions:
                    # Histograms
                    axs[0,0].hist(subsystem_emissions, bins=100, density=True, alpha=0.6, label=subsystem_name, range=(0, subsystem_emissions_mean[subsystem_name]*2))
                    axs[0,0].set_xlim(np.min(subsystem_emissions), np.max(subsystem_emissions))

                    # PDF
                    x = np.linspace(np.min(subsystem_emissions), np.max(subsystem_emissions), 100)
                    pdf = norm.pdf(x, loc=np.mean(subsystem_emissions), scale=np.std(subsystem_emissions))
                    axs[0,0].plot(x, pdf, 'k', linewidth=2, label='Normal Distribution')
                    sns.kdeplot(subsystem_emissions, fill=True, label='Kernel Density Estimate of Emissions', ax=axs[0,0])

                    axs[0,0].set_title('Histogram of Emissions with Normal Distribution')
                    axs[0,0].set_xlabel('Emissions (kg CO2)')
                    axs[0,0].set_ylabel('Probability Density')
                    axs[0,0].legend()

                    # ErrorBars
                    mean_emissions = np.mean(subsystem_emissions)
                    std_emissions = np.std(subsystem_emissions)
                    axs[0,1].errorbar(subsystem_name, mean_emissions, yerr=std_emissions, fmt='o', capsize=5, label=subsystem_name)
                    axs[0,1].fill_between([subsystem_name], mean_emissions - 2*std_emissions, mean_emissions + 2*std_emissions, color='gray', alpha=0.2, label='95% Confidence Interval')
                    axs[0,1].set_xlim(-0.5, 0.5)
                    axs[0,1].set_xticks([0])
                    axs[0,1].set_xticklabels([subsystem_name])
                    axs[0,1].set_xlabel('Subsystem')
                    axs[0,1].set_ylabel('Emissions (kg CO2)')
                    axs[0,1].set_title('Error Bars for Subsystem Emissions')
                    axs[0,1].legend()

                    # Violin plot
                    axs[1,0].violinplot([subsystem_emissions], showmeans=True, showmedians=True)
                    axs[1,0].set_title(f'Violin Plot - {subsystem_name}')
                    axs[1,0].set_xlabel('Probability Density')
                    axs[1,0].set_ylabel('Emissions (kg CO2)')

                    # CDF
                    axs[1,1].plot(np.sort(subsystem_emissions), np.linspace(0, 1, len(subsystem_emissions), endpoint=False))
                    axs[1,1].set_title(f'CPD Plot - {subsystem_name}')
                    axs[1,1].set_xlabel('Emissions (kg CO2)')
                    axs[1,1].set_ylabel('Cumulative Probability')

                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
                    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=2.5)
                    plt.show()
        else:
            print("No valid emissions results found.") 

    def visualize_emission_uncertainty(self, mc_uncertainties):
        mc_uncertainties = {k: v for k, v in mc_uncertainties.items() if not np.isnan(v)}
        subsystem_order = ['ASU', 'HydrogenProduction', 'SyngasCompression', 'HaberBoschUnit', 'CoolingSubsystem']
        parameter_type_order = ['input_stream', 'output_streams', 'efficiency', 'electricity_conversion']
        parameter_labels = sorted(mc_uncertainties.keys(), key=lambda x: (subsystem_order.index(x.split('.')[0]), parameter_type_order.index(x.split('.')[1]) if len(x.split('.')) > 2 else float('inf')))
        uncertainties = [mc_uncertainties[label] for label in parameter_labels]

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
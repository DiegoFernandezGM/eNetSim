import json
from registry import SubsystemRegistry
from flows import Flows
from stream import Stream 
from energy import calculate_energy_consumption

class Network:
    def __init__(self, config):
        if isinstance(config, str):  # string
            with open(config, 'r') as f:
                self.config_data = json.load(f)
        elif isinstance(config, dict):  # dictionary
            self.config_data = config
        else:
            raise ValueError("Config file must be a string (file path) or a dictionary")

        self.subsystems = {}
        self.streams = {}
        self.subsystem_registry = SubsystemRegistry()
        self._register_subsystems()

    def _register_subsystems(self):
        for subsystem_name, subsystem_config in self.config_data.items():
            if subsystem_name not in ["electricity_emission_factor", "iterations"]:
                subsystem_type = subsystem_config['type']
                class_name = subsystem_type
                module_name = 'dynamic_modules'  # dummy module name
                # Create a dynamic class for the subsystem
                def __init__(self, config):
                    self.config = config
                    self.name = subsystem_name
                    self.energy_consumption = config.get('energy_consumption', 0)
                    self.next_subsystem = config.get('next_subsystem', None)
                subsystem_class = type(class_name, (object,), {'__init__': __init__})
                # Register the subsystem class
                self.subsystem_registry.register_subsystem_class(subsystem_type, subsystem_class)

    def build_network(self):
        for subsystem_name, subsystem_config in self.config_data.items():
            if subsystem_name not in ["electricity_emission_factor", "iterations"]:
                subsystem = self.subsystem_registry.create_subsystem(subsystem_config)
                self.subsystems[subsystem_name] = subsystem
                self.create_streams(subsystem, subsystem_config, subsystem_name, iteration=1)

    def create_streams(self, subsystem, subsystem_config, subsystem_name, iteration):
        input_streams = self.create_input_streams(subsystem_config, subsystem_name, iteration)
        subsystem.inputs = input_streams
        output_streams = self.create_output_streams(input_streams, subsystem_config)
        subsystem.outputs = output_streams
    
    def create_input_streams(self, subsystem_config, subsystem_name, iteration):
        input_streams = []
        if "input_stream" in subsystem_config:
            input_stream_config = subsystem_config["input_stream"]
            flows = Flows()
            input_stream = flows.create_input(input_stream_config, self.streams)
            input_streams.append(input_stream)
        elif "input_streams" in subsystem_config:
            flows = Flows()
            for input_stream_config in subsystem_config["input_streams"]:
                input_stream = flows.create_inputs(input_stream_config, self.streams, self.subsystems, subsystem_name, iteration)
                if input_stream is not None:
                    input_streams.append(input_stream)
        return input_streams
    
    def create_output_streams(self, input_streams, subsystem_config):
        output_streams = []
        flows = Flows()
        stream = Stream()
        for output_stream_config in subsystem_config["output_streams"]:
            if "mix_ratio" in output_stream_config:
                output_stream = stream.mix_streams(input_streams, output_stream_config)
            else:
                output_stream = flows.create_outputs(input_streams, output_stream_config, self.streams)
            output_streams.append(output_stream)
        return output_streams

    def simulate(self):
        iterations = self.config_data.get('iterations', 1)
        for iteration in range(iterations):
            #print(f"\nIteration {iteration + 1}:")
            for subsystem_name, subsystem in self.subsystems.items():
                #print(f"\nProcessing Subsystem: {subsystem_name}")
                #print("Input Streams:")
                #for input_stream in subsystem.inputs:
                    #input_stream.print_info()
                #print("Output Streams:")
                #for output_stream in subsystem.outputs:
                    #output_stream.print_info()
                self.create_streams(subsystem, self.config_data[subsystem_name], subsystem_name, iteration + 1)
    
    def calculate_energy_consumption(self):
        iterations = self.config_data.get('iterations', 1)
        energy_consumption_results = {}  # Dictionary of results
        for iteration in range(iterations):
            #print(f"\nIteration {iteration + 1}:")
            iteration_results = {}
            for subsystem_name, subsystem in self.subsystems.items():
                calculate_energy_consumption(self, iteration+1)
                energy_consumption = subsystem.energy_consumption
                iteration_results[subsystem_name] = energy_consumption
                #print(f"Energy consumption for {subsystem_name}: {round(energy_consumption, 2)}")
            
            for subsystem_name, energy_consumption in iteration_results.items():
                if subsystem_name not in energy_consumption_results:
                    energy_consumption_results[subsystem_name] = [energy_consumption]
                else:
                    energy_consumption_results[subsystem_name].append(energy_consumption)
        
        return energy_consumption_results
    
    def get_emission_factor(self):
        return self.config_data['electricity_emission_factor']

    def calculate_emissions(self):
        energy_consumption_results = self.calculate_energy_consumption()
        total_emissions = 0
        subsystem_emissions = {}
        #print(f"\n")
        for subsystem_name, energy_consumptions in energy_consumption_results.items():
            subsystem_emissions_tmp = 0
            for energy_consumption in energy_consumptions:
                emission_factor = self.get_emission_factor()
                subsystem_emissions_tmp += energy_consumption * emission_factor
                subsystem_emissions_tmp = round(subsystem_emissions_tmp,2)
                subsystem_emissions[subsystem_name] = subsystem_emissions_tmp

            print(f"{subsystem_name}: {subsystem_emissions[subsystem_name]} kg CO2")
            total_emissions += subsystem_emissions_tmp
        print(f"\nTotal CO2 emissions: {round(total_emissions, 2)} kg CO2")
        return subsystem_emissions, total_emissions

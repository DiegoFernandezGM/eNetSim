from importer import *

from registry import SubsystemRegistry
from flows import Flows
from stream import Stream 
from energy import calculate_energy_consumption

class Network:
    def __init__(self, config):

        ##  Initialise network with a given configuration
        ##  This config can be provided as a file path (string) or directly as dictionary

        if isinstance(config, str):  # Load config from file path string
            with open(config, 'r') as f:
                self.config_data = json.load(f)
        elif isinstance(config, dict):  # Use provided dictionary as config
            self.config_data = config
        else:
            raise ValueError("Config file must be a string (file path) or a dictionary")

        # Ditionaries to store subsystems and streams
        self.subsystems = {}
        self.streams = {}

        # Register all subsystems to manage the classes
        self.subsystem_registry = SubsystemRegistry()
        self._register_subsystems()

    def _register_subsystems(self):

        ##  Dynamically register subsystem classes based on the config
        ##  Allows flexibility in defining new units without modifying core code

        for subsystem_name, subsystem_config in self.config_data.items():
            if subsystem_name not in ["electricity_emission_factor", "iterations"]:
                subsystem_type = subsystem_config['type']
                class_name = subsystem_type
                module_name = 'dynamic_modules'  # dummy module name
                
                # Dynamically defines class for the subsystem
                def __init__(self, config):
                    self.config = config
                    self.name = subsystem_name
                    self.energy_consumption = config.get('energy_consumption', 0)
                    self.next_subsystem = config.get('next_subsystem', None)
                
                # Create the subsystem class dynamically
                subsystem_class = type(class_name, (object,), {'__init__': __init__})
                
                # Register the subsystem class
                self.subsystem_registry.register_subsystem_class(subsystem_type, subsystem_class)

    def build_network(self):

        ##  Build network by instantiating all units defined in the config
        ##  Initialise their in/output streams

        for subsystem_name, subsystem_config in self.config_data.items():
            if subsystem_name not in ["electricity_emission_factor", "iterations"]:
                subsystem = self.subsystem_registry.create_subsystem(subsystem_config)
                self.subsystems[subsystem_name] = subsystem
                self.create_streams(subsystem, subsystem_config, subsystem_name, iteration=1)

    def create_streams(self, subsystem, subsystem_config, subsystem_name, iteration):
        
        ##  Create in/output streams for a given subsystem
        ##  Streams necessary for data exchange between units

        input_streams = self.create_input_streams(subsystem_config, subsystem_name, iteration)
        subsystem.inputs = input_streams
        output_streams = self.create_output_streams(input_streams, subsystem_config)
        subsystem.outputs = output_streams
    
    def create_input_streams(self, subsystem_config, subsystem_name, iteration):
        
        ##  Create input streams for a unit based on its config
        ##  Supports both single and multiple input streams

        input_streams = []
        if "input_stream" in subsystem_config: # Single input stream
            input_stream_config = subsystem_config["input_stream"]
            flows = Flows()
            input_stream = flows.create_input(input_stream_config, self.streams)
            input_streams.append(input_stream)
        elif "input_streams" in subsystem_config: # Multiple input stream
            flows = Flows()
            for input_stream_config in subsystem_config["input_streams"]:
                input_stream = flows.create_inputs(input_stream_config, self.streams, self.subsystems, subsystem_name, iteration)
                if input_stream is not None:
                    input_streams.append(input_stream)
        
        return input_streams
    
    def create_output_streams(self, input_streams, subsystem_config):
        
        ##  Create output streams for a subsystem
        ##  Supports stream mixing if required by the configuration

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
        
        ##  Simulate network over the defined nº of iterations
        ##  During each iteration: streams are updated and final ammonia flow is calculated

        iterations = self.config_data.get('iterations', 1)
        ammonia_flow = 0
        for iteration in range(iterations):
            for subsystem_name, subsystem in self.subsystems.items():
                self.create_streams(subsystem, self.config_data[subsystem_name], subsystem_name, iteration + 1)
                for output_stream_config in self.config_data[subsystem_name]["output_streams"]:
                    if 'final' in output_stream_config and output_stream_config['final'] is True:
                        for stream in subsystem.outputs:
                            ammonia_flow = stream.get_property('massflow')
        return ammonia_flow
    
    def calculate_energy_consumption(self):

        ##  Calculate energy consumption for the entire network over defined iterations

        iterations = self.config_data.get('iterations', 1)
        energy_consumption_results = {}  # Dictionary of results
        for iteration in range(iterations):
            iteration_results = {}
            for subsystem_name, subsystem in self.subsystems.items():
                calculate_energy_consumption(self, iteration+1)
                energy_consumption = subsystem.energy_consumption
                iteration_results[subsystem_name] = energy_consumption
            
            # Store the results for each subsystem
            for subsystem_name, energy_consumption in iteration_results.items():
                if subsystem_name not in energy_consumption_results:
                    energy_consumption_results[subsystem_name] = [energy_consumption]
                else:
                    energy_consumption_results[subsystem_name].append(energy_consumption)
        
        return energy_consumption_results
    
    def get_emission_factor(self):

        ##  Retrieve emission factor from config
        ##  Factor used to calculate CO2 emissions

        return self.config_data['electricity_emission_factor']

    def calculate_emissions(self):
        
        ##  Calculate CO2 emissions based on energy consumption and emission factor
        ##  Computes emissions per unit of ammonia produced

        energy_consumption_results = self.calculate_energy_consumption()
        total_emission = 0
        subsystem_emission = {}
        subsystem_emissions = {}

        iteration = self.config_data.get('iterations', 1)
        ammonia_flow = self.simulate()

        for subsystem_name, energy_consumptions in energy_consumption_results.items():
            energy_consumption = energy_consumptions[iteration-1]
            emission_factor = self.get_emission_factor()
            subsystem_emission[subsystem_name] = energy_consumption * emission_factor
            total_emission += subsystem_emission[subsystem_name]
            
            if ammonia_flow > 0:    # Calculate emissions per unit of ammonia produced                  
                subsystem_emissions[subsystem_name] = subsystem_emission[subsystem_name]/(ammonia_flow)
                total_emissions = total_emission/(ammonia_flow)

        return subsystem_emissions, total_emissions

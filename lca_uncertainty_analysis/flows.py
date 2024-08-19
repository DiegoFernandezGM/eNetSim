from stream import Stream

class Flows:

    def create_inputs(self, config, streams, subsystems, subsystem_name, iteration):
        
        
        ##  Creates an input stream for a subsystem based on the provided configuration
        ##  The stream can be created from: the output of another unit, a previous iteration or a direct configuration

        input_stream = None
        
        # Create input stream from another subsystem's output
        if "from_subsystem" in config:
            from_subsystem_name = config["from_subsystem"]
            if from_subsystem_name in subsystems:
                from_subsystem = subsystems[from_subsystem_name]
                if hasattr(from_subsystem, 'outputs'):
                    output_streams = from_subsystem.outputs
                    fluid_name = config["fluid_name"]
                    output_streams = [s for s in output_streams if s.get_property('fluid') == fluid_name]
                    if output_streams:
                        # Skip this input stream if the iteration is less than the specified iteration
                        if "iteration" in config and config["iteration"] > iteration:
                            return None
                        for output_stream in output_streams:
                            input_stream = Stream(**output_stream.get_all_properties())
                            streams[fluid_name] = input_stream
                    else:
                        print(f"Error: No output stream with fluid {fluid_name} found in subsystem {from_subsystem_name}.")
                else:
                    print(f"Error: Subsystem {from_subsystem_name} has no outputs.")
        
        # Create input stream from previous iteration
        elif "iteration" in config and config["iteration"] == iteration:
            input_stream = streams[config["fluid_name"]]
        
        # Create new input stream based on the configuration
        else:
            input_stream = self.create_input(config, streams)
            streams[config['fluid_name']] = input_stream
        
        return(input_stream)

    def create_input(self, config, streams):
        
        ##  Creates new input stream directly from the provided configuration
        
        input_stream = None
        input_args = {
            'fluid_name': config["fluid_name"],
            'flow_rate': config["flow_rate"],
            'temperature': config["temperature"],
            'pressure': config["pressure"],
            'mass_flow': config["mass_flow"],
            'enthalpy': config["enthalpy"],
            'internal_energy': config["internal_energy"],
            'entropy': config["entropy"],
            'density': config["density"],
            'gamma': config["gamma"],
            'phase': config["phase"]
        }
        
        # Create input stream using the above provided arguments
        input_stream = Stream(**input_args)
        input_stream.calculate_all_properties()
        streams[config['fluid_name']] = input_stream
        
        return(input_stream)

    def create_outputs(self, input_streams, output_stream_config, streams):
        
        ##  Creates an output stream for a unit based on the configuration and input streams

        output_stream = None
       
       # Identify the correct input stream based on the fluid name and configuration
        if 'origin' in output_stream_config and output_stream_config['origin'] is True:
            input_stream = input_streams[0]
        else:
            input_stream = None
            for stream in input_streams:
                if stream.get_property('fluid') == output_stream_config["fluid_name"]:
                    input_stream = stream
                    break
           
        if input_stream is None:
            raise ValueError(f"No matching input stream found for fluid: {output_stream_config['fluid_name']}")       

        # Define output stream properties (use defaults from input stream if not specified)
        output_stream_args = {
            "fluid_name": output_stream_config["fluid_name"],
            "temperature": output_stream_config.get("temperature", input_stream.get_property("T")),
            "pressure": output_stream_config.get("pressure", input_stream.get_property("P")),
            "flow_rate": output_stream_config.get("flow_rate", input_stream.get_property("massflow")),
        }       

        # Adjust flow rate if split ratio is provided
        if "split_ratio" in output_stream_config:
            output_stream_args["flow_rate"] = input_stream.get_property("massflow") * output_stream_config["split_ratio"]

        # Include recycled fraction if specified in configuration
        if "recycled_fraction" in output_stream_config:
            output_stream_args["recycled_fraction"] = output_stream_config["recycled_fraction"]
        
        # Create output stream and calculate all properties
        output_stream = Stream(**output_stream_args)
        output_stream.calculate_all_properties()
        streams[output_stream_config['fluid_name']] = output_stream
        
        return(output_stream)

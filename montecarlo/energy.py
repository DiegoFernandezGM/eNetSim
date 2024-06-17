def calculate_energy_consumption(network, iteration):
    for subsystem_name, subsystem in network.subsystems.items():
        if iteration == 1 and 'first_energy_consumption' in subsystem.config:
            energy_consumption_formula = subsystem.config['first_energy_consumption']
        else:
            energy_consumption_formula = subsystem.config['energy_consumption']
        
        efficiency = subsystem.config['efficiency']
        electricity_conversion = subsystem.config['electricity_conversion']

        input_streams = subsystem.inputs
        output_streams = subsystem.outputs
        
        inputs = {}
        #Gather stream properties
        for i, stream in enumerate(input_streams):
            inputs[f'input_streams_{i}'] = stream.get_all_properties()
            #print(f"{inputs[f'input_streams_{i}']}")
        for i, stream in enumerate(output_streams):
            inputs[f'output_streams_{i}'] = stream.get_all_properties()
            #print(f"{inputs[f'output_streams_{i}']}")
        
        # Add to inputs dictionary
        inputs['input_streams'] = input_streams
        inputs['output_streams'] = output_streams
        inputs['iteration'] = iteration
        inputs['efficiency'] = efficiency
        inputs['electricity_conversion'] = electricity_conversion

        # Energy consumption formulas in Json File
        try:
            energy_consumption = eval(energy_consumption_formula, {"inputs": inputs}, inputs)
        except Exception as e:
            raise ValueError(f"Error evaluating energy consumption formula for subsystem {subsystem_name}: {e}")

        subsystem.energy_consumption = energy_consumption
        #print(f"Energy consumption for {subsystem_name}:  {round(energy_consumption, 2)}")
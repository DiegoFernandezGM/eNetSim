def calculate_energy_consumption(network, iteration):
    
    ##  Calculates energy consumption for each network subsystem based on the specified formulas in the configuration. 
    ## The energy consumption formula used can vary depending on the unit and the iteration

    for subsystem_name, subsystem in network.subsystems.items():

        # Choose the energy consumption formula for the Haber Bosch Unit based on the iteration (due to recycling)

        if iteration == 1 and 'first_energy_consumption' in subsystem.config:
            energy_consumption_formula = subsystem.config['first_energy_consumption']
        else:
            energy_consumption_formula = subsystem.config['energy_consumption']
        
        # Retrieve configuration parameters
        efficiency = subsystem.config['efficiency']
        electricity_conversion = subsystem.config['electricity_conversion']
        
        #Collect in/output streams from the subsystem
        input_streams = subsystem.inputs
        output_streams = subsystem.outputs
        
        inputs = {}
        
        # Store all properties of input streams in inputs dictionary
        for i, stream in enumerate(input_streams):
            inputs[f'input_streams_{i}'] = stream.get_all_properties()
            #print(f"{inputs[f'input_streams_{i}']}")
         
        # Store all properties of output streams in inputs dictionary
        for i, stream in enumerate(output_streams):
            inputs[f'output_streams_{i}'] = stream.get_all_properties()
            #print(f"{inputs[f'output_streams_{i}']}")
        
        # Add restant parameters to the inputs dictionary
        inputs['input_streams'] = input_streams
        inputs['output_streams'] = output_streams
        inputs['iteration'] = iteration
        inputs['efficiency'] = efficiency
        inputs['electricity_conversion'] = electricity_conversion
        
        try:
            # Evaluate config energy consumption formula using the inputs dictionary
            energy_consumption = eval(energy_consumption_formula, {"inputs": inputs}, inputs)
        except Exception as e:
            # Raise error message if formula evaluation fails
            raise ValueError(f"Error evaluating energy consumption formula for subsystem {subsystem_name}: {e}")

        # Store calculated energy consumption in subsystem object
        subsystem.energy_consumption = energy_consumption
        
        # Debugging line to check the calculated energy consumption
        # print(f"Energy consumption for {subsystem_name}:  {round(energy_consumption, 2)}")
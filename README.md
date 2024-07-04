
# LCA Uncertainty code

Created by:

**Diego Fernandez**


### What does the code do?

The code simulates a network of subsystems to calculate energy consumption and emissions. It also performs Monte Carlo, Quasi-Monte Carlo, and Latin Hypercube simulations to analyze uncertainties and sensitivities within the network.

Python script for a simulation that models a network of subsystems with input and output streams. The subsystems consume energy and emit CO2 emissions. The script uses several classes and functions to define the subsystems, streams, and energy consumption, and to perform Monte Carlo,Quasi-Monte Carlo and Latin Hypercube simulations, and compare between techniques.

## Inputs:
The inputs required are:
- `config.json`: Configuration file detailing the subsystems and their properties.
- Subsystem modules and classes should be correctly defined and available for dynamic import.

## Outputs:
The code outputs are:
- Energy consumption of each subsystem.
- CO₂ emissions of each subsystem and the total network.
- Results of Monte Carlo, Quasi-Monte Carlo, and Latin Hypercube simulations in CSV files: MonteCarloResults.csv, QuasiMCResults.csv, LatinHResults.csv
- simulation.log: A log file containing details of the simulation runs.
- Statistical analysis and plots comparing the subsystems.


## Units Used in Software
Stream Properties:
-temperature: Kelvin (K)
-pressure: Pascals (Pa)
-flow_rate: Mass flow rate in kg/s
-mass_flow: Mass flow in kg
-enthalpy: J/kg
-internal_energy: J/kg
-entropy: J/(kg·K)
-density: kg/m³
-gamma: Specific heat ratio (dimensionless)
-phase: Dimensionless indicator of phase (e.g., 0 for liquid, 1 for vapor)

Emissions Calculation:
-emission_factor: kg CO₂/kWh (SEAI statistics 2023: 0.2548 kgCO2/kWh)
-energy_consumption: kWh

### Getting the code

- Clone the repository using the following command (in the CLI/Terminal)
```bash
$ git clone  git@github.com:DiegoFernandezGM/LCA_Uncertainty_Python.git
```

- Or use the GitHub CLI
    
```
$ gh repo clone DiegoFernandezGM/LCA_Uncertainty_Python
```

- Or download a version from [here] (https://github.com/DiegoFernandezGM/LCA_Uncertainty_Python/releases/new)

### Running the code

To run the code, from Command Line Interface (CLI)

```
$ python main.py
```

To run the code in verbose mode:

```
$ python main.py -v```
```

Or using Visual Studio Code type **Run**

To pipe results to a file  use ``` python main.py > results.txt ```

### Code Structure

The main function, main(), sets up the network by loading the configuration from a JSON file, building the network, and simulating it. It then performs Monte Carlo and Latin Hypercube simulations using the MonteCarloSampler and Sampler classes, and compares the statistical measures of the results using the Evaluator class.

The Network class defines the network of subsystems and streams. The SubsystemRegistry class is used to register and create subsystem classes dynamically. The Stream class defines the input and output streams, and the ThermoStream class is a subclass of Stream that calculates thermodynamic properties of the streams. The calculate_energy_consumption function calculates the energy consumption of each subsystem based on its configuration.

The MonteCarloSampler class performs Monte Carlo simulations by generating random samples of the input parameters and running the simulation for each sample. The Sampler class performs Latin Hypercube simulations using the SALib library. The Evaluator class compares the statistical measures of the results from the Monte Carlo and Latin Hypercube simulations.

**main.py**
The main entry point for the simulation. It sets up logging, parses command-line arguments, loads the configuration, builds the network, and runs simulations.

**network.py**
Defines the Network class, which builds and simulates the network of subsystems, calculates energy consumption, and emissions.

**flows.py**
Defines the Flows class, which handles the creation of input and output streams between subsystems.

**registry.py**
Defines the SubsystemRegistry class, which registers and creates subsystem instances dynamically.

**stream.py**
Defines the Stream class, which models the properties and behavior of a fluid stream in the network.

**thermostream.py**
Defines the ThermoStream class, which uses the CoolProp library to calculate thermodynamic properties of the fluid streams.


```python
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
```
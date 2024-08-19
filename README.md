
# Life Cycle Analysis Software with Uncertainty Analysis

Created by:

**Diego Fernandez**

## What does the code do?

The code provides a comprehensive framework for Uncertainty Analysis (UA) within Life Cycle Assessments (LCA) for industrial energy systems, particularly focused on energy consumption and emissions analysis. It allows users to model a network of subsystems, simulate their operations, and assess the uncertainty associated with specified input parameters through two distinct routes (Probabilisitc UA and/or Possibilistic UA), encompassing several statistical and analytical methods. Within these pathways, the framework can be classified into three primary methods for uncertainty quantification:

**1. Probabilistic-Sampling Uncertainty Analysis:**
Utilises Monte Carlo (MCS), Quasi-Monte Carlo (QMCS), and Latin Hypercube (LHS) sampling techniques to generate a large number of samples from the input distributions, simulate the system, and analyse the variability in the outputs (energy consumption and emissions). This approach also supports Bayesian inference to update the input distributions based on observed data. Furthermore, a Sensitivity Analysis (SA) implementing Sobol indices can be integrated with these sampling tools. 

**2. Probabilistic-Analytical Uncertainty Analysis:**
Employs a first-order Taylor series expansion to approximate how uncertainties in the input parameters propagate through the system and affect the outputs. This method is computationally efficient and provides insights into the sensitivity of outputs to various inputs.

**3. Possibilistic Uncertainty Analysis:** 
Applies fuzzy logic to model uncertainty when the data is imprecise or vague, as opposed to inherently random. This method uses fuzzy membership functions, alpha-cuts, and Z-numbers to propagate uncertainties through the system.

The code orchestrates the entire process, from loading configurations, simulating the network, performing various uncertainty analyses, and visualising the results. It is designed to be flexible and extendable, allowing users to modify configurations, add new subsystems, and experiment with different uncertainty analysis techniques.

## Inputs:
The inputs required are:

### 1. Configuration Files

- ***config.json***: This file contains the configuration of the entire energy system, including the definition of each subsystem, their input/output streams, operational parameters, energy consumption formulas, and the number of iterations for the simulation.

>> **Subsystem Parameters**: Each subsystem in the configuration file is defined with its type, input streams, output streams, and specific operational parameters like efficiency and electricity conversion factors.

>> **Energy Consumption Formulas**: The configuration also includes formulas to calculate energy consumption based on the inputs and outputs of each subsystem.

- ***uncertain_params.json***: This file defines the uncertain parameters in the system. It specifies the name of the parameter, the type of uncertainty (probabilistic or possibilistic), and the associated distributions or membership functions.

>> **Probabilistic Parameters**: Defined by distributions such as Gaussian, Uniform, Lognormal, etc.

>> **Possibilistic Parameters**: Defined by fuzzy membership functions, like Triangular, Trapezoidal, Sigmoidal, etc.

### 2. Command-Line Arguments

- **v or --verbose**: Optional flag to enable verbose mode, which provides detailed output during the execution of the code.

- ***config.json***: The path to the configuration file, if different from the default config.json.

- ***uncertain_params.json***: The path to the uncertain parameters file, if different from the default 'uncertain_params.json'.

### 3. Parameter Distributions

- **Monte Carlo and Related Techniques**: The input parameters for these techniques are sampled from specified probability distributions, defined within the ***assign_probability_distributions*** function of each respective technique class , if a unique input value is provided, or calculated using the ***distribution.py*** class, if a range of possible input values is provided.

- **Fuzzy Parameters**: For possibilistic analysis, the inputs are defined by membership functions representing fuzzy sets defined in ***uncertain_params.json***.

## Outputs:
The code outputs are:

### 1. Simulation Results

Emissions and Energy Consumption: The primary outputs of the code are the calculated energy consumption and CO2 emissions for each subsystem and the overall system. These are computed for each iteration or sample of the simulation and stored for further analysis.

### 2. Uncertainty Metrics

**Uncertainty Quantification**: The code calculates and outputs the uncertainty in energy consumption and emissions for each subsystem and the entire system. This includes:

- **Mean**: The average output across all simulations.

- **Standard Deviation**: A measure of the variability in the outputs.

- **Confidence Intervals**: For probabilistic methods, the code computes confidence intervals to represent the range within which the true value likely falls.

- **Sobol Sensitivity Indices**: For Quasi-Monte Carlo simulations, Sobol sensitivity indices are calculated to identify the most influential parameters.

### 3. Visualisations

- **Histograms**: Plots showing the distribution of the outputs (e.g., emissions) across all samples.

- **Error Bars**: Visualisation of the mean output with associated uncertainty (standard deviation).

- **Violin Plots**: A combination of boxplots and kernel density plots to show the distribution of the outputs.

- **Cumulative Distribution Functions (CDFs)**: Plots showing the cumulative probability distribution of the outputs.

### 4. Output Files
- **CSV Files**: The results of each simulation can be written to CSV files, which include detailed emissions data for each subsystem and overall system performance metrics.

- ***MonteCarloResults.csv***: Results from Monte Carlo simulations.

- ***QuasiMCResults.csv***: Results from Quasi-Monte Carlo simulations.

- ***LatinHResults.csv***: Results from Latin Hypercube simulations.

### 5. Logs
- ***simulation.log***: A log file capturing the progress of the simulation, including detailed information about the configurations, sampling techniques used, and any errors encountered during execution.

## Units Used in Software
The software uses the following units consistently across all calculations:

### Thermodynamic Properties
- **Temperature**: Kelvin (K)

- **Pressure**: Pascals (Pa)

- **Flow Rate**: kg/s (kilograms per second)

- **Mass Flow**: kg/s (kilograms per second)

- **Enthalpy**: J/kg (Joules per kilogram)

- **Internal Energy**: J/kg (Joules per kilogram)

- **Entropy**: J/(kg·K) (Joules per kilogram per Kelvin)

- **Density**: kg/m³ (kilograms per cubic meter)

### Energy Consumption
- **Energy**: Joules (J)

- **Electricity Conversion Factor**: J/s (Joules per second)

### Emissions
- **CO2 Emissions**: kg CO2 / kg NH3 (kilograms of CO2 per kilograms of NH3 produced)

### Miscellaneous
- **Efficiency**: Dimensionless (expressed as a fraction, e.g., 0.8 for 80% efficiency)

- **Gamma (Adiabatic Index)**: Dimensionless

- **Phase**: Dimensionless (e.g., 0 for liquid, 1 for vapor)

## Getting the code

- Clone the repository using the following command (in the CLI/Terminal)

```bash
$ git clone  git@github.com:DiegoFernandezGM/LCA_Uncertainty_Python.git
```

- Or use the GitHub CLI
    
```
$ gh repo clone DiegoFernandezGM/LCA_Uncertainty_Python
```

- Or download a version from [https://github.com/DiegoFernandezGM/LCA_Uncertainty_Python/releases/new]

## How to Run the Code
To run the uncertainty analysis, follow these steps:

**1. Install Dependencies**

Ensure that all required Python packages are installed:

```
$ pip install -r requirements.txt
```

**2. Prepare Configuration Files**

Ensure your system's configuration in the ***config.json*** and file and uncertain parameters in the ***uncertain_params.json*** file are correctly set up.

**3. Running the Code**

To run the code from Command Line Interface (CLI):

```
$ python main.py
```

**4. Running the Code in Verbose Mode**

To run the code with detailed output (verbose mode):

```
$ python main.py -v
```

**5. Running the Code in Visual Studio Code**

If using Visual Studio Code, simply open the project and type **Run** to execute the code.


**6. Redirecting Output to a File**

To pipe and save the results to a file, use:

```
$ python main.py > results.txt
```

**7. Specifying Custom Configuration Files**

Use custom configuration files if needed:

```
$ python main.py --config=path/to/your/config.json --uncertain_params=path/to/your/uncertain_params.json
```

**8. View Results**

Check the output CSV files and log files for detailed results. Use the built-in visualisation functions to analyse the outputs.


**9. Advanced Usage**

Run specific UA scripts directly:

- **Bayesian Analysis:**

```
$ python bayesian.py
```

- **Analytical Uncertainty Analysis:**

```
$ python analytical.py
```

- **Possibilistic Uncertainty Analysis:**

```
$ python possibilistic.py
```

**10. Analysing the Logs**

After running simulations, check simulation.log for detailed logs.

## Code Structure

The main function, ***main()***, serves as the entry point of the program and is responsible for setting up, simulating, and analysing a network of interconnected subsystems. It begins by setting up command-line argument parsing, allowing the user to run the code in verbose mode if desired. Depending on the mode selected, the program sets the logging level to either DEBUG (verbose mode) or INFO (fast mode), providing different levels of output detail.

The program then proceeds to load the network configuration from a JSON file (***config.json***). This configuration file defines the subsystems in the network, their input and output streams, and other relevant parameters such as efficiency and energy consumption formulas. The configuration is crucial as it dictates how the network is constructed and simulated.

With the configuration loaded, the program initialises the ***Network*** class. The ***Network*** class is designed to represent the entire network of subsystems, which are dynamically created based on the configuration. The class uses a ***SubsystemRegistry*** to manage and instantiate these subsystems, ensuring that the network can be easily extended or modified by simply updating the configuration file.

Once the network is built, the program simulates it over a specified number of iterations. During each iteration, the subsystems interact through their input and output streams, and the energy consumption and emissions are calculated for each subsystem. These calculations are performed using the ***calculate_energy_consumption*** function, which evaluates the energy consumption formulas defined in the configuration.

After the initial simulation, the program performs a series of UA techniques to assess the impact of uncertain parameters on the network's performance. The first technique applied is MCS, facilitated by the ***MonteCarloSampler*** class. This class generates random samples of the input parameters based on assigned probability distributions and runs the simulation for each sample. The results are then collected and analysed to understand the variability and uncertainty in the network's outputs.

In addition to MCS, the program also performs QMCS and LHS using the ***Sampler*** class. QMCS uses low-discrepancy sequences to generate samples, providing a more uniform coverage of the input space compared to traditional MC methods. LHS, on the other hand, divides the input space into intervals and samples within each interval to ensure a more comprehensive exploration of the input parameter space. These techniques offer different advantages in terms of accuracy and computational efficiency.

The results from these sampling techniques are compared using the ***Evaluator*** class, which calculates and compares various statistical measures such as mean, variance, standard deviation, and confidence intervals. The ***Evaluator*** class also performs statistical hypothesis tests (e.g., t-tests and ANOVA) to determine whether there are significant differences between the results obtained from the different sampling methods. Furthermore, the module integrates a Point Estimate Method (PEM) analysis through its ***PEMAnalysis*** component. This PEM analysis computes higher-order statistics like skewness, kurtosis, and variance for each subsystem based on the sampling results.

Beyond probabilistic sampling methods, the program also includes a Bayesian inference analysis through the ***BayesianAnalysis*** class. This module updates the probability distributions of the input parameters based on observed data, allowing for a more refined assessment of uncertainty. The Bayesian analysis results include posterior distributions of the parameters, uncertainty intervals, and visualisations of the posterior samples.

Additionally, the program performs an probabilistic analytical UA using the first-order Taylor series approach, implemented in the ***UncertaintyAnalysis*** class. This method provides an approximation of the uncertainty in the network's outputs based on the linear propagation of input uncertainties.

Finally, the program incorporates a possibilistic UA using fuzzy logic principles, handled by the ***PossibilisticUncertaintyAnalysis*** class. This approach deals with uncertainty in a non-probabilistic manner, using fuzzy sets and Z-numbers to model uncertain parameters. Z-numbers provide a way to represent both the uncertainty and the reliability of the data. Defuzzification methods are then applied to derive crisp outputs from these fuzzy sets.

Throughout the program, various visualisation techniques are employed to help interpret the results, including histograms, error bars, violin plots, cumulative distribution functions (CDFs), and sensitivity analyses. These plots provide insights into the behaviour of the network under uncertainty and help identify the most influential parameters.

The code is organised into several Python files and two JSON configuration files, each contributing to different aspects of UA:

- ***main.py***: The main entry point for running simulations. It orchestrates the different routes and techniques used in the analysis.

- ***importer.py***: Handles the import of required libraries and dependencies.

- ***network.py***: Contains the Network class, which models the energy system and simulates its operation.

- ***registry.py***: Manages the registration and dynamic creation of subsystem classes within the network.

- ***flows.py***: Manages the creation and handling of input and output streams within subsystems.

- ***stream.py***: Defines the Stream class, representing a fluid stream with various thermodynamic properties.

- ***thermostream.py***: Handles the thermodynamic properties of streams using the CoolProp library.

- ***energy.py***: Provides functions to calculate the energy consumption of subsystems within the network.

- ***montecarlo.py***: Implements Monte Carlo sampling simulation methods.

- ***distribution.py***: Includes the DistributionAnalyser class, responsible for analysing and managing statistical distributions.

- ***sampletechs.py***: Implements additional sampling techniques, including Quasi-Monte Carlo and Latin Hypercube sampling.

- ***evaluation.py***: Provides methods for evaluating and comparing the results of different sampling techniques.

- ***pem.py***: Implements Point Estimate Method (PEM) analysis for uncertainty quantification.

- ***bayesian.py***: Implements the Bayesian UA.

- ***analytical.py***: Implements the Probabilistic-Analytical UA.

- ***possibilistic.py***: Implements the Possibilistic UA.

- ***config.json***: Defines the configuration for the system network, specifying the subsystems, their inputs, outputs, and operational parameters.

- ***uncertain_params.json***: A JSON configuration file defining the uncertain parameters in the system, including their types, distributions, and characteristics.

## Detailed Description of Key Files
### 1. main.py
The ***main.py*** file is the central entry point for executing the uncertainty analysis framework. It orchestrates the entire process, handling command-line arguments, loading configurations, and coordinating the execution of various uncertainty analysis methods.

- **Argument Parsing**: Sets up command-line options to control the verbosity of the execution.

- **Configuration Loading**: Loads the system configuration from ***config.json*** and initialises the network.

- **Network Simulation**: Builds and simulates the network, calculating energy consumption and emissions.

- **Probabilistic Route**:
Runs MCS, QMCS, and LHS simulations.
Visualises the emission uncertainties and performs statistical evaluations.
Conducts Bayesian analysis to update parameter distributions and assess uncertainties.
Performs Analytical UA using first-order Taylor series expansion.

- **Possibilistic Route**:
Loads uncertain parameters and applies Possibilistic UA using fuzzy logic and Z-numbers.
Visualises and outputs the results of the possibilistic analysis.

### 2. importer.py
This file manages the import of all necessary libraries and modules required throughout the project. It ensures that dependencies are available for different components of the UA.

### 3. network.py
The ***Network*** class is central to simulating the industrial system. It models the configuration, operations, and energy consumption of various subsystems, providing essential data for uncertainty analysis.

- **Subsystem Registration**: Dynamically registers subsystem classes based on the configuration, allowing flexibility in defining new units.

- **Stream Handling**: Manages the creation and processing of input and output streams within subsystems.

- **Simulation**: Runs the network simulation across defined iterations, updating streams and calculating the final ammonia flow.

- **Energy Consumption**: Calculates the energy consumption for each subsystem based on configuration formulas.

- **Emissions Calculation**: Computes CO2 emissions based on energy consumption and the emission factor defined in the configuration.

### 4. registry.py
The ***SubsystemRegistry*** class manages the registration and dynamic creation of subsystem classes within the network.

- **Subsystem Class Registration**: Maps unit types to their corresponding classes.

- **Subsystem Creation**: Dynamically creates subsystem instances based on their configuration.

### 5. flows.py
The ***Flows*** class is responsible for creating and managing input and output streams within subsystems.

- **Input Stream Creation**: Generates input streams based on configuration, either from other subsystems' outputs or directly from configuration parameters.

- **Output Stream Creation**: Produces output streams for subsystems, supporting stream mixing if required.

### 6. stream.py
The ***Stream*** class represents a fluid stream with various thermodynamic properties.

- **Property Management**: Handles setting, getting, and calculating various properties of the stream, such as temperature, pressure, and enthalpy.

- **Stream Mixing**: Provides functionality to mix two streams into a single output stream.

- **Heat Transfer**: Simulates heat transfer processes within the stream.

### 7. thermostream.py
The ***ThermoStream*** class handles the thermodynamic properties of streams using the CoolProp library.

- **Property Calculation**: Calculates properties like enthalpy, internal energy, and entropy based on temperature and pressure.

- **Phase Determination**: Identifies the phase (liquid, gas, or two-phase) of the stream based on its properties.

- **Mixing and Heat Exchange**: Provides methods for mixing streams and simulating heat exchangers.

### 8. energy.py
This file contains functions for calculating the energy consumption of subsystems within the network.

- **Energy Consumption Calculation**: Evaluates the energy consumption for each subsystem based on the formulas defined in the configuration.

- **Iteration Management**: Adjusts the energy consumption formula based on the iteration, especially for subsystems with recycling processes.

### 9. montecarlo.py
This module implements Monte Carlo sampling and simulation methods.

- **Input/Output Parameter Identification**: Identifies parameters for sampling based on the system configuration.

- **Probability Distribution Assignment**: Assigns probability distributions to the identified parameters.

- **Sample Generation**: Generates samples for the parameters based on their distributions.

- **Simulation Execution**: Runs the MCS simulation and calculates emission uncertainties.

- **Visualisation**: Provides methods to visualise the simulation results, including histograms, error bars, violin plots, and cumulative distribution functions.

### 10. distribution.py
The ***DistributionAnalyser*** class is responsible for analysing statistical distributions associated with system parameters.

- **Distribution Fitting**: Fits various probability distributions to the data.

- **AIC/BIC Calculation**: Computes the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) to evaluate the goodness of fit.

- **Best Distribution Selection**: Selects the best-fitting distribution based on the AIC/BIC values.

- **Sample Generation**: Generates samples from the best-fitting distribution.

### 11. sampletechs.py
This module implements additional sampling techniques, including Quasi-Monte Carlo and Latin Hypercube sampling.

- **Quasi-Monte Carlo Sampling**: Generates quasi-random samples using the Sobol sequence.

- **Latin Hypercube Sampling**: Produces samples using the Latin Hypercube method.

- **Simulation Execution**: Runs simulations using the generated samples and calculates emission uncertainties.

- **Sensitivity Analysis**: Performs Sobol sensitivity analysis on the simulation results.

### 12. evaluation.py
The ***Evaluator*** class provides methods for evaluating and comparing the results of different sampling techniques.

- **Statistical Measure Calculation**: Computes mean, variance, standard deviation, and confidence intervals for the simulation results.

- **Comparison**: Compares the statistical measures across different sampling techniques.

- **Boxplot Visualisation**: Plots boxplots to visualise the distribution of results from different techniques.

- **Statistical Testing**: Performs t-tests and ANOVA to compare the results statistically.

- **PEM Analysis**: Conducts PEM analysis on the sampling simulation results.

### 13. pem.py
The ***PEMAnalysis*** class implements Point Estimate Method analysis for uncertainty quantification.

- **Moment Calculation**: Calculates statistical moments (mean, standard deviation, skewness, kurtosis) for the input data.

- **Concentration Points**: Determines concentration points used in PEM analysis.

- **Uncertainty Metrics**: Computes uncertainty metrics such as variance, skewness, and kurtosis for each unit.

### 14. bayesian.py

This module introduces Bayesian UA. It combines prior information with observed data to update the probability distribution of uncertain parameters.

- **Identify Subsystems**: Identifies subsystems in the configuration subject to analysis.

- **Define Prior Distribution**: Defines the prior distribution for the parameters, typically assuming a uniform distribution.

- **Calculate Likelihood**: Computes the likelihood of observed data given the parameters.

- **Calculate Posterior**: Calculates the posterior distribution using Bayes' theorem.

- **Sample Posterior**: Generates samples from the posterior distribution using MCMC or slice sampling.

- **Analyse Uncertainty**: Outputs key statistics and credible intervals for each subsystem's parameters.

### 15. analytical.py
This file focuses on Analytical UA based on the 1st order approximation of Taylor series, which evaluates how uncertainties in input parameters affect output uncertainties using partial derivatives.

- **Assign Probability Distributions**: Associates each identified parameter with an appropriate probability distribution.

- **Compute Uncertainties**: Assigns probability distributions to input/output parameters and estimates their uncertainties.

- **Compute Partial Derivatives***: Calculates how small changes in input parameters influence the output emissions.

- **Estimate Uncertainties**: Combines partial derivatives with assigned uncertainties to estimate the overall uncertainty in emissions.

- **Plot Uncertainties**: Visualises the uncertainties of input parameters and their impact on system emissions

### 16. possibilistic.py
This module introduces Possibilistic UA, using fuzzy logic to handle situations where traditional probabilistic approaches may not apply.

- **Membership Functions**: Defines various types of membership functions (e.g., Triangular, Gaussian, Bell Shaped) used to represent uncertain parameters.

- **Define Membership Functions**: Assigns membership functions to parameters based on their fuzzy characteristics.

- **Calculate Alpha-cuts**: Calculates alpha-cuts for membership functions, which represent intervals of possible values for uncertain parameters at different confidence levels.

- **Uncertainty Propagation**: Propagates uncertainties through the system by applying alpha-cuts and calculating the resulting emissions.

- **Use of Z-numbers**: Incorporates Z-nº to represent both the uncertainty of the data and the confidence in that data. This allows for a more nuanced representation of uncertainty in the system.

- **Defuzzification of results**: Includes several defuzzification methods, such as the centroid, bisector, mean-of-maximum, smallest-of-maximum, and lowest-of-maximum methods. This process converts the fuzzy outputs of the analysis into crisp, actionable values.

- **Visualise Uncertainty**: Plots the results of the possibilistic analysis, including parameter uncertainties and overall uncertainty propagation.

### 17. config.json
The ***config.json*** file defines the configuration for the energy system, including the subsystems, their inputs, outputs, and operational parameters. It is used to initialise the Network class and guide the simulation process.

Example Structure:

```
{
    "ASU": {
        "type": "ASU",
        "input_stream": {
            "fluid_name": "Air",
            "flow_rate": 100,
            "temperature": 300,
            "pressure": 101325,
            "mass_flow": 120,
            "enthalpy": 300000,
            "internal_energy": 298000,
            "entropy": 2900,
            "density": 1.2,
            "gamma": 1.4,
            "phase": 0
        },
        "output_streams": [
            {"fluid_name": "Nitrogen", "temperature": 77, "split_ratio": 0.79, "next_subsystem": "SyngasCompression", "origin": true},
            {"fluid_name": "Oxygen", "temperature": 90, "split_ratio": 0.21, "next_subsystem": null, "origin": true}
        ],
        "efficiency": 0.8,
        "electricity_conversion": 3600,
        "energy_consumption": "(input_streams_0['flow_rate'] * 2.44 * abs(output_streams_0['temperature'] - input_streams_0['temperature'])*inputs['efficiency'])/inputs['electricity_conversion']"
    },
    ...
}
```

### 18. uncertain_params.json
This JSON file contains the definitions for uncertain parameters used in the system, specifying their names, types, distributions, and possibility distributions.

Example Structure:

```
[
  {
    "name": "ASU.input_stream.temperature",
    "type": "gaussian",
    "mean": 300,
    "stddev": 5,
    "possibility_distribution": "math.exp(-((x - 300) ** 2) / (2 * 5 ** 2))"
  },
  ...
]
```
from importer import *

from network import Network
from montecarlo import MonteCarloSampler
from sampletechs import Sampler
from bayesian import BayesianAnalysis
from evaluation import Evaluator
from analytical import UncertaintyAnalysis
from possibilistic import PossibilisticUncertaintyAnalysis

def main():

    ##  Main function to run the UA simulations
    ##  Handles parsing command-line arguments, loading configurations and executing various UA routes and techniques
    
    # Argument parser setup for command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Run in verbose mode")
    args = parser.parse_args()

    # Set logging level based on the verbose flag
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        print("Verbose mode enabled")
    else:
        logging.basicConfig(level=logging.INFO)
        print("Fast mode enabled")
    
    # Load the configuration file for the network
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialise and simulate the network based on the config
    network = Network(config)
    network.build_network()
    network.simulate()
    network.calculate_energy_consumption()   
    network.calculate_emissions()

    # Re-load the configuration to ensure no changes from simulation steps affect it
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Set up logging to output to a file
    logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(message)s')
    
    # PROBABILISTIC ROUTE

    # MonteCarlo, Quasi-MonteCarlo & Latin Hypercube:
    nsamples = 100 # number of samples for sampling techniques
    
    # Monte Carlo sampling
    mc_sampler = MonteCarloSampler(config)
    mc_results, uncertainty_mc = mc_sampler.run_simulation(nsamples, 'MonteCarlo', output_file='MonteCarloResults.csv')
    print(uncertainty_mc)
    mc_sampler.visualize_emission_uncertainty('MonteCarlo', uncertainty_mc)
    
    # Quasi-Monte Carlo sampling with SA (Sobol indices)
    sampler = Sampler(config)
    qmc_results, uncertainty_qmc = sampler.run_quasi_simulation(nsamples, 'Quasi-MonteCarlo', output_file='QuasiMCResults.csv', sensitivity = True) # Parameters Sensitivity: Sobol Indices
    print(uncertainty_qmc)
    mc_sampler.visualize_emission_uncertainty('Quasi-MonteCarlo', uncertainty_qmc)
    
    # Latin Hypercube sampling
    lhs_results, uncertainty_lhs = sampler.run_latin_simulation(nsamples, 'Latin Hypercube', output_file='LatinHResults.csv')
    print(uncertainty_lhs)
    mc_sampler.visualize_emission_uncertainty('Latin Hypercube', uncertainty_lhs)

    # Plot detailed unit results for each sampling technique (Histograms, error bars, violin and CDF plots)
    mc_sampler.plot_subsystems('MonteCarlo', mc_results)
    sampler.plot_subsystems('Quasi-MonteCarlo', qmc_results)
    sampler.plot_subsystems('Latin Hypercube', lhs_results)
    
    # Organize results by subsystem
    mc_subsystem = {subsystem: [result[subsystem] for result in mc_results] for subsystem in mc_results[0].keys()}
    qmc_subsystem = {subsystem: [result[subsystem] for result in qmc_results] for subsystem in qmc_results[0].keys()}
    lhs_subsystem = {subsystem: [result[subsystem] for result in lhs_results] for subsystem in lhs_results[0].keys()}    

    # Perform statistical analysis on the results
    evaluator = Evaluator(mc_subsystem, qmc_subsystem, lhs_subsystem)
    evaluator.compare_statistical_measures()
    evaluator.plot_boxplots()
    evaluator.perform_statistical_tests()
    
    # Perform Point Estimate Method analysis on results
    evaluator.perform_pem_analysis()
    evaluator.plot_pem_results()
    
    # Bayesian inference analysis
    bayesian_analysis = BayesianAnalysis(config, network)
    posterior_samples = bayesian_analysis.perform_bayesian_analysis()
    bayesian_analysis.plot_posterior_samples(posterior_samples)
    bayesian_analysis.plot_posterior_distributions(posterior_samples)
    bayesian_analysis.plot_uncertainty_intervals(posterior_samples)
    bayesian_analysis.analyze_uncertainty(posterior_samples)
    
    # Analytical uncertainty analysis using the first-order Taylor series approach
    uncertainty_analysis = UncertaintyAnalysis(config)
    uncertainty_emissions = uncertainty_analysis.run()
    uncertainty_analysis.plot_uncertainties(uncertainty_emissions)   
    for key, value in uncertainty_emissions.items():
        print(f"Estimated uncertainty in emissions for {key}: {value:.2f}")

    # POSSIBILISTIC ROUTE

    # Load uncertain parameters dictionary for possibilistic UA
    with open('uncertain_params.json', 'r') as f:
        uncertain_params = json.load(f)

    # Define defuzzification methods needed
    defuzzification_methods = {
        'centroid'}
    #    'bisector',
    #    'mom',
    #    'som',
    #    'lom'}

    # Run possibilistic UA
    possibilistic_analysis = PossibilisticUncertaintyAnalysis(Network, config, uncertain_params, defuzzification_methods)
    results = possibilistic_analysis.run_possibilistic_uncertainty_analysis()

    # Output and visualise results
    for method, result in results.items():
        crisp_output = result["crisp_outputs"]
        uncertainty_propagation = result["uncertainty_propagation"]
        parameter_uncertainties = result["parameter_uncertainties"]
    
        print(f"Defuzzification Method: {method}")
        print(f"Crisp Output: {crisp_output}")
        print(f"Uncertainty Propagation: {uncertainty_propagation}")
        print(f"Parameter Uncertainties: {parameter_uncertainties}")

        possibilistic_analysis.visualize_uncertainty(crisp_output, uncertainty_propagation, parameter_uncertainties)

if __name__ == "__main__":
    main()

    


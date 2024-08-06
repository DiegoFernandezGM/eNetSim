import json
from network import Network
from montecarlo import MonteCarloSampler
from sampletechs import Sampler
from bayesian import BayesianAnalysis
import logging
import argparse
from evaluation import Evaluator
from analytical import UncertaintyAnalysis
from possibilistic import PossibilisticUncertaintyAnalysis
import math
import numpy as np 
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Run in verbose mode")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        print("Verbose mode enabled")
    else:
        logging.basicConfig(level=logging.INFO)
        print("Fast mode enabled")

    with open('config.json', 'r') as f:
        config = json.load(f)
    network = Network(config)
    network.build_network()
    network.simulate()
    network.calculate_energy_consumption()   
    network.calculate_emissions()

    with open('config.json', 'r') as f:
        config = json.load(f)

    logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(message)s')
    
    #MonteCarlo, Quasi-MonteCarlo & Latin Hypercube
    mc_sampler = MonteCarloSampler(config)
    mc_results, uncertainty_mc = mc_sampler.run_simulation(100, 'MonteCarlo', output_file='MonteCarloResults.csv')
    mc_sampler.visualize_emission_uncertainty(uncertainty_mc)
    #sampler = Sampler(config)
    #qmc_results = sampler.run_quasi_simulation(100, 'Quasi-MonteCarlo', output_file='QuasiMCResults.csv', sensitivity = True) # Parameters Sensitivity: Sobol Indices
    #lhs_results = sampler.run_latin_simulation(100, 'Latin Hypercube', output_file='LatinHResults.csv')
    
    #mc_sampler.plot_subsystems(mc_results)
    #sampler.plot_subsystems(qmc_results)
    #sampler.plot_subsystems(lhs_results)
    
    #mc_subsystem = {subsystem: [result[subsystem] for result in mc_results] for subsystem in mc_results[0].keys()}
    #qmc_subsystem = {subsystem: [result[subsystem] for result in qmc_results] for subsystem in qmc_results[0].keys()}
    #lhs_subsystem = {subsystem: [result[subsystem] for result in lhs_results] for subsystem in lhs_results[0].keys()}    

    # Statistical analysis
    #evaluator = Evaluator(mc_subsystem, qmc_subsystem, lhs_subsystem)
    #evaluator.compare_statistical_measures()
    #evaluator.plot_boxplots()
    #evaluator.perform_statistical_tests()
    #evaluator.perform_pem_analysis()
    #evaluator.plot_pem_results()
    
    # Bayesian inference analysis
    #bayesian_analysis = BayesianAnalysis(config, network)
    #posterior_samples = bayesian_analysis.perform_bayesian_analysis()
    #bayesian_analysis.plot_posterior_samples(posterior_samples)
    #bayesian_analysis.plot_posterior_distributions(posterior_samples)
    #bayesian_analysis.plot_uncertainty_intervals(posterior_samples)
    #bayesian_analysis.analyze_uncertainty(posterior_samples)
    
    # Analytical uncertainty analysis
    uncertainty_analysis = UncertaintyAnalysis(config)
    #mean_emissions, parameter_labels, uncertainty_ts  = uncertainty_analysis.calculate_emission_uncertainty(iteration=5)
    #uncertainty_analysis.visualize_emission_uncertainty(parameter_labels, uncertainty_ts)
    uncertainty_emissions = uncertainty_analysis.run()
    uncertainty_analysis.plot_uncertainties(uncertainty_emissions)
    for key, value in uncertainty_emissions.items():
        print(f"Estimated uncertainty in emissions for {key}: {value:.2f}")

    #print(f"Mean Emissions: {mean_emissions}")
    #print(f"Emission Uncertainty - TS: {uncertainty_ts}")
    #print(f"Emission Uncertainty - MC: {uncertainty_mc}")

    
    # Function to convert lambda strings back to functions
    #def lambda_parser(lambda_str):
    #    return eval(f"lambda x: {lambda_str}")

    # Read JSON file
    with open('uncertain_params.json', 'r') as f:
        uncertain_params = json.load(f)

    defuzzification_methods = {
        'centroid'}
    #    'bisector',
    #    'mom',
    #    'som',
    #    'lom'}

    possibilistic_analysis = PossibilisticUncertaintyAnalysis(Network, config, uncertain_params, defuzzification_methods)
    results = possibilistic_analysis.run_possibilistic_uncertainty_analysis()

    for method, result in results.items():
        crisp_output = result["crisp_outputs"]
        uncertainty_propagation = result["uncertainty_propagation"]
        parameter_uncertainties = result["parameter_uncertainties"]
    
        print(f"Defuzzification Method: {method}")
        print(f"Crisp Output: {crisp_output}")
        print(f"Uncertainty Propagation: {uncertainty_propagation}")
        print(f"Parameter Uncertainties: {parameter_uncertainties}")

        possibilistic_analysis.visualize_uncertainty(crisp_output, uncertainty_propagation, parameter_uncertainties)

    print(uncertainty_mc)
    #print()

if __name__ == "__main__":
    main()

    


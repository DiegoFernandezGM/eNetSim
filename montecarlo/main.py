import json
from network import Network
from montecarlo import MonteCarloSampler
from sampletechs import Sampler
import logging
import argparse
from evaluation import Evaluator

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
    #network.calculate_energy_consumption()   
    network.calculate_emissions()

    with open('config.json', 'r') as f:
        config = json.load(f)

    logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(message)s')
    
    mc_sampler = MonteCarloSampler(config)
    mc_results = mc_sampler.run_simulation(100, 'MonteCarlo', output_file='MonteCarloResults.csv')
    sampler = Sampler(config)
    qmc_results = sampler.run_quasi_simulation(100, 'Quasi-MonteCarlo', output_file='QuasiMCResults.csv', sensitivity = True) # Parameters Sensitivity: Sobol Indices
    lhs_results = sampler.run_latin_simulation(100, 'Latin Hypercube', output_file='LatinHResults.csv')

    #mc_sampler.plot_subsystems(mc_results)
    #sampler.plot_subsystems(qmc_results)
    #sampler.plot_subsystems(lhs_results)
    mc_subsystem = {subsystem: [result[subsystem] for result in mc_results] for subsystem in mc_results[0].keys()}
    qmc_subsystem = {subsystem: [result[subsystem] for result in qmc_results] for subsystem in qmc_results[0].keys()}
    lhs_subsystem = {subsystem: [result[subsystem] for result in lhs_results] for subsystem in lhs_results[0].keys()}    
   
    evaluator = Evaluator(mc_subsystem, qmc_subsystem, lhs_subsystem)
    evaluator.compare_statistical_measures()
    evaluator.plot_boxplots()
    evaluator.perform_statistical_tests() 
    
if __name__ == "__main__":
    main()

    


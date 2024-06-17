import json
from network import Network
from montecarlo import MonteCarloSampler
from sampletechs import Sampler

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    network = Network(config)
    network.build_network()
    network.simulate()
    #network.calculate_energy_consumption()   
    network.calculate_emissions()

    with open('config.json', 'r') as f:
        config = json.load(f)

    mc_sampler = MonteCarloSampler(config)
    results = mc_sampler.run_simulation(100)
    mc_sampler.plot_subsystems(results)

    sampler = Sampler(config)
    qmc_results = sampler.run_quasi_simulation(1000)
    sampler.plot_subsystems(qmc_results)
    lhs_results = sampler.run_latin_simulation(1000)
    sampler.plot_subsystems(lhs_results)

if __name__ == "__main__":
    main()

    


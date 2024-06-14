import json
from network import Network

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    network = Network(config)
    network.build_network()
    network.simulate()
    #network.calculate_energy_consumption()
    network.calculate_emissions()

if __name__ == "__main__":
    main()
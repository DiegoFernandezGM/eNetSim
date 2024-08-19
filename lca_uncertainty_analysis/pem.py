from importer import *

class PEMAnalysis:
    def __init__(self, mc_subsystems):
        
        ##  Initialises the Point Estimate Method (PEM) class with Monte Carlo results
        ##  Computes the various statistical measures functions (mean, standard deviation, skewness, kurtosis, variance)

        self.mc_results = mc_subsystems
        self.num_subsystems = len(mc_subsystems)
        self.z_values = list(mc_subsystems.values())
        
        # Calculate stddev and pdfs
        self.stds = [np.std(values) for values in self.z_values]
        self.pdfs = [1 / len(values) for values in self.z_values]

        # Initialise lists to store calculated statistics
        self.means = []
        self.stds = []
        self.third_moments = []
        self.zk_i = []
        self.epsilonk_i = []
        self.P_k_i = []
        self.skewnesses = []
        self.kurtoses = []
        self.variances = []

        # Calculate all necessary moments and statistics
        for subsystem, values in self.mc_results.items():
            means, stds, third_moments = self.calculate_moments(values)
            zk_i, epsilonk_i, P_k_i = self.calculate_concentration_points(means, stds, third_moments)
            skewness = self.calculate_skewness(zk_i, epsilonk_i, P_k_i)
            kurtosis = self.calculate_kurtosis(zk_i, epsilonk_i, P_k_i)
            variance = self.calculate_variance(zk_i, epsilonk_i, P_k_i)

            # Append calculated values to respective lists
            self.means.append(means)
            self.stds.append(stds)
            self.third_moments.append(third_moments)
            self.zk_i.append(zk_i)
            self.epsilonk_i.append(epsilonk_i)
            self.P_k_i.append(P_k_i)
            self.skewnesses.append(skewness)
            self.kurtoses.append(kurtosis)
            self.variances.append(variance)

    def calculate_moments(self, values):

        ##  Calculates mean, stddev and 3rd moment for given set of values

        mean = np.mean(values)
        std = np.std(values)
        third_moment = np.mean((values - mean)**3)
        
        return mean, std, third_moment

    def calculate_concentration_points(self, mean, std, third_moment):
        
        ##  Calculates concentration points: zk, epsilonk and Pk based on the above estimated mean,stddev and 3rd moment
        
        zk = ((1/2)*(third_moment/std**3)) + (-1)*math.sqrt(1 + (1/2)*(third_moment/std**3)**2)
        epsilonk = mean + std*zk
        Pk = (-1)*(zk / (2*math.sqrt(1 + (1/2)*(third_moment/std**3)**2)))
        
        return zk, epsilonk, Pk

    def calculate_skewness(self, zk, epsilonk, Pk):
        
        ##  Calculates skewness based on the above estimated concentration points: zk, epsilonk and Pk.

        ey = Pk*epsilonk
        ey2 = Pk*epsilonk**2
        ey3 = Pk*epsilonk**3
        skewness = (ey3 - 3*ey*ey2 + 2*ey**3) / ((ey2 - ey**2)**(3/2))
        
        return skewness

    def calculate_kurtosis(self, zk, epsilonk, Pk):
        
        ##  Calculates kurtosis based on the concentration points: zk, epsilonk and Pk

        ey = Pk*epsilonk
        ey2 = Pk*epsilonk**2
        ey3 = Pk*epsilonk**3
        ey4 = Pk*epsilonk**4
        kurtosis = (ey4 - 4*ey*ey3 + 6*ey**2*ey2 - 3*ey**4) / ((ey2 - ey**2)**2)
        
        return kurtosis

    def calculate_variance(self, zk, epsilonk, Pk):
        
        ##  Calculates variance based on concentration points: zk, epsilonk and Pk
        
        ey = Pk*epsilonk
        ey2 = Pk*epsilonk**2
        variance = ey2 - ey**2
        
        return variance

    def perform_pem_analysis(self):

        ## Prints each subsystem's PEM analysis results: mean, stddec, skewness, kurtosis and variance

        print("PEM Analysis Results:")
        for i in range(self.num_subsystems):
            print(f"Subsystem {i+1}:")
            print(f"Mean: {self.means[i]}")
            print(f"Standard Deviation: {self.stds[i]}")
            print(f"Skewness: {self.skewnesses[i]}")
            print(f"Kurtosis: {self.kurtoses[i]}")
            print(f"Variance: {self.variances[i]}")
            print()
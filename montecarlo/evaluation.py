import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
import seaborn as sns
from pem import PEMAnalysis

class Evaluator:
    def __init__(self, mc_results, qmc_results, lhs_results):
        self.mc_results = mc_results
        self.qmc_results = qmc_results
        self.lhs_results = lhs_results
        self.stats = self.calculate_statistical_measures()
        self.pem_analysis = PEMAnalysis(mc_results)

    def calculate_statistical_measures(self):
        techniques = ['mc', 'qmc', 'lhs']
        results = [self.mc_results, self.qmc_results, self.lhs_results]
        stats = {}

        for i, tech in enumerate(techniques):
            stats[tech] = {}
            for subsystem, values in results[i].items():
                mean = np.mean(values)
                var = np.var(values)
                std = np.std(values)
                ci = self.calculate_confidence_interval(values)

                stats[tech][subsystem] = {
                    'mean': mean,
                    'variance': var,
                    'std_dev': std,
                    'confidence_interval': ci
                }

        return stats

    def calculate_confidence_interval(self, data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        ci = std * np.sqrt((1 - confidence) / (n - 1))
        return (mean - ci, mean + ci)

    def compare_statistical_measures(self):
        for subsystem in self.stats['mc'].keys():
            print(f"Statistical Measures for {subsystem}:")
            print(" Technique  |  Mean  |  Variance  |  Standard Deviation  |  Confidence Interval")
            print("-----------|--------|------------|---------------------|---------------------")

            for tech, stats in self.stats.items():
                print(f" {tech.capitalize()}  | {stats[subsystem]['mean']:.4f} | {stats[subsystem]['variance']:.4f} | {stats[subsystem]['std_dev']:.4f} | {stats[subsystem]['confidence_interval'][0]:.4f} - {stats[subsystem]['confidence_interval'][1]:.4f}")

    def plot_boxplots(self):
        for subsystem in self.stats['mc'].keys():
            plt.figure(figsize=(10, 6))
            sns.set_style('whitegrid')
            plt.boxplot([self.mc_results[subsystem], self.qmc_results[subsystem], self.lhs_results[subsystem]],
                        labels=['MonteCarlo', 'Quasi-MoneCarlo', 'Latin Hypercube'],
                        boxprops={'color': 'blue'},
                        flierprops={'color': 'red'},
                        whiskerprops={'color': 'green'},
                        capprops={'color': 'purple'})
            plt.xlabel('Technique', fontsize=14)
            plt.ylabel('Output Value', fontsize=14)
            plt.title(f"Distribution of Output Values for {subsystem}", fontsize=16)
            plt.legend(loc='upper right', title='Techniques')
            plt.show()

    def perform_statistical_tests(self):
        for subsystem in self.stats['mc'].keys():
            t_stat, p_val = ttest_ind(self.mc_results[subsystem], self.qmc_results[subsystem])
            print(f"t-test (Monte Carlo vs Quasi-Monte Carlo) for {subsystem}: t-stat={t_stat:.4f}, p-val={p_val:.4f}")

            t_stat, p_val = ttest_ind(self.mc_results[subsystem], self.lhs_results[subsystem])
            print(f"t-test (Monte Carlo vs Latin Hypercube) for {subsystem}: t-stat={t_stat:.4f}, p-val={p_val:.4f}")

            t_stat, p_val = ttest_ind(self.qmc_results[subsystem], self.lhs_results[subsystem])
            print(f"t-test (Quasi-Monte Carlo vs Latin Hypercube) for {subsystem}: t-stat={t_stat:.4f}, p-val={p_val:.4f}")

            f_stat, p_val = f_oneway(self.mc_results[subsystem], self.qmc_results[subsystem], self.lhs_results[subsystem])
            print(f"ANOVA (all techniques) for {subsystem}: f-stat={f_stat:.4f}, p-val={p_val:.4f}")

    def perform_pem_analysis(self):
        self.pem_analysis.perform_pem_analysis()

    def plot_pem_results(self):
        sns.set_style('whitegrid')

        # Figure 1: Skewness
        plt.figure(figsize=(10, 6))
        for i in range(self.pem_analysis.num_subsystems):
            plt.plot([self.pem_analysis.skewnesses[i]], label=f'Skewness Subsystem {i+1}', marker='o')
        plt.xlabel('Input Variable Z', fontsize=14)
        plt.ylabel('Skewness', fontsize=14)
        plt.title('Skewness of All Subsystems', fontsize=16)
        plt.legend(loc='upper right', title='Skewness')
        plt.show()

        # Figure 2: Kurtosis
        plt.figure(figsize=(10, 6))
        for i in range(self.pem_analysis.num_subsystems):
            plt.plot([self.pem_analysis.kurtoses[i]], label=f'Kurtosis Subsystem {i+1}', marker='o')
        plt.xlabel('Input Variable Z', fontsize=14)
        plt.ylabel('Kurtosis', fontsize=14)
        plt.title('Kurtosis of All Subsystems', fontsize=16)
        plt.legend(loc='upper right', title='Kurtosis')
        plt.show()

        # Figure 3: Variance
        plt.figure(figsize=(10, 6))
        for i in range(self.pem_analysis.num_subsystems):
            plt.plot([self.pem_analysis.variances[i]], label=f'Variance Subsystem {i+1}', marker='o')
        plt.xlabel('Input Variable Z', fontsize=14)
        plt.ylabel('Variance', fontsize=14)
        plt.title('Variance of All Subsystems', fontsize=16)
        plt.legend(loc='upper right', title='Variance')
        plt.show()
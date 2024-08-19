from importer import *

from pem import PEMAnalysis

class Evaluator:
    def __init__(self, mc_results, qmc_results, lhs_results):      
        self.mc_results = mc_results
        self.qmc_results = qmc_results
        self.lhs_results = lhs_results
        self.stats = self.calculate_statistical_measures()
        self.pem_analysis = PEMAnalysis(mc_results)

    def calculate_statistical_measures(self):
        
        ##  Calculates basic statistical measures (mean, variance, standard deviation, and confidence interval) for each sampling technique results 
        
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
        
        ##  Calculates confidence intervals for a dataset based on specified confidence levels
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        ci = std * np.sqrt((1 - confidence) / (n - 1))
        
        return (mean - ci, mean + ci)

    def compare_statistical_measures(self):
        
        ##  Compares statistical measures (mean, variance, standard deviation and confidence interval) across different sampling methods
        
        for subsystem in self.stats['mc'].keys():
            print(f"Statistical Measures for {subsystem}:")
            print(" Technique  |  Mean  |  Variance  |  Standard Deviation  |  Confidence Interval")
            print("-----------|--------|------------|---------------------|---------------------")

            for tech, stats in self.stats.items():
                print(f" {tech.capitalize()}  | {stats[subsystem]['mean']:.4f} | {stats[subsystem]['variance']:.4f} | {stats[subsystem]['std_dev']:.4f} | {stats[subsystem]['confidence_interval'][0]:.4f} - {stats[subsystem]['confidence_interval'][1]:.4f}")

    def plot_boxplots(self):
        
        ##  Plots boxplots for the unit output values across different sampling techniques      

        base_color = '#6C757D' # Color for boxes
        dark_color = 'black'  # Color for edges + medians
        light_color = '#495057'  # Color for whiskers + caps
        
        for subsystem in self.stats['mc'].keys():
            plt.figure(figsize=(12, 8))
            sns.set_style('whitegrid')
            
            boxprops = {'edgecolor': dark_color, 'linewidth': 2}
            medianprops = {'color': 'black', 'linewidth': 2}
            whiskerprops = {'color': light_color, 'linewidth': 1.5}
            capprops = {'color': light_color, 'linewidth': 1.5}
            flierprops = {'markerfacecolor': 'grey', 'markeredgewidth': 0.5}
            
            bp = plt.boxplot([self.mc_results[subsystem], self.qmc_results[subsystem], self.lhs_results[subsystem]],
                            labels=['Monte Carlo', 'Quasi-Monte Carlo', 'Latin Hypercube'],
                            patch_artist=True,
                            boxprops=boxprops,
                            medianprops=medianprops,
                            whiskerprops=whiskerprops,
                            capprops=capprops,
                            flierprops=flierprops)
            
            for patch in bp['boxes']:
                patch.set_facecolor(base_color)
            
            # Set chart labels, title, and grid
            plt.xlabel('Technique', fontsize=14, fontweight='bold', family='Arial')
            plt.ylabel('Output Value [kg CO2/kg NH3]', fontsize=14, fontweight='bold', family='Arial')
            plt.title(f"Distribution of Output Values for {subsystem}", fontsize=16, fontweight='bold', pad=20, family='Arial')
            plt.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, color='lightgray')
            
            plt.tight_layout(pad=3)
            plt.show()

    def perform_statistical_tests(self):
        
        ##  Statistical hypothesis tests (t-tests and ANOVA) to compare each subsytem results from different sampling techniques
        
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

        ##  Performs the PEM analysis using the MC results
        
        self.pem_analysis.perform_pem_analysis()

    def plot_pem_results(self):
        
        ##  PEM analysis results for each unit: skewness, kurtosis, and variance

        plt.style.use('seaborn-darkgrid')

        common_params = {
            'color': 'teal',
            'marker': 'o',
            'markersize': 8,
            'linewidth': 2
        }

        # Figure 1: Skewness
        plt.figure(figsize=(12, 7))
        
        for i in range(self.pem_analysis.num_subsystems):
            plt.plot([self.pem_analysis.skewnesses[i]], label=f'Subsystem {i+1}', **common_params)
        
        # Set chart labels, title, legend, and grid
        plt.xlabel('Input Variable Z', fontsize=14, fontweight='bold')
        plt.ylabel('Skewness', fontsize=14, fontweight='bold')
        plt.title('Skewness of All Subsystems', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12, frameon=False, title='Subsystems')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Style spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

        # Figure 2: Kurtosis
        plt.figure(figsize=(12, 7))
        
        for i in range(self.pem_analysis.num_subsystems):
            plt.plot([self.pem_analysis.kurtoses[i]], label=f'Subsystem {i+1}', **common_params)
        
        # Set chart labels, title, legend, and grid
        plt.xlabel('Input Variable Z', fontsize=14, fontweight='bold')
        plt.ylabel('Kurtosis', fontsize=14, fontweight='bold')
        plt.title('Kurtosis of All Subsystems', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12, frameon=False, title='Subsystems')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Style spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

        # Figure 3: Variance
        plt.figure(figsize=(12, 7))
        
        for i in range(self.pem_analysis.num_subsystems):
            plt.plot([self.pem_analysis.variances[i]], label=f'Subsystem {i+1}', **common_params)
        
        # Set chart labels, title, legend, and grid
        plt.xlabel('Input Variable Z', fontsize=14, fontweight='bold')
        plt.ylabel('Variance', fontsize=14, fontweight='bold')
        plt.title('Variance of All Subsystems', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12, frameon=False, title='Subsystems')
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Style spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
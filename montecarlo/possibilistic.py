import numpy as np
import math
from skfuzzy import fuzzy_and, fuzzy_or
import matplotlib.pyplot as plt
from network import Network
import copy
import logging

class PossibilisticUncertaintyAnalysis:
    def __init__(self, model, config, input_parameters, defuzzification_methods):
        self.model = model
        self.config = config
        self.uncertain_params = input_parameters
        self.defuzzification_methods = defuzzification_methods
        self.alpha_levels = [i / 20.0 for i in range(1, 20)]        
        self.membership_functions = {}
        self.output_alpha_cuts = []
        self.z_numbers = {}
    
    class FuzzySet:
        def __init__(self, name, membership_function_obj, domain):
            self.name = name
            self.membership_function_obj = membership_function_obj
            self.domain = domain

        def find_alpha_cut(self, alpha):
            return self.membership_function_obj.find_alpha_cut(alpha)

        def intersection(self, other, alpha):
            x1, x2 = self.find_alpha_cut(alpha)
            other_x1, other_x2 = other.find_alpha_cut(alpha)
            lower_bound = max(x1, other_x1)
            upper_bound = min(x2, other_x2)
            return lower_bound, upper_bound

        def union(self, other, alpha):
            x1, x2 = self.find_alpha_cut(alpha)
            other_x1, other_x2 = other.find_alpha_cut(alpha)
            lower_bound = min(x1, other_x1)
            upper_bound = max(x2, other_x2)
            return lower_bound, upper_bound
        
        def fuzzy_multiply(self, other):
            new_domain = (max(self.domain[0], other.domain[0]), min(self.domain[1], other.domain[1]))
            print(f"Multiplying fuzzy sets with domains {self.domain} and {other.domain} resulting in new domain {new_domain}")

            def combined_membership_function(x):
                if new_domain[0] <= x <= new_domain[1]:
                    # Calculate the crisp input values for each fuzzy set
                    self_x = self.domain[0] + (self.domain[1] - self.domain[0]) * (x - new_domain[0]) / (new_domain[1] - new_domain[0])
                    other_x = other.domain[0] + (other.domain[1] - other.domain[0]) * (x - new_domain[0]) / (new_domain[1] - new_domain[0])

                    # Calculate the membership degrees for each fuzzy set
                    result_self = self.membership_function_obj.membership_function(self_x)
                    result_other = other.membership_function_obj.membership_function(other_x)

                    # Calculate the combined membership degree using the minimum operator
                    result = min(result_self, result_other)
                    print(f"Multiplying membership values at x={x}: {result_self} * {result_other} = {result}")
                    return result
                else:
                    return 0

            new_mf_obj = PossibilisticUncertaintyAnalysis.CombinedMembershipFunction(combined_membership_function)
            print(f"Created Combined MF Multiply: {new_mf_obj}")

            self_params = self.membership_function_obj.get_membership_function_params() if hasattr(self.membership_function_obj, 'get_membership_function_params') else 'Combined'
            other_params = other.membership_function_obj.get_membership_function_params() if hasattr(other.membership_function_obj, 'get_membership_function_params') else 'Combined'
            print(f"Combining Fuzzy Sets: {self.name} (parameters: {self_params})")
            print(f"With Fuzzy Set: {other.name} (parameters: {other_params})")

            # Adjust the membership properties based on the actual x values being processed
            new_params = (new_domain[0], new_domain[1], (new_domain[1] - new_domain[0]) / (self.domain[1] - self.domain[0]), self.membership_function_obj.get_membership_function_params()[1])
            new_domain = (new_domain[0], new_domain[1], self.domain[0], self.domain[1])
            return PossibilisticUncertaintyAnalysis.FuzzySet(f"{self.name} * {other.name}", new_mf_obj, new_domain, new_params)
        
        def fuzzy_add(self, other):
            new_domain = (min(self.domain[0], other.domain[0]), max(self.domain[1], other.domain[1]))
            print(f"Adding fuzzy sets with domains {self.domain} and {other.domain} resulting in new domain {new_domain}")

            def combined_membership_function(x):
                if new_domain[0] <= x <= new_domain[1]:
                    result_self = self.membership_function_obj.membership_function(x)
                    result_other = other.membership_function_obj.membership_function(x)
                    result = max(result_self, result_other)
                    return result
                else:
                    return 0

            new_mf_obj = PossibilisticUncertaintyAnalysis.CombinedMembershipFunction(combined_membership_function)
            print(f"Created Combined MF Add: {new_mf_obj}")

            return PossibilisticUncertaintyAnalysis.FuzzySet(f"{self.name} + {other.name}", new_mf_obj, new_domain)

    class CombinedMembershipFunction:
        def __init__(self, combined_function):
            self.combined_function = combined_function

        def membership_function(self, x):
            return self.combined_function(x)
        
        def get_membership_function_params(self):
            return "Combined"
        
    class TriangularMembershipFunction:
        required_params = ['a', 'b', 'c']

        def __init__(self, name, a, b, c):
            self.name = name
            self.a = a
            self.b = b
            self.c = c

        def membership_function(self, x):
            if x <= self.a:
                result =  0
            elif x <= self.b:
                result =  (x - self.a) / (self.b - self.a)
            elif x <= self.c:
                result =  (self.c - x) / (self.c - self.b)
            else:
                result =  0
            
            print(f"Triangular MF: x={x}, result={result}")
            return result
        
        def find_alpha_cut(self, alpha):
            if alpha == 0:
                return self.a, self.c
            elif alpha == 1:
                return self.b, self.b
            else:
                x1 = self.a + (self.b - self.a) * alpha
                x2 = self.c - (self.c - self.b) * alpha
                return x1, x2

        def get_membership_function_params(self):
            return self.a, self.b, self.c

    class TrapezoidalMembershipFunction:
        required_params = ['a', 'b', 'c', 'd']

        def __init__(self, name, a, b, c, d):
            self.name = name
            self.a = a
            self.b = b
            self.c = c
            self.d = d

        def membership_function(self, x):
            if x <= self.a:
                result =  0
            elif x <= self.b:
                result =  (x - self.a) / (self.b - self.a)
            elif x <= self.c:
                result =  1
            elif x <= self.d:
                result =  (self.d - x) / (self.d - self.c)
            else:
                result =  0
            print(f"Trapezoidal MF: x={x}, result={result}")
            return result
        
        def find_alpha_cut(self, alpha):
            if alpha == 0:
                result =  self.a, self.d
            elif alpha == 1:
                result =  self.b, self.c
            else:
                x1 = self.a + (self.b - self.a) * alpha
                x2 = self.d - (self.d - self.c) * alpha
                result =  x1, x2
            return result
        
        def get_membership_function_params(self):
            return self.a, self.b, self.c, self.d
    
    class GaussianMembershipFunction:
        required_params = ['mean', 'stddev']
        def __init__(self, name, mean, stddev):
            self.name = name
            self.mean = mean
            self.stddev = stddev
        
        def membership_function(self, x):
            #print(f"Gaussian MF: x={x}, mean={self.mean}, stddev={self.stddev}")
            result =  math.exp(-((x - self.mean) ** 2) / (2 * self.stddev ** 2))
            return result
        
        def find_alpha_cut(self, alpha):
            if alpha == 0:
                return -3*self.stddev + self.mean, 3*self.stddev + self.mean
            else:
                x1 = self.mean - math.sqrt(-2 * self.stddev ** 2 * math.log(alpha))
                x2 = self.mean + math.sqrt(-2 * self.stddev ** 2 * math.log(alpha))
                return x1, x2
        
        def get_membership_function_params(self):
            return self.mean, self.stddev
            
    class SigmoidalMembershipFunction:
        required_params = ['a', 'c']
        def __init__(self, name, a, c):
            self.name = name
            self.a = a
            self.c = c
        
        def membership_function(self, x):
            result =  1 / (1 + np.exp(-self.a * (x - self.c)))
            print(f"Sigmoidal MF: x={x}, result={result}")
            return result

        def find_alpha_cut(self, alpha):
            if alpha == 0:
                return -float('inf'), float('inf')
            elif alpha == 1:
                return self.c, self.c
            else:
                x1 = self.c - math.log((1 - alpha) / max(alpha, 1e-10)) / self.a
                x2 = self.c + math.log(alpha / max(1 - alpha, 1e-10)) / self.a
                return x1, x2
        
        def get_membership_function_params(self):
            return self.a, self.c
            
    class BellShapedMembershipFunction:
        required_params = ['a', 'b', 'c']
        def __init__(self, name, a, b, c):
            self.name = name
            self.a = a
            self.b = b
            self.c = c

        def membership_function(self, x):
            result =  1 / (1 + ((x - self.c) / self.a) ** (2 * self.b))
            print(f"BellShaped MF: x={x}, result={result}")
            return result 
        
        def find_alpha_cut(self, alpha):
            if alpha == 0:
                return self.c - 10, self.c + 10
            else:
                x1 = self.c - math.sqrt((1 / alpha - 1) ** (1 / self.b)) * self.a
                x2 = self.c + math.sqrt((1 / alpha - 1) ** (1 / self.b)) * self.a
                return x1, x2
        
        def get_membership_function_params(self):
            return self.a, self.b, self.c
            
    class ZShapedMembershipFunction:
        required_params = ['a', 'b']
        def __init__(self, name, a, b):
            self.name = name
            self.a = a
            self.b = b

        def membership_function(self, x):
            if x <= self.a:
                result =  1
            elif x <= self.b:
                result =  (self.b - x) / (self.b - self.a)
            else:
                result =  0
            print(f"ZShaped MF: x={x}, result={result}")
            return result
        
        def find_alpha_cut(self, alpha):
            if alpha == 0:
                return self.a, self.b
            elif alpha == 1:
                return self.b, self.b
            else:
                x1 = self.b - (self.b - self.a) * alpha
                x2 = self.b
                return x1, x2
                
        def get_membership_function_params(self):
            return self.a, self.b
            
    class SShapedMembershipFunction:
        required_params = ['a', 'b']
        def __init__(self, name, a, b):
            self.name = name
            self.a = a
            self.b = b

        def membership_function(self, x):
            if x <= self.a:
                result =  0
            elif x <= (self.a + self.b) / 2:
                result =  2 * ((x - self.a) / (self.b - self.a)) ** 2
            elif x <= self.b:
                result =  1 - 2 * ((self.b - x) / (self.b - self.a)) ** 2
            else:
                result =  1
            print(f"SShaped MF: x={x}, result={result}")
            return result 
        
        def find_alpha_cut(self, alpha):
            if alpha == 0:
                return self.a, self.b
            elif alpha == 1:
                return self.a, self.a
            else:
                x1 = self.a
                x2 = self.a + (self.b - self.a) * alpha
                return x1, x2
        
        def get_membership_function_params(self):
            return self.a, self.b
            
    class LeftShoulderMembershipFunction:
        required_params = ['a', 'b']
        def __init__(self, name, a, b):
            self.name = name
            self.a = a
            self.b = b

        def membership_function(self, x):
            if x <= self.a:
                result =  1
            elif x <= self.b:
                result =  (self.b - x) / (self.b - self.a)
            else:
                result =  0
            print(f"LeftShoulder MF: x={x}, result={result}")
            return result
        
        def find_alpha_cut(self, alpha):
            if alpha == 0:
                return self.a, self.b
            elif alpha == 1:
                return self.a, self.a
            else:
                x1 = self.a
                x2 = self.a + (self.b - self.a) * alpha
                return x1, x2
        
        def get_membership_function_params(self):
            return self.a, self.b
            
    class RightShoulderMembershipFunction:
        required_params = ['a', 'b']
        def __init__(self, name, a, b):
            self.name = name
            self.a = a
            self.b = b

        def membership_function(self, x):
            if x <= self.a:
                result =  0
            elif x <= self.b:
                result =  (x - self.a) / (self.b - self.a)
            else:
                result =  1
            print(f"RightShoulder MF: x={x}, result={result}")
            return result
            
        def find_alpha_cut(self, alpha):
            if alpha == 0:
                return self.a, self.b
            elif alpha == 1:
                return self.b, self.b
            else:
                x1 = self.a + (self.b - self.a) * (1 - alpha)
                x2 = self.b
                return x1, x2
        
        def get_membership_function_params(self):
            return self.a, self.b
            
    def define_membership_functions(self, params):
        for param in params:
            mf_class = {
                'triangular': self.TriangularMembershipFunction,
                'trapezoidal': self.TrapezoidalMembershipFunction,
                'gaussian': self.GaussianMembershipFunction,
                'sigmoidal': self.SigmoidalMembershipFunction,
                'bell_shaped': self.BellShapedMembershipFunction,
                'z_shaped': self.ZShapedMembershipFunction,
                's_shaped': self.SShapedMembershipFunction,
                'right_shoulder': self.RightShoulderMembershipFunction,
                'left_shoulder': self.LeftShoulderMembershipFunction
            }.get(param['type'])

            if not mf_class:
                raise ValueError(f"Unknown membership function type: {param['type']}")

            mf = mf_class(param['name'], *[param.get(p) for p in mf_class.required_params])
            domain = self.calculate_domain(mf, param)
            fuzzy_set = self.FuzzySet(param['name'], mf, domain)
            fuzzy_set.possibility_distribution = param.get('possibility_distribution', lambda x: 1)
            self.membership_functions[param['name']] = fuzzy_set
            print(f"Defined membership function: {param['name']}, type: {param['type']}, params: {param}")
    
    def apply_possibility_distribution(self, x, param):
        possibility_distribution = param.get('possibility_distribution')
        if possibility_distribution:
            return eval(possibility_distribution)
        else:
            return 1
        
    def calculate_domain(self, mf, param):
        if isinstance(mf, self.TriangularMembershipFunction):
            return (param.get('a'), param.get('c'))
        elif isinstance(mf, self.TrapezoidalMembershipFunction):
            return (param.get('a'), param.get('d'))
        elif isinstance(mf, self.GaussianMembershipFunction):
            mean, stddev = param.get('mean'), param.get('stddev')
            return (mean - 3 * stddev, mean + 3 * stddev)
        elif isinstance(mf, self.BellShapedMembershipFunction):
            a, b, c = param.get('a'), param.get('b'), param.get('c')
            return (c - 10 * a, c + 10 * a)
        elif isinstance(mf, self.SigmoidalMembershipFunction):
            return (param.get('c') - 10, param.get('c') + 10)
        elif isinstance(mf, self.ZShapedMembershipFunction):
            return (param.get('a'), param.get('b'))
        elif isinstance(mf, self.SShapedMembershipFunction):
            return (param.get('a'), param.get('b'))
        elif isinstance(mf, self.LeftShoulderMembershipFunction):
            return (param.get('a'), param.get('b'))
        elif isinstance(mf, self.RightShoulderMembershipFunction):
            return (param.get('a'), param.get('b'))
        else:
            return (0, 1)
        
    def calculate_alpha_cuts(self):
        self.alpha_cuts = {}
        for param, fuzzy_set in self.membership_functions.items():
            self.alpha_cuts[param] = []
            for alpha in self.alpha_levels:
                lower, upper = fuzzy_set.find_alpha_cut(alpha)
                mean = (lower + upper) / 2
                range_ = upper - lower
                self.alpha_cuts[param].append((lower, upper, alpha, mean, range_))
                print(f"Alpha cut for {param} at alpha {alpha}: ({lower}, {upper}, mean={mean}, range={range_})")
    
    def calculate_emissions(self, config_sample, iteration):
        network = Network(config_sample)
        network.build_network()
        network.simulate()
        subsystem_emissions, total_emissions = network.calculate_emissions()
        #print(f"Iteration: {iteration}, Total Emissions: {total_emissions}, Subsystem Emissions: {subsystem_emissions}")
        return total_emissions, subsystem_emissions

    def update_config(self, config_lower, config_upper, main_param, sub_param, lower, upper):
        # Traverse to the correct location in the config
        value_lower = config_lower
        value_upper = config_upper
        for part in main_param.split('.'):
            if isinstance(value_lower, list):
                index = int(part)
                value_lower = value_lower[index]
                value_upper = value_upper[index]
            else:
                value_lower = value_lower.get(part, None)
                value_upper = value_upper.get(part, None)
            if value_lower is None or value_upper is None:
                return

        # Check if the final part is a list or a dictionary
        if isinstance(value_lower, dict):
            value_lower[sub_param] = lower
            value_upper[sub_param] = upper
        elif isinstance(value_lower, list):
            index = int(sub_param)
            value_lower[index] = lower
            value_upper[index] = upper
        else:
            raise ValueError(f"Unsupported structure for parameter: {main_param}.{sub_param}")
        
    def defuzzify(self, fuzzy_set, method='centroid'):
        if isinstance(fuzzy_set, list):
            fuzzy_set = self.create_fuzzy_set_from_aggregated(fuzzy_set)
            
        if method == 'centroid':
            return self.centroid(fuzzy_set)
        elif method == 'bisector':
            return self.bisector(fuzzy_set)
        elif method == 'mom':
            return self.mean_of_maximum(fuzzy_set)
        elif method == 'som':
            return self.smallest_of_maximum(fuzzy_set)
        elif method == 'lom':
            return self.largest_of_maximum(fuzzy_set)
        else:
            raise ValueError("Invalid defuzzification method")

    def create_fuzzy_set_from_aggregated(self, aggregated_outputs):
        def combined_membership_function(x):
            for output, alpha in aggregated_outputs:
                if x == output:
                    return alpha
            return 0
        
        domain = (min(output for output, _ in aggregated_outputs), max(output for output, _ in aggregated_outputs))
        new_mf_obj = self.CombinedMembershipFunction(combined_membership_function)
        return self.FuzzySet("AggregatedOutput", new_mf_obj, domain)
    
    def centroid(self, fuzzy_set):
        domain_start, domain_end = fuzzy_set.domain[0], fuzzy_set.domain[1]
        values = np.linspace(domain_start, domain_end, 100)
        membership_degrees = [fuzzy_set.membership_function_obj.membership_function(value) for value in values]
        numerator = sum(value * degree for value, degree in zip(values, membership_degrees))
        denominator = sum(membership_degrees)

        print(f"Centroid method: Values: {values}")
        print(f"Centroid method: Membership Degrees: {membership_degrees}")
        print(f"Centroid method: Numerator: {numerator}")
        print(f"Centroid method: Denominator: {denominator}")

        return numerator / denominator if denominator != 0 else 0
    
    def bisector(self, fuzzy_set):
        values = np.linspace(fuzzy_set.domain[0], fuzzy_set.domain[1], 100)
        membership_degrees = [fuzzy_set.membership_function_obj.membership_function(value) for value in values]
        total_membership_degree = sum(membership_degrees)
        cumulative_membership_degree = 0
        for value, degree in zip(values, membership_degrees):
            cumulative_membership_degree += degree
            if cumulative_membership_degree >= total_membership_degree / 2:
                return value
        return values[-1]

    def mean_of_maximum(self, fuzzy_set):
        values = np.linspace(fuzzy_set.domain[0], fuzzy_set.domain[1], 100)
        membership_degrees = [fuzzy_set.membership_function_obj.membership_function(value) for value in values]
        max_degree = max(membership_degrees)
        max_values = [value for value, degree in zip(values, membership_degrees) if degree == max_degree]
        return sum(max_values) / len(max_values) if max_values else 0

    def smallest_of_maximum(self, fuzzy_set):
        values = np.linspace(fuzzy_set.domain[0], fuzzy_set.domain[1], 100)
        membership_degrees = [fuzzy_set.membership_function_obj.membership_function(value) for value in values]
        max_degree = max(membership_degrees)
        max_values = [value for value, degree in zip(values, membership_degrees) if degree == max_degree]
        return min(max_values) if max_values else 0

    def largest_of_maximum(self, fuzzy_set):
        values = np.linspace(fuzzy_set.domain[0], fuzzy_set.domain[1], 100)
        membership_degrees = [fuzzy_set.membership_function_obj.membership_function(value) for value in values]
        max_degree = max(membership_degrees)
        max_values = [value for value, degree in zip(values, membership_degrees) if degree == max_degree]
        return max(max_values) if max_values else 0

    def fuzzy_arithmetic_operation(self, op, *args):
        fuzzy_sets = []
        alpha_levels = []
        for i, arg in enumerate(args):
            (alpha_cut, alpha_level, mf_obj) = arg
            lower, upper = alpha_cut
            fuzzy_set = self.FuzzySet(f"A{i}", mf_obj, [lower, upper])
            fuzzy_sets.append(fuzzy_set)
            alpha_levels.append(alpha_level)

        result = fuzzy_sets[0]
        for fuzzy_set in fuzzy_sets[1:]:
            if op == 'multiply':
                print(f"Before multiply: Result domain: {result.domain}, Fuzzy set domain: {fuzzy_set.domain}")
                result = result.fuzzy_multiply(fuzzy_set)
                print(f"After multiply: Result domain: {result.domain}")
            elif op == 'add':
                print(f"Before add: Result domain: {result.domain}, Fuzzy set domain: {fuzzy_set.domain}")
                result = result.fuzzy_add(fuzzy_set)
                print(f"After add: Result domain: {result.domain}")
            else:
                raise ValueError("Invalid fuzzy arithmetic operation")

        return result, alpha_levels
    
    def uncertainty_propagation(self, fuzzy_set, operation, other_fuzzy_set=None, alpha=0.5):
        if other_fuzzy_set is None:
            other_fuzzy_set = fuzzy_set

        if operation == "intersection":
            lower_bound, upper_bound = fuzzy_set.intersection(other_fuzzy_set, alpha)
            # Adjust bounds for complex interactions generically
            lower_bound *= 0.95  # Adjust bounds for complex interactions
            upper_bound *= 1.05
            return (lower_bound + upper_bound) / 2
        elif operation == "union":
            lower_bound, upper_bound = fuzzy_set.union(other_fuzzy_set, alpha)
            # Adjust bounds for complex interactions generically
            lower_bound *= 0.95
            upper_bound *= 1.05
            return (lower_bound + upper_bound) / 2
        else:
            raise ValueError("Invalid operation")
        
    def create_z_number(self, name, A, B):
        self.z_numbers[name] = {"A": A, "B": B}

    def get_z_number(self, name):
        return self.z_numbers.get(name)

    def calculate_certainty_degree(self, z_number_name, x):
        z_number = self.get_z_number(z_number_name)
        A = z_number["A"]
        B = z_number["B"]
        
        membership_degree = A.membership_function_obj.membership_function(x)
        
        if isinstance(B, self.FuzzySet):
            certainty_degree = fuzzy_and(membership_degree, B.membership_function_obj.membership_function(x))
        elif callable(B):
            certainty_degree = B(x)
        else:
            raise ValueError("Invalid certainty degree B")
        
        return certainty_degree
    
    def propagate_uncertainty(self):
        self.output_alpha_cuts = []
        uncertainty_propagation_values = []

        for alpha_level in self.alpha_levels:
            input_ranges = {}
            for param, cuts in self.alpha_cuts.items():
                for cut in cuts:
                    if cut[2] == alpha_level:
                        lower, upper, alpha, mean, range_ = cut
                        input_ranges[param] = (lower, upper, mean, range_)

            config_lower = copy.deepcopy(self.config)
            config_upper = copy.deepcopy(self.config)

            for param, (lower, upper, mean, range_) in input_ranges.items():
                param_parts = param.split('.')
                main_param = '.'.join(param_parts[:-1])
                sub_param = param_parts[-1]

                self.update_config(config_lower, config_upper, main_param, sub_param, lower, upper)

            mean_lower_emissions, _ = self.calculate_emissions(config_lower, iteration=1)
            mean_upper_emissions, _ = self.calculate_emissions(config_upper, iteration=1)

            if mean_upper_emissions != mean_lower_emissions:
                mf_obj = self.membership_functions[list(self.membership_functions.keys())[0]].membership_function_obj  # Just to avoid key error
                self.output_alpha_cuts.append(((mean_lower_emissions, mean_upper_emissions), alpha_level, mf_obj))
                uncertainty_propagation_values.append(mean_upper_emissions - mean_lower_emissions)

        self.final_uncertainty_propagation = max(uncertainty_propagation_values) if uncertainty_propagation_values else 0.0
        
    def calculate_output(self, crisp_input, param_name):
        membership_function = self.membership_functions.get(param_name)
        
        if not membership_function:
            raise ValueError(f"Membership function for parameter '{param_name}' not found.")
        
        membership_degree = membership_function.membership_function_obj.membership_function(crisp_input)
        output_value = membership_degree * crisp_input
        
        print(f"Calculated output for parameter '{param_name}' with crisp input {crisp_input}: {output_value}")
        
        return output_value

    def calculate_crisp_input(self, alpha_cuts, param_name):
        membership_function = self.membership_functions[param_name]
        domain = membership_function.domain
        lower_bound = domain[0]
        upper_bound = domain[1]
        crisp_input = lower_bound + (upper_bound - lower_bound) * alpha_cuts[0] / alpha_cuts[1]
        return crisp_input
    
    def normalize_params(self, param, lower, upper):
        mean = (lower + upper) / 2
        range_ = upper - lower
        if range_ == 0:
            logging.warning(f"Warning: Range is zero for param: {param}, lower: {lower}, upper: {upper}. Setting range to 1 to avoid division by zero.")
            range_ = 1
        norm_param = (param - mean) / range_
        logging.debug(f"Normalized param: {param}, lower: {lower}, upper: {upper}, normalized: {norm_param}")
        return norm_param, mean, range_

    def denormalize_params(self, normalized_value, mean, range_):
        denorm_param = normalized_value * range_ + mean
        logging.debug(f"Denormalized value: {denorm_param} from normalized: {normalized_value}, mean: {mean}, range: {range_}")
        return denorm_param

    def run_possibilistic_uncertainty_analysis(self):
        logging.info("Starting possibilistic uncertainty analysis")
        self.define_membership_functions(self.uncertain_params)
        self.calculate_alpha_cuts()

        for param in self.uncertain_params:
            membership_function = self.membership_functions[param["name"]]
            self.create_z_number(param["name"], membership_function, param.get("possibility_distribution", lambda x: 1))

        self.propagate_uncertainty()

        if not self.output_alpha_cuts:
            raise ValueError("No alpha cuts were generated. Check the calculation of alpha cuts.")

        results = {}
        global_max_uncertainty = 0
        for param in self.uncertain_params:
            param_name = param["name"]
            if param_name in self.alpha_cuts:
                param_uncertainties = [upper - lower for (lower, upper, _, _, _) in self.alpha_cuts[param_name]]
                if param_uncertainties:
                    max_uncertainty = max(param_uncertainties)
                    global_max_uncertainty = max(global_max_uncertainty, max_uncertainty)

        if global_max_uncertainty == 0:
            logging.warning("Warning: global_max_uncertainty is zero. Setting it to 1 to avoid division by zero.")
            global_max_uncertainty = 1

        for method in self.defuzzification_methods:
            if not self.output_alpha_cuts:
                logging.warning("Warning: No output alpha cuts available for fuzzy arithmetic operation.")
                continue

            fuzzy_outputs_alpha = {}
            for param in self.uncertain_params:
                param_name = param["name"]
                for (alpha_cut, alpha, _) in self.output_alpha_cuts:
                    crisp_input = self.calculate_crisp_input(alpha_cut, param_name)
                    fuzzy_output = self.calculate_output(crisp_input, param_name)
                    if alpha not in fuzzy_outputs_alpha:
                        fuzzy_outputs_alpha[alpha] = []
                    fuzzy_outputs_alpha[alpha].append(fuzzy_output)

            fuzzy_outputs_aggregated = []
            for alpha, outputs in fuzzy_outputs_alpha.items():
                aggregated_output = max(outputs)
                fuzzy_outputs_aggregated.append((aggregated_output, alpha))

            crisp_outputs = self.defuzzify(fuzzy_outputs_aggregated, method=method)

            parameter_uncertainties = {}
            for param in self.uncertain_params:
                param_name = param["name"]
                if param_name in self.alpha_cuts:
                    param_uncertainties = [upper - lower for (lower, upper, _, _, _) in self.alpha_cuts[param_name]]
                    normalized_uncertainties = [unc / global_max_uncertainty for unc in param_uncertainties]
                    parameter_uncertainties[param_name] = np.mean(normalized_uncertainties)

            results[method] = {
                "crisp_outputs": crisp_outputs,
                "uncertainty_propagation": self.final_uncertainty_propagation,
                "parameter_uncertainties": parameter_uncertainties,
            }
        return results

    def visualize_uncertainty(self, crisp_outputs, uncertainty_propagation, parameter_uncertainties):
        print(f"Total Uncertainty Propagation: {uncertainty_propagation}")

        plt.figure(figsize=(12, 8))
        bars = plt.bar(parameter_uncertainties.keys(), parameter_uncertainties.values(), color='lightcoral')
        plt.xlabel('Parameters')
        plt.ylabel('Uncertainty')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        for bar, unc in zip(bars, parameter_uncertainties.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00000005, f'{unc:.2e}', ha='center', va='bottom', rotation=90)
        plt.title('Uncertainty of Parameters')
        plt.show()

        alpha_levels = [alpha_level for ((_, _), alpha_level, _) in self.output_alpha_cuts]
        lower_bounds = [lower for ((lower, _), _, _) in self.output_alpha_cuts]
        upper_bounds = [upper for ((_, upper), _, _) in self.output_alpha_cuts]

        plt.figure(figsize=(10, 6))
        plt.fill_between(alpha_levels, lower_bounds, upper_bounds, color='skyblue', alpha=0.5)
        plt.xlabel('Alpha Level')
        plt.ylabel('Emission Range')
        plt.title('Possibilistic Uncertainty Analysis')
        plt.grid(True)
        plt.show()
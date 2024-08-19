from importer import *

from network import Network

class PossibilisticUncertaintyAnalysis:
    def __init__(self, model, config, input_parameters, defuzzification_methods):
        
        ##  Initialises Possibilistic UA class with the given model, configuration, input params and defuzzification methods
        ##  Sets up necessary attributes for performing the analysis

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
            
            ##  Represents a fuzzy set with a specific MF and domain

            self.name = name
            self.membership_function_obj = membership_function_obj
            self.domain = domain

        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut of fuzzy set at the specified alpha level

            return self.membership_function_obj.find_alpha_cut(alpha)

        def intersection(self, other, alpha):

            ##  Calculates intersection of this fuzzy set with another fuzzy set at specified alpha level

            x1, x2 = self.find_alpha_cut(alpha)
            other_x1, other_x2 = other.find_alpha_cut(alpha)
            lower_bound = max(x1, other_x1)
            upper_bound = min(x2, other_x2)
            
            return lower_bound, upper_bound

        def union(self, other, alpha):

            ##  Calculates union of this fuzzy set with another fuzzy set at the specified alpha level

            x1, x2 = self.find_alpha_cut(alpha)
            other_x1, other_x2 = other.find_alpha_cut(alpha)
            lower_bound = min(x1, other_x1)
            upper_bound = max(x2, other_x2)

            return lower_bound, upper_bound
        
        def fuzzy_multiply(self, other):

            ##  Multiplies this fuzzy set with another fuzzy set: combines their MFs

            new_domain = (max(self.domain[0], other.domain[0]), min(self.domain[1], other.domain[1]))

            def combined_membership_function(x):
                if new_domain[0] <= x <= new_domain[1]:
                    
                    # Calculate crisp input values for each fuzzy set
                    self_x = self.domain[0] + (self.domain[1] - self.domain[0]) * (x - new_domain[0]) / (new_domain[1] - new_domain[0])
                    other_x = other.domain[0] + (other.domain[1] - other.domain[0]) * (x - new_domain[0]) / (new_domain[1] - new_domain[0])

                    # Calculate membership degrees for each fuzzy set
                    result_self = self.membership_function_obj.membership_function(self_x)
                    result_other = other.membership_function_obj.membership_function(other_x)

                    # Calculate combined membership degree using the minimum operator
                    result = min(result_self, result_other)

                    return result
                else:
                    return 0

            new_mf_obj = PossibilisticUncertaintyAnalysis.CombinedMembershipFunction(combined_membership_function)

            self_params = self.membership_function_obj.get_membership_function_params() if hasattr(self.membership_function_obj, 'get_membership_function_params') else 'Combined'
            other_params = other.membership_function_obj.get_membership_function_params() if hasattr(other.membership_function_obj, 'get_membership_function_params') else 'Combined'

            # Adjust membership properties based on the actual x values being processed
            new_params = (new_domain[0], new_domain[1], (new_domain[1] - new_domain[0]) / (self.domain[1] - self.domain[0]), self.membership_function_obj.get_membership_function_params()[1])
            new_domain = (new_domain[0], new_domain[1], self.domain[0], self.domain[1])
            
            return PossibilisticUncertaintyAnalysis.FuzzySet(f"{self.name} * {other.name}", new_mf_obj, new_domain, new_params)
        
        def fuzzy_add(self, other):

            ##  Adds this fuzzy set with another fuzzy set: combines their MFs

            new_domain = (min(self.domain[0], other.domain[0]), max(self.domain[1], other.domain[1]))

            def combined_membership_function(x):
                if new_domain[0] <= x <= new_domain[1]:
                    result_self = self.membership_function_obj.membership_function(x)
                    result_other = other.membership_function_obj.membership_function(x)
                    result = max(result_self, result_other)
                    
                    return result
                else:
                    return 0

            new_mf_obj = PossibilisticUncertaintyAnalysis.CombinedMembershipFunction(combined_membership_function)

            return PossibilisticUncertaintyAnalysis.FuzzySet(f"{self.name} + {other.name}", new_mf_obj, new_domain)

    class CombinedMembershipFunction:
        def __init__(self, combined_function):

            ##  Represents a combined MF created from multiple fuzzy sets

            self.combined_function = combined_function

        def membership_function(self, x):

            ##  Evaluates MF at a specific point

            return self.combined_function(x)
        
        def get_membership_function_params(self):
            
            return "Combined"
        
    class TriangularMembershipFunction:
        required_params = ['a', 'b', 'c']

        def __init__(self, name, a, b, c):

            ##  Represents triangular MF

            self.name = name
            self.a = a
            self.b = b
            self.c = c

        def membership_function(self, x):

            ##  Evaluates triangular MF at a specific point

            if x <= self.a:
                result =  0
            elif x <= self.b:
                result =  (x - self.a) / (self.b - self.a)
            elif x <= self.c:
                result =  (self.c - x) / (self.c - self.b)
            else:
                result =  0
            
            return result
        
        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut for the triangular MF

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

            ##  Represents a trapezoidal MF

            self.name = name
            self.a = a
            self.b = b
            self.c = c
            self.d = d

        def membership_function(self, x):

            ##  Evaluates the trapezoidal MF at a specific point
           
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
            
            return result
        
        def find_alpha_cut(self, alpha):
            
            ##  Finds alpha-cut for the trapezoidal MF

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

            ##  Respresents a gaussian MF

            self.name = name
            self.mean = mean
            self.stddev = stddev
        
        def membership_function(self, x):

            ##  Evaluates the gaussian MF at a specific point

            result =  math.exp(-((x - self.mean) ** 2) / (2 * self.stddev ** 2))
            
            return result
        
        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut for the guassian MF

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

            ##  Represents a sigmoidal MF

            self.name = name
            self.a = a
            self.c = c
        
        def membership_function(self, x):

            ##  Evaluates the sigmoidal MF at a specific point

            result =  1 / (1 + np.exp(-self.a * (x - self.c)))

            return result

        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut for the sigmoidal MF

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

            ## Represents a bell-shaped MF

            self.name = name
            self.a = a
            self.b = b
            self.c = c

        def membership_function(self, x):

            ##  Evaluates the bell-shaped MF at a specific point

            result =  1 / (1 + ((x - self.c) / self.a) ** (2 * self.b))

            return result 
        
        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut for the bell-shaped MF

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

            ## Represents a z-shaped MF

            self.name = name
            self.a = a
            self.b = b

        def membership_function(self, x):

            ##  Evaluates the z-shaped MF at a specific point

            if x <= self.a:
                result =  1
            elif x <= self.b:
                result =  (self.b - x) / (self.b - self.a)
            else:
                result =  0
            return result
        
        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut for the z-shaped MF

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

            ## Represents a s-shaped MF

            self.name = name
            self.a = a
            self.b = b

        def membership_function(self, x):

            ##  Evaluates the s-shaped MF at a specific point

            if x <= self.a:
                result =  0
            elif x <= (self.a + self.b) / 2:
                result =  2 * ((x - self.a) / (self.b - self.a)) ** 2
            elif x <= self.b:
                result =  1 - 2 * ((self.b - x) / (self.b - self.a)) ** 2
            else:
                result =  1
            return result 
        
        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut for the s-shaped MF

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

            ##  Represents a left-shoulder MF
            self.name = name
            self.a = a
            self.b = b

        def membership_function(self, x):

            ##  Evaluates the left-shoulder MF at a specific point

            if x <= self.a:
                result =  1
            elif x <= self.b:
                result =  (self.b - x) / (self.b - self.a)
            else:
                result =  0
            return result
        
        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut for the left-shoulder MF

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

            ##  Represents a right-shoulder MF

            self.name = name
            self.a = a
            self.b = b

        def membership_function(self, x):

            ##  Evaluates the right-shoulder MF at a specific point

            if x <= self.a:
                result =  0
            elif x <= self.b:
                result =  (x - self.a) / (self.b - self.a)
            else:
                result =  1
            return result
            
        def find_alpha_cut(self, alpha):

            ##  Finds alpha-cut for the right-shoulder MF

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

        ##  Defines the MFs for all uncertain parameters based on the json configuration

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
    
    def apply_possibility_distribution(self, x, param):

        ## Retrieve possibility distribution from configuration if available

        possibility_distribution = param.get('possibility_distribution')
        if possibility_distribution:
            return eval(possibility_distribution)
        else:
            return 1
        
    def calculate_domain(self, mf, param):

        ##  Calculates domain of the MF based on its type and params
        ##  Defines the range over which the MF is valid

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

        ##  Calculates alpha-cuts for all defined MFs across specified alpha levels
        ##  These determine the intervals of possible values for uncertain parameters at the different confidence levels (alpha levels)

        self.alpha_cuts = {}
        for param, fuzzy_set in self.membership_functions.items():
            self.alpha_cuts[param] = []
            for alpha in self.alpha_levels:
                lower, upper = fuzzy_set.find_alpha_cut(alpha)
                mean = (lower + upper) / 2
                range_ = upper - lower
                self.alpha_cuts[param].append((lower, upper, alpha, mean, range_))
    
    def calculate_emissions(self, config_sample, iteration):
        
        ##  Calculates the emissions for given config and iteration
        
        network = Network(config_sample)
        network.build_network()
        network.simulate()
        subsystem_emissions, total_emissions = network.calculate_emissions()
        
        return total_emissions, subsystem_emissions

    def update_config(self, config_lower, config_upper, main_param, sub_param, lower, upper):
        
        ##  Updates the configuration with new lower and upper bounds based on alpha-cut values
        ##  Modifies the dictionary directly to reflect the range of possible values or each parameter at different alpha levels
        ##  Traverse to the correct location in the config
        
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
        
        ##  Defuzzifies the fuzzy set using the specified method
        ##  Converts a fuzzy set into a single crisp value 
        ##  "Best guess" given the uncertainty modeled by the fuzzy set

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
        
        ##  Creates a fuzzy set from aggregated outputs
        ##  Used when multiple fuzzy results need to be combined

        def combined_membership_function(x):
            for output, alpha in aggregated_outputs:
                if x == output:
                    return alpha
            return 0
        
        domain = (min(output for output, _ in aggregated_outputs), max(output for output, _ in aggregated_outputs))
        new_mf_obj = self.CombinedMembershipFunction(combined_membership_function)
        
        return self.FuzzySet("AggregatedOutput", new_mf_obj, domain)
    
    def centroid(self, fuzzy_set):

        ##  Calculates centroid of the fuzzy set
        ##  Finds the center of gravity of the fuzzy set's area

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

        ##  Calculates bisector of the fuzzy set
        ##  Finds point that divides the fuzzy set into two regions of equal area: balancing the area on either side of this point

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

        ##  Calculates the average of the values where the MF reaches its maximum
        ##  Useful when you want to focus on peak region

        values = np.linspace(fuzzy_set.domain[0], fuzzy_set.domain[1], 100)
        membership_degrees = [fuzzy_set.membership_function_obj.membership_function(value) for value in values]
        max_degree = max(membership_degrees)
        max_values = [value for value, degree in zip(values, membership_degrees) if degree == max_degree]
        
        return sum(max_values) / len(max_values) if max_values else 0

    def smallest_of_maximum(self, fuzzy_set):
        
        ##  Selects smallest value in the domain where the MF reaches its max
        ##  Useful in conservative scenarios

        values = np.linspace(fuzzy_set.domain[0], fuzzy_set.domain[1], 100)
        membership_degrees = [fuzzy_set.membership_function_obj.membership_function(value) for value in values]
        max_degree = max(membership_degrees)
        max_values = [value for value, degree in zip(values, membership_degrees) if degree == max_degree]
        
        return min(max_values) if max_values else 0

    def largest_of_maximum(self, fuzzy_set):

        ##  Selects largest value in domain where the MF reaches its maximum
        ##  Used for more optimistic or aggressive scenarios

        values = np.linspace(fuzzy_set.domain[0], fuzzy_set.domain[1], 100)
        membership_degrees = [fuzzy_set.membership_function_obj.membership_function(value) for value in values]
        max_degree = max(membership_degrees)
        max_values = [value for value, degree in zip(values, membership_degrees) if degree == max_degree]
        
        return max(max_values) if max_values else 0

    def fuzzy_arithmetic_operation(self, op, *args):
        
        ##  Performs a fuzzy arithmetic operation (either multiplication or addition) on multiple fuzzy sets

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
                result = result.fuzzy_multiply(fuzzy_set)
            elif op == 'add':
                result = result.fuzzy_add(fuzzy_set)
            else:
                raise ValueError("Invalid fuzzy arithmetic operation")

        return result, alpha_levels
    
    def uncertainty_propagation(self, fuzzy_set, operation, other_fuzzy_set=None, alpha=0.5):
        
        ##  Propagates uncertainty by performing operations (intersection or union) on fuzzy sets
        ##  Applies a slight adjustment to the bounds
        upper_bound *= 1.05
        if other_fuzzy_set is None:
            other_fuzzy_set = fuzzy_set

        if operation == "intersection":
            lower_bound, upper_bound = fuzzy_set.intersection(other_fuzzy_set, alpha)
            lower_bound *= 0.95  
            upper_bound *= 1.05
            return (lower_bound + upper_bound) / 2
        elif operation == "union":
            lower_bound, upper_bound = fuzzy_set.union(other_fuzzy_set, alpha)
            lower_bound *= 0.95
            upper_bound *= 1.05
            return (lower_bound + upper_bound) / 2
        else:
            raise ValueError("Invalid operation")
        
    def create_z_number(self, name, A, B):

        ##  Creates a Z-number:
        ##  Fuzzy number with an associated degree of certainty
        ##  A: primary fuzzy set
        ##  B: secondary fuzzy set (certainty distribution)
        
        self.z_numbers[name] = {"A": A, "B": B}

    def get_z_number(self, name):

        ##  Retrieves a previously created Z-nº

        return self.z_numbers.get(name)

    def calculate_certainty_degree(self, z_number_name, x):
        
        ##  Calculates certainty degree for a given value based on a Z-nº
        
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

        ##  Propagates uncertainty through the system by applying alpha-cuts to the MFs and calculating resulting emissions for each configuration

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
        
        ##  Calculates output for a given crisp input using the MF associated with a param

        membership_function = self.membership_functions.get(param_name)
        
        if not membership_function:
            raise ValueError(f"Membership function for parameter '{param_name}' not found.")
        
        membership_degree = membership_function.membership_function_obj.membership_function(crisp_input)
        output_value = membership_degree * crisp_input
                
        return output_value

    def calculate_crisp_input(self, alpha_cuts, param_name):
        
        ##  Calculates crisp input corresponding to a specific alpha-cut and param name

        membership_function = self.membership_functions[param_name]
        domain = membership_function.domain
        lower_bound = domain[0]
        upper_bound = domain[1]
        crisp_input = lower_bound + (upper_bound - lower_bound) * alpha_cuts[0] / alpha_cuts[1]
        
        return crisp_input
    
    def normalise_params(self, param, lower, upper):
        
        ##  Normalises param value to a std range

        mean = (lower + upper) / 2
        range_ = upper - lower
        if range_ == 0:
            logging.warning(f"Warning: Range is zero for param: {param}, lower: {lower}, upper: {upper}. Setting range to 1 to avoid division by zero.")
            range_ = 1
        norm_param = (param - mean) / range_
        logging.debug(f"Normalised param: {param}, lower: {lower}, upper: {upper}, normalised: {norm_param}")
        
        return norm_param, mean, range_

    def denormalise_params(self, normalised_value, mean, range_):
        
        ##  Converts normalised value back to original scale

        denorm_param = normalised_value * range_ + mean
        logging.debug(f"Denormalised value: {denorm_param} from normalised: {normalised_value}, mean: {mean}, range: {range_}")
        
        return denorm_param

    def run_possibilistic_uncertainty_analysis(self):
        
        ##  Runs possibilistic UA by defining MFs, calculating alpha-cuts, propagating uncertainties and defuzzifying results
        
        logging.info("Starting Possibilistic Uncertainty Analysis")
        start_time = time.time()
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
                    normalised_uncertainties = [unc / global_max_uncertainty for unc in param_uncertainties]
                    parameter_uncertainties[param_name] = np.mean(normalised_uncertainties)
            
            end_time = time.time()
            logging.info("Simulation took {:.2f} seconds".format(end_time - start_time))

            results[method] = {
                "crisp_outputs": crisp_outputs,
                "uncertainty_propagation": self.final_uncertainty_propagation,
                "parameter_uncertainties": parameter_uncertainties,
            }

        return results

    def visualise_uncertainty(self, crisp_outputs, uncertainty_propagation, parameter_uncertainties):
        
        ##  Plot results of the possibilistic analysis: parameter uncertainties and overall uncertainty propagation

        print(f"Total Uncertainty Propagation: {uncertainty_propagation}")

        bar_color = '#8E7CC3' 
        fill_color = '#D6CCE6' 
        
        # Plot 1: Parameter Uncertainty
        plt.figure(figsize=(12, 8))
        plt.gcf().set_facecolor('white')
        
        bars = plt.barh(list(parameter_uncertainties.keys()), list(parameter_uncertainties.values()), 
                        color=bar_color, edgecolor='black', height=0.6)
        
        # Set chart labels, title, legend
        plt.xlabel('Uncertainty in Emissions [kg CO2/kg NH3]', fontsize=14, fontweight='bold', family='Arial')
        plt.ylabel('Parameters', fontsize=14, fontweight='bold', family='Arial')
        plt.title('Parameter Uncertainty - Possibilistic Route', fontsize=16, fontweight='bold', family='Arial', pad=20)

        # Customise tick parameters and grid
        plt.xticks(fontsize=12, family='Arial')
        plt.yticks(fontsize=12, family='Arial')
        plt.xlim(0, max(parameter_uncertainties.values()) * 1.1)
        plt.grid(True, which='major', axis='both', linestyle='-', linewidth=0.8, color='lightgray')        
        plt.gca().invert_yaxis()
        plt.tight_layout(pad=3)

        # Annotate the bars with their respective uncertainty values
        for bar, unc in zip(bars, parameter_uncertainties.values()):
            plt.text(bar.get_width() + max(parameter_uncertainties.values()) * 0.01, 
                    bar.get_y() + bar.get_height()/2, f'{unc:.2e}', 
                    ha='left', va='center', fontsize=10, color='black', family='Arial')

        plt.show()

        alpha_levels = [alpha_level for ((_, _), alpha_level, _) in self.output_alpha_cuts]
        lower_bounds = [lower for ((lower, _), _, _) in self.output_alpha_cuts]
        upper_bounds = [upper for ((_, upper), _, _) in self.output_alpha_cuts]

        # Plot 2: Possibilistic UA Propagation
        plt.figure(figsize=(10, 6))
        plt.fill_between(alpha_levels, lower_bounds, upper_bounds, color=fill_color, edgecolor='black', alpha=0.7)

        # Set chart labels, title, legend and grid
        plt.xlabel('Alpha Level', fontsize=14, fontweight='bold', family='Arial')
        plt.ylabel('Emission Range [kg CO2/kg NH3]', fontsize=14, fontweight='bold', family='Arial')
        plt.title('Possibilistic Uncertainty Analysis', fontsize=16, fontweight='bold', family='Arial', pad=20)
        plt.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, color='lightgray')

        plt.tight_layout(pad=3)
        plt.show()
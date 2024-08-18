import CoolProp.CoolProp as CP

# Apply a linear mixing rule between Nitrogen and Hydrogen
CP.apply_simple_mixing_rule('Nitrogen', 'Hydrogen', 'linear')  # Apply the mixing rule

class ThermoStream:
    def __init__(self):

        ##  Initialises an object representing a thermodynamic stream with various properties

        self.massflow = 0.0
        self.label = 0
        self.enthalpy = 0.0
        self.name = ""
        self.T = 0.0
        self.phase = -1
        self.Xm = 0.0
        self.mass = 0.0
        self.P = 0.0
        self.gamma = 0.0
        self.energy = 0.0
        self.rho = 0.0
        self.entropy = 0.0
        self.Cp = 0.0
        self.Cv = 0.0
        self.fluid = ""

    @classmethod
    def from_config(cls, config):

        ##  Creates an instance from a configuration dictionary

        instance = cls()
        for property_name, value in config.items():
            if hasattr(instance, f"set_{property_name}"):
                getattr(instance, f"set_{property_name}")(value)
        instance.calculate_all_properties()
        
        return instance

    def calculate_all_properties(self):

        ##  Calculates all properties of the stream using various methods in the class
        ##  Updates properties such as enthalpy, energy, density, pressure, entropy, gamma...

        self.calc_enthalpy(self)
        self.calc_energy(self)
        self.calc_rho(self)
        self.calc_P(self)
        self.calc_S(self)
        self.calc_gamma(self)
        self.calc_phase(self)
        self.calc_Xm(self)
        self.calc_rho_2phase(self)
        self.calc_S_2phase(self)
        self.calc_energy_2phase(self)
        self.calc_gamma_2phase(self)

    # Setters for various properties
    def set_massflow(self, m):
        self.massflow = m

    def set_mass(self, m):
        self.mass = m

    def set_enthalpy(self, h):
        self.enthalpy = h

    def set_label(self, ID):
        self.label = ID

    def set_name(self, streamname):
        self.name = streamname

    def set_fluid(self, fluidname):
        self.fluid = fluidname

    def set_P(self, press):
        self.P = press

    def set_T(self, temp):
        self.T = temp

    def set_gamma(self, gam):
        self.gamma = gam

    def set_energy(self, energy):
        self.energy = energy

    def set_entropy(self, entropy):
        self.entropy = entropy
    
    def set_rho(self, rho):
        self.rho = rho

    # Methods to calculate thermodynamic properties using CoolProp
    def calc_enthalpy(self):
        self.enthalpy = CP.PropsSI("H", "T", self.T, "P", self.P, self.fluid)

    def calc_energy(self):
        self.energy = self.enthalpy - self.P / self.rho

    def calc_T_from_H(self):
        self.T = CP.PropsSI("T", "H", self.enthalpy, "P", self.P, self.fluid)

    def calc_T_from_E(self):
        self.T = CP.PropsSI("T", "U", self.energy, "P", self.P, self.fluid)

    def calc_rho(self):
        self.rho = CP.PropsSI("D", "T", self.T, "P", self.P, self.fluid)

    def calc_P(self):
        self.P = CP.PropsSI("P", "T", self.T, "D", self.rho, self.fluid)

    def calc_S(self):
        self.entropy = CP.PropsSI("S", "T", self.T, "D", self.rho, self.fluid)

    def calc_gamma(self):
        self.Cp = CP.PropsSI("CPMASS", "T", self.T, "P", self.P, self.fluid)
        self.Cv = CP.PropsSI("CVMASS", "T", self.T, "P", self.P, self.fluid)
        self.gamma = self.Cp / self.Cv

    def calc_phase(self):
        self.phase = CP.PropsSI("Phase", "T", self.T, "Q", self.Xm, self.fluid)

    def calc_Xm(self):
        Hliq = CP.PropsSI("H", "Q", 0, "P", self.P, self.fluid)
        Hgas = CP.PropsSI("H", "Q", 1, "P", self.P, self.fluid)
        self.Xm = (self.enthalpy - Hliq) / (Hgas - Hliq)

    def calc_rho_2phase(self):
        self.rho = CP.PropsSI("D", "Q", self.Xm, "P", self.P, self.fluid)

    def calc_S_2phase(self):
        self.entropy = CP.PropsSI("S", "T", self.T, "Q", self.Xm, self.fluid)

    def calc_energy_2phase(self):
        self.energy = CP.PropsSI("U", "T", self.T, "Q", self.Xm, self.fluid)

    def calc_gamma_2phase(self):
        self.Cp = CP.PropsSI("CPMASS", "T", self.T, "Q", self.Xm, self.fluid)
        self.Cv = CP.PropsSI("CVMASS", "T", self.T, "Q", self.Xm, self.fluid)
        self.gamma = self.Cp / self.Cv

    # Getters for various properties
    def get_fluid(self):
        return self.fluid
    
    def get_enthalpy(self):
        return self.enthalpy

    def get_energy(self):
        return self.energy

    def get_entropy(self):
        return self.entropy

    def get_massflow(self):
        return self.massflow

    def get_mass(self):
        return self.mass

    def get_P(self):
        return self.P

    def get_T(self):
        return self.T

    def get_rho(self):
        return self.rho

    def get_gamma(self):
        return self.gamma
    
    def get_phase(self):
        return self.phase

    def mix_streams(self, s1, s2):

        ##   Mixes two streams into the current stream

        m1 = s1.get_massflow()
        m2 = s2.get_massflow()
        self.massflow = m1 + m2
        self.enthalpy = (m1 * s1.get_enthalpy() + m2 * s2.get_enthalpy()) / self.massflow
        self.update_all_from_PH()

    def outlet_adiabatic_compressor(self, s1, Pratio):
        
        ##  Simulates an adiabatic compression process

        self.P = Pratio * s1.get_P()
        arg = (s1.get_gamma() - 1.0) / s1.get_gamma()
        self.T = s1.get_T() * pow(Pratio, arg)
        self.update_all_from_PT()

    def outlet_adiabatic_turbine(self, s1, Pratio):
        
        ##  Simulates adiabatic expansion process through a turbine

        rt = 1 / Pratio
        self.P = s1.get_P() * rt
        arg = (s1.get_gamma() - 1.0) / s1.get_gamma()
        self.T = s1.get_T() * pow(rt, arg)
        self.update_all_from_PT()

    def stream_heat_Pcons(self, s1, Qadd):

        ##  Simulates heat transfer at cte pressure

        self.enthalpy = s1.get_enthalpy() + Qadd / s1.get_massflow()
        self.P = s1.get_P()
        self.update_all_from_PH()

    def stream_heat_Vcons(self, s1, Qadd):

        ##  Simulates heat transfer at cte volume

        self.energy = s1.get_energy() + Qadd / s1.get_mass()
        self.rho = s1.get_rho()
        self.update_all_from_rhoE()

    def outlet_heat_exchanger(self, s1high, s1low, s2low):
        
        ##  Simulates outlet condition of heat exchanger

        DH = s1high.get_enthalpy() - s1low.get_enthalpy()
        self.enthalpy = s2low.get_enthalpy() + DH / s1high.get_massflow()
        self.P = s2low.get_P()
        self.massflow = s2low.get_massflow()
        self.update_all_from_PH()

    # Methods to update all properties based on different sets of variables
    def update_all_from_PT(self):
        self.calc_enthalpy()
        self.calc_rho()
        self.calc_gamma()
        self.calc_energy()
        self.calc_S()

    def update_all_from_PH(self):
        self.calc_T_from_H()
        self.calc_Xm()
        if 0.0 < self.Xm < 1.0:
            self.phase = 6
            self.calc_rho_2phase()
            self.calc_S_2phase()
            self.calc_energy_2phase()
            self.calc_gamma_2phase()
        else:
            self.calc_phase()
            self.calc_rho()
            self.calc_gamma()
            self.calc_energy()
            self.calc_S()

    def update_all_from_rhoE(self):
        self.calc_T_from_E()
        self.calc_P()
        self.calc_phase()
        self.calc_enthalpy()
        self.calc_gamma()
        self.calc_S()

    def print_info(self):

        ##  Prints detailed stream data: thermodynamic properties

        print("-------------------")
        print(self.name)
        print(f"Fluid: {self.fluid}")
        print(f"Label: {self.label}")
        print(f"Massflow [kg/s]: {round(self.massflow, 2)}")
        print(f"Enthalpy [J/kg]: {round(self.enthalpy, 2)}")
        print(f"Internal energy [J/kg]: {round(self.energy, 2)}")
        print(f"Entropy [J/kg K]: {round(self.entropy, 2)}")
        print(f"Pressure [Pa]: {round(self.P, 2)}")
        print(f"Temperature [K]: {round(self.T, 2)}")
        print(f"Density [kg/m3]: {round(self.rho, 2)}")
        print(f"Gamma (adb. coef): {self.gamma}")
        print(f"Phase: {self.phase}")
        if self.phase == 0:
            print("Phase: Liquid")
            print(f"Cp Cv [J/kg K]: {self.Cp} {self.Cv}")
        elif self.phase == 6:
            print(f"Phase: Two phase, X = {self.Xm}")
            print(f"Cp Cv [J/kg K]: {self.Cp} {self.Cv}")
            print("(Cp not relevant if 0<X<1)")
        elif self.phase == 2:
            print("Phase: Gas")
            print(f"Cp Cv [J/kg K]: {self.Cp} {self.Cv}")
        else:
            print("Phase not known")
        print("-------------------")

from thermostream import ThermoStream

class Stream:
    def __init__(self, fluid_name=None, flow_rate=0, temperature=298, pressure=101325, mass_flow=0, enthalpy=0, internal_energy=0, entropy=0, density=0, gamma=0, phase=0, **kwargs):
        self.input_stream = ThermoStream()
        if fluid_name:
            self.input_stream.set_fluid(fluid_name)
        if 'flow_rate' in kwargs:
            self.input_stream.set_massflow(kwargs['flow_rate'])
        else:
            self.input_stream.set_massflow(flow_rate)
        if 'temperature' in kwargs:
            self.input_stream.set_T(kwargs['temperature'])
        else:
            self.input_stream.set_T(temperature)
        if 'pressure' in kwargs:
            self.input_stream.set_P(kwargs['pressure'])
        else:
            self.input_stream.set_P(pressure)
        if 'mass_flow' in kwargs:
            self.input_stream.set_mass(kwargs['mass_flow'])
        else:
            self.input_stream.set_mass(mass_flow)
        if 'enthalpy' in kwargs:
            self.input_stream.set_enthalpy(kwargs['enthalpy'])
        else:
            self.input_stream.set_enthalpy(enthalpy)
        if 'internal_energy' in kwargs:
            self.input_stream.set_energy(kwargs['internal_energy'])
        else:
            self.input_stream.set_energy(internal_energy)
        if 'entropy' in kwargs:
            self.input_stream.set_entropy(kwargs['entropy'])
        else:
            self.input_stream.set_entropy(entropy)
        if 'density' in kwargs:
            self.input_stream.set_rho(kwargs['density'])
        else:
            self.input_stream.set_rho(density)
        if 'gamma' in kwargs:
            self.input_stream.set_gamma(kwargs['gamma'])
        else:
            self.input_stream.set_gamma(gamma)

    def get_property(self, property_name):
        if hasattr(self.input_stream, f"get_{property_name}"):
            return getattr(self.input_stream, f"get_{property_name}")()
        else:
            raise ValueError(f"Stream does not have property '{property_name}'")

    def get_all_properties(self):
        return {
            'fluid_name': self.get_property('fluid'),
            'flow_rate': self.get_property('massflow'),
            'temperature': self.get_property('T'),
            'pressure': self.get_property('P'),
            'mass_flow': self.get_property('mass'),
            'enthalpy': self.get_property('enthalpy'),
            'internal_energy': self.get_property('energy'),
            'entropy': self.get_property('entropy'),
            'density': self.get_property('rho'),
            'gamma': self.get_property('gamma'),
            'phase': self.get_property('phase')
        }

    def set_property(self, property_name, value):
        if hasattr(self.input_stream, f"set_{property_name}"):
            getattr(self.input_stream, f"set_{property_name}")(value)
        else:
            raise ValueError(f"Stream does not have property '{property_name}'")

    def calculate_all_properties(self):
        self.input_stream.update_all_from_PT()

    def mix(self, other_stream):
        self.input_stream.mix_streams(self.input_stream, other_stream.input_stream)

    def mix_streams(self, input_streams, output_stream_config):

        if len(input_streams) != 2:
            raise ValueError("Mixing requires exactly two input streams.")

        combined_massflow = sum(stream.get_property("massflow") for stream in input_streams)

        # Each input stream rate half of the combined mass flow rate
        for stream in input_streams:
            stream.flow_rate = combined_massflow / 2

        # Determine properties for the output stream (if specific properties not provided, use input streams properties)
        output_stream_args = {
            "fluid_name": output_stream_config["fluid_name"],
            "temperature": output_stream_config.get("temperature", input_streams[0].get_property("T")),
            "pressure": output_stream_config.get("pressure", input_streams[0].get_property("P")),
            "flow_rate": combined_massflow,
        }

        output_stream = Stream(**output_stream_args)
        output_stream.calculate_all_properties()
        return output_stream
    
    def split(self, fraction):
        if not (0 <= fraction <= 1):
            raise ValueError("Fraction must be between 0 and 1.")

        split_stream = Stream(self.get_property('fluid_name'))
        split_stream.set_property('massflow', self.get_property('massflow') * fraction)
        split_stream.set_property('enthalpy', self.get_property('enthalpy'))
        split_stream.set_property('P', self.get_property('P'))

        self.input_stream.set_massflow(self.get_property('massflow') * (1 - fraction))

        split_stream.update_all_from_PH()
        self.update_all_from_PH()

        return split_stream

    def heat_transfer(self, heat_transfer_rate):
        self.input_stream.stream_heat_Pcons(self.input_stream, heat_transfer_rate)

    def update_all_from_PT(self):
        self.input_stream.update_all_from_PT()

    def update_all_from_PH(self):
        self.input_stream.update_all_from_PH()

    def update_all_from_rhoE(self):
        self.input_stream.update_all_from_rhoE()

    def print_info(self):
        self.input_stream.print_info()

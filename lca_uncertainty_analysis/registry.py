import importlib

class SubsystemRegistry:
    def __init__(self):

        ##  Initialises the registry with an empty dictionary to hold subsystem classes
        ##  The dictionary maps unit types (as strings) to their corresponding classes

        self.subsystem_classes = {}

    def register_subsystem_class(self, subsystem_type, subsystem_class):
        
        ##  Registers a subsystem class with a specific type in the registry

        self.subsystem_classes[subsystem_type] = subsystem_class

    def get_subsystem_class(self, subsystem_type):
        
        ##  Retrieves the class associated with a given unit type from the registry

        return self.subsystem_classes.get(subsystem_type)

    def create_subsystem(self, subsystem_config):
        
        ##  Creates a subsystem instance based on its config

        subsystem_type = subsystem_config['type']
        subsystem_class = self.get_subsystem_class(subsystem_type)
        if subsystem_class is None:
            raise ValueError(f"Unknown subsystem type: {subsystem_type}")
        
        return subsystem_class(subsystem_config)
    
    def import_and_register_subsystem_class(self, subsystem_type, module_name, class_name):
        
         ## Dynamically imports a module to access a class from it and register it

        module = importlib.import_module(module_name)
        subsystem_class = getattr(module, class_name)
        self.register_subsystem_class(subsystem_type, subsystem_class)
        
        return subsystem_class
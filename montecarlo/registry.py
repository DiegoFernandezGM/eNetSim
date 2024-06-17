import importlib

class SubsystemRegistry:
    def __init__(self):
        self.subsystem_classes = {}

    def register_subsystem_class(self, subsystem_type, subsystem_class):
        self.subsystem_classes[subsystem_type] = subsystem_class

    def get_subsystem_class(self, subsystem_type):
        return self.subsystem_classes.get(subsystem_type)

    def create_subsystem(self, subsystem_config):
        subsystem_type = subsystem_config['type']
        subsystem_class = self.get_subsystem_class(subsystem_type)
        if subsystem_class is None:
            raise ValueError(f"Unknown subsystem type: {subsystem_type}")
        return subsystem_class(subsystem_config)
    
    def import_and_register_subsystem_class(self, subsystem_type, module_name, class_name):
        module = importlib.import_module(module_name)
        subsystem_class = getattr(module, class_name)
        self.register_subsystem_class(subsystem_type, subsystem_class)
        return subsystem_class
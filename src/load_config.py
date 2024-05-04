import yaml

def load_constants_from_yaml(file_path):
    with open(file_path, 'r') as file:
        constants = yaml.safe_load(file)
    return constants
import yaml

def load_yaml(filename):
    """Load a yaml file as a dictionary.

    :param filename: A filepath to a yaml file.
    :type filename: str

    :return: The contents of the yaml as a dictionary.
    :rtype: dict
    """

    with open(filename, 'r') as file:
        return yaml.safe_load(file)
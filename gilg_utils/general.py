import yaml
import pickle
import json

def load_yaml(filename: str):
    """Load a yaml file as a dictionary.

    :param filename: A filepath to a yaml file.
    :type filename: str

    :return: The contents of the yaml as a dictionary.
    :rtype: dict
    """

    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def load_pickle(filename: str):
    """Load object from a .pkl filepath.

    :param filename: A filepath to a .pkl file.
    :type filename: str

    :return: The object saved in the pickle.
    :rtype: dict
    """

    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_pickle(obj, filename: str):
    """Save object to a .pkl filepath.

    :param obj: A python object to save to the .pkl.
    :type obj: object
    
    :param filename: A filepath to a .pkl file.
    :type filename: str

    :return: None
    :rtype: None
    """

    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_json(filename: str):
    """Load object from a .json filepath.

    :param filename: A filepath to a .json file.
    :type filename: str

    :return: The object saved in the pickle.
    :rtype: dict
    """

    with open(filename, 'r') as file:
        return json.load(file)

def save_json(obj, filename: str):
    """Save object to a .json filepath. JSON is a text format so this is only recommended for string, string lists, and string dictionaries.

    :param obj: A python object to save to the .json.
    :type obj: object
    
    :param filename: A filepath to a .json file.
    :type filename: str

    :return: None
    :rtype: None
    """

    with open(filename, 'w') as file:
        json.dump(obj, file, ensure_ascii=False)
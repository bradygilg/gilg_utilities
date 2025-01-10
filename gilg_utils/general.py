import yaml
import pickle
import json
import pandas as pd

def maxdisplay(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df)
    
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

def add_jane_multicolumn(input_df,label_column='responder_6'):
    """Add multicolumn index to input_df used for jane street kaggle competition."""
    column_categories = []
    for c in input_df.columns:
        if '_id' in c:
            category = 'Key'
        elif 'feature' in c:
            category = 'Data'
        elif c==label_column:
            category = 'Label'
        else:
            category = 'Meta'
        column_categories.append(category)

    input_df = input_df.fillna(0)
    input_df.columns = pd.MultiIndex.from_arrays([column_categories,input_df.columns],names=('Category','Column'))
    return input_df

def impute_columns(input_df,model):
    """Impute missing columns with zero."""
    input_df = input_df.copy()
    f_columns = input_df['Data'].columns
    for f in model.features:
        if f not in f_columns:
            input_df[('Data',f)] = 0
    return input_df

def select_multicolumns(df,category,column_list):
    """Select a set of columns from a category of a multiindex dataframe."""
    categories = df.columns.get_level_values('Category')
    columns = df.columns.get_level_values('Column')
    keep_mask = (categories!=category) | (columns.isin(column_list))
    return df.loc[:,keep_mask]
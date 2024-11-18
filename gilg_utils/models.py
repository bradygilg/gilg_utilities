import numpy as np
import pandas as pd
from os import path, makedirs
from gilg_utils.general import save_json, load_json, save_pickle, load_pickle

class Model:
    """Top level base class that implements generic methods inherited by all other model classes."""

    def __init__(self, *args, **kwargs):
        """Sets object parameters upon initialization. Usually not used."""
        for key,value in kwargs.items():
            setattr(self, key, value)

    def train(self, train_df: pd.DataFrame):
        """Save the features and training IDs that were used to train the model.
        
        :param train_df: Input dataframe with training label under "Label" category and features under "Data" category.
        :type train_df: pd.DataFrame
        """
        self.features = list(train_df['Data'].columns)
        self.train_ids = train_df['Key']

    def predict(self, test_df:pd.DataFrame):
        """Predict label and return the predictions as a series.
        
        :param test_df: Test dataframe with features under "Data" category.
        :type test_df: pd.DataFrame
        """
        return pd.Series(dtype=float)

    def save(self, folder: str):
        """Save all attributes necessary to recreate a model.
        
        :param folder: Folder path will model outputs will be saved.
        :type folder: str
        """

        makedirs(folder, exist_ok=True)
        features_path = path.join(folder, 'features.json')
        train_ids_path = path.join(folder, 'train_ids.pkl')

        save_json(self.features, features_path)
        save_pickle(self.train_ids, train_ids_path)

    def load(self, folder: str):
        """Load all attributes of a saved model from a folder.

        :param folder: The folder path where the model will be loaded from.
        :type folder: str

        :return: None
        :rtype: None
        """
        features_path = path.join(folder, 'features.json')
        train_ids_path = path.join(folder, 'train_ids.pkl')

        self.features = load_json(features_path)
        self.train_ids = load_pickle(train_ids_path)


class LinearRegressor(Model):
    """Simple linear regression from SKLearn."""

    def train(self, train_df: pd.DataFrame):
        """Run sklearn linear regression and save the trained model to self.regressor_model.

        :param train_df: Input dataframe with training label under "Label" category and features under "Data" category.
        :type train_df: pd.DataFrame

        :return: None
        :rtype: None
        """
        super().train(train_df)
        from sklearn.linear_model import LinearRegression
        x_values = train_df['Data'].values
        y_values = train_df['Label'].values
        regressor_model = LinearRegression().fit(x_values, y_values)
        self.regressor_model = regressor_model

    def predict(self, test_df: pd.DataFrame):
        """Run linear regression model on input data.

        :param test_df: Dataframe to make predictions on.
        :type test_df: pd.DataFrame

        :return: Predicted values in a series.
        :rtype: pd.Series
        """
        super().predict(test_df)
        input_np_array = test_df['Data'][self.features].values
        output = self.regressor_model.predict(input_np_array).flatten()
        output = pd.Series(output)
        output.index = test_df.index
        return output

    def save(self, folder: str):
        """Save all attributes necessary to recreate a model.
        
        :param folder: Folder path will model outputs will be saved.
        :type folder: str
        """
        super().save(folder)
        model_path = path.join(folder, 'model.pkl')
        save_pickle(self.regressor_model, model_path)

    def load(self, folder: str):
        """Load all attributes of a saved model from a folder.

        :param folder: The folder path where the model will be loaded from.
        :type folder: str

        :return: None
        :rtype: None
        """
        super().load(folder)
        model_path = path.join(folder, 'model.pkl')
        self.regressor_model = load_pickle(model_path)
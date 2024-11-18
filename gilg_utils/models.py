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

class PytorchNeuralNetworkRegressor(Model):
    """Simple single hidden layer dense neural network in pytorch."""

    def train(self,
              train_df: pd.DataFrame,
              test_df: pd.DataFrame = None,
              dimension: int = 16,
              learning_rate: float = 0.01,
              dropout_rate: float = 0.1,
              loss_function_name: str = 'MSELoss',
              optimizer_name: str = 'SGD',
              max_epochs: int = 100,
              seed: int = 363,
              callback_period: int = 10):
        """Standardize input features, save the scaler to self.scaler, then train a pytorch neural network on the standardized features and save to self.regressor_model.

        :param train_df: Input dataframe with training label under "Label" category and features under "Data" category.
        :type train_df: pd.DataFrame

        :param test_df: An optional dataframe with training label under "Label" category and features under "Data" category. Performance on test set will be displayed during training.
        :type test_df: pd.DataFrame

        :param dimension: Size of the hidden layer.
        :type dimension: int
        
        :param learning_rate: Learning rate of the training loop.
        :type learning_rate: float
        
        :param dropout_rate: Percentage of weights to zero after each training epoch.
        :type dropout_rate: float
        
        :param loss_function_name: The pytorch loss to use.
        :type loss_function_name: str

        :param optimizer_name: The pytorch optimizer to use.
        :type optimizer_name: str
        
        :param max_epochs: The number of epochs to run in training.
        :type max_epochs: int

        :param seed: The random seed used to initialize weights.
        :type seed: int

        :param callback_period: Period of epochs between print statements.
        :type callback_period: int
        
        :return: None
        :rtype: None
        """
        super().train(train_df)
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        # Build the scaler
        train_data_x = train_df['Data'].values
        train_data_y = train_df['Label'].values
        
        scaler = StandardScaler()
        scaler.fit(train_data_x)
        train_data_x = scaler.transform(train_data_x)
            
        # Specify model parameters
        torch.manual_seed(seed)
        train_data_x = torch.tensor(train_data_x).float()
        train_data_y = torch.tensor(train_data_y).float().reshape(-1,1)

        test_flag = test_df is not None
        if test_flag:
            test_data_x = test_df['Data'].values
            test_data_y = test_df['Label'].values
            test_data_x = scaler.transform(test_data_x)
            test_data_x = torch.tensor(test_data_x).float()
            test_data_y = torch.tensor(test_data_y).float().reshape(-1,1)

        model = nn.Sequential(
                    nn.Linear(train_data_x.size()[1], dimension),
                    nn.Dropout(dropout_rate),
                    nn.ReLU(),
                    nn.Linear(dimension, 1)
        )

        loss_function = getattr(nn, loss_function_name)()
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)

        # Main training loop
        train_loss = np.inf
        epoch = 0
        while epoch <= max_epochs:
            pred_y = model(train_data_x)
            loss = loss_function(pred_y, train_data_y)
            train_loss = loss.item()

            model.zero_grad()
            loss.backward()

            if test_flag & (epoch % callback_period == 0):
                test_pred_y = model(test_data_x)
                loss = loss_function(test_pred_y, test_data_y)
                test_loss = loss.item()
                print(f'Epoch {epoch} Train loss: {np.round(train_loss,4)} Test loss: {np.round(test_loss,4)}')
            epoch += 1

        self.regressor_model = model
        self.scaler = scaler


    def predict(self, test_df: pd.DataFrame):
        """Run pytorch regression model on input data.

        :param test_df: Dataframe to make predictions on.
        :type test_df: pd.DataFrame

        :return: Predicted values in a series.
        :rtype: pd.Series
        """
        super().predict(test_df)
        import torch
        # Scale features
        test_data_x = test_df['Data'][self.features].values
        test_data_x = self.scaler.transform(test_data_x)
        test_data_x = torch.tensor(test_data_x).float()

        # Test model
        model = self.regressor_model
        model.eval()
        pred_y = model(test_data_x)
        pred_y = pred_y.detach().numpy().reshape(-1)
        output = pd.Series(pred_y)
        output.index = test_df.index
        return output

    def save(self, folder: str):
        """Save all attributes necessary to recreate a model.
        
        :param folder: Folder path will model outputs will be saved.
        :type folder: str
        """
        super().save(folder)
        import torch
        model_path = path.join(folder, 'nn.mdl')
        scaler_path = path.join(folder, 'scaler.pkl')

        torch.save(self.regressor_model, model_path)
        save_pickle(self.scaler, scaler_path)

    def load(self, folder: str):
        """Load all attributes of a saved model from a folder.

        :param folder: The folder path where the model will be loaded from.
        :type folder: str

        :return: None
        :rtype: None
        """
        super().load(folder)
        import torch
        model_path = path.join(folder, 'nn.mdl')
        scaler_path = path.join(folder, 'scaler.pkl')
        self.regressor_model = torch.load(model_path, weights_only=False)
        self.scaler = load_pickle(scaler_path)

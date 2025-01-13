import numpy as np
import pandas as pd
from os import path, makedirs
from gilg_utils.general import save_json, load_json, save_pickle, load_pickle, add_jane_multicolumn, impute_columns
from gilg_utils.models import Model
from gilg_utils.jane_processors import diff_transformation, expand_lags
import polars as pl
from tqdm import tqdm
import glob
import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

class DiffModel(Model):
    """Apply a diff transformation to the input before training or predicting."""
    
    def __init__(self, label_column, model_class, *args, **kwargs):
        """Sets object parameters upon initialization. Usually not used."""
        super().__init__(*args,**kwargs)
        self.label_column = label_column
        self.model_class = model_class

    def train(self, train_df: pd.DataFrame, test_df=None, *args, **kwargs):
        """Difference the train_df first and then train the real model.

        :param train_df: Input dataframe with training label under "Label" category and features under "Data" category.
        :type train_df: pd.DataFrame

        :return: None
        :rtype: None
        """
        train_df = diff_transformation(train_df)
        if test_df is not None:
            test_df = diff_transformation(test_df)
        super().train(train_df,test_df)
        model = self.model_class()
        model.train(train_df, test_df, *args, **kwargs)
        
        self.model = model

    def predict(self, test_df: pd.DataFrame):
        """Difference the test_df first and then predict the real model.

        :param test_df: Dataframe to make predictions on.
        :type test_df: pd.DataFrame

        :return: Predicted values in a series.
        :rtype: pd.Series
        """
        test_df = diff_transformation(test_df)
        super().predict(test_df)
        model = self.model
        output = model.predict(test_df)
        return output

    def save(self, folder: str):
        """Save all attributes necessary to recreate a model.
        
        :param folder: Folder path will model outputs will be saved.
        :type folder: str
        """
        super().save(folder)
        model_path = folder
        self.model.save(model_path)

    def load(self, folder: str):
        """Load all attributes of a saved model from a folder.

        :param folder: The folder path where the model will be loaded from.
        :type folder: str

        :return: None
        :rtype: None
        """
        super().load(folder)
        model_path = folder
        self.model = self.model_class()
        self.model.load(model_path)

class ForwardDiffModel(Model):
    """Take a diff trained model and extrapolate it forward on a new test set."""

    def __init__(self, label_column, diff_model, n_time_lags, *args, **kwargs):
        """Sets object parameters upon initialization. Usually not used."""
        super().__init__(*args,**kwargs)
        self.label_column = label_column
        self.model = diff_model
        self.n_time_lags = n_time_lags
        self.lags = None
        self.time_lags = {}
        self.prediction_lags = {}
        # self.diffs is the difference from the current timestep to the ith one in the past
        self.diffs = {}
        for i in range(n_time_lags+1):
            self.time_lags[i] = None
            self.prediction_lags[i] = None
            self.diffs[i] = None
            
    def update_properties(self, test, lags):
        """Update the class attributes with the new information."""
        if lags is not None:
            self.lags = lags
        input_df = test.to_pandas()
        for i in range(self.n_time_lags)[::-1]:
            self.time_lags[(i+1)] = self.time_lags.get(i)
            self.prediction_lags[(i+1)] = self.prediction_lags.get(i)
        self.time_lags[0] = input_df.fillna(0)
        self.prediction_lags[0] = None
        
    def restrict_time_lags(self):
        """Limit the time lags to the symbol ids that are found in the current timestep, and set new symbols to 0."""
        self.restricted_time_lags = {}
        for i in range(self.n_time_lags):
            time_lag_df = pd.DataFrame(self.time_lags[0]['symbol_id'])
            saved_time_lag = self.time_lags.get(i+1)
            if saved_time_lag is None:
                time_lag_df = None
            else:
                time_lag_df = time_lag_df.merge(saved_time_lag,how='left',on='symbol_id').fillna(0)
            self.restricted_time_lags[i+1] = time_lag_df

    def restrict_prediction_lags(self):
        """Limit the prediction lags to the symbol ids that are found in the current timestep, and set new symbols to 0."""
        self.restricted_prediction_lags = {}
        for i in range(self.n_time_lags):
            prediction_lag_df = pd.DataFrame(self.time_lags[0]['symbol_id'])
            saved_prediction_lag = self.prediction_lags.get(i+1)
            if saved_prediction_lag is None:
                prediction_lag_df[self.label_column] = 0
            else:
                prediction_lag_df = prediction_lag_df.merge(saved_prediction_lag,how='left',on='symbol_id').fillna(0)
            self.restricted_prediction_lags[i+1] = prediction_lag_df

    def compute_diffs(self):
        """Compute the differences from cached timesteps. self.diffs is the difference from the current timestep to the ith one in the past."""
        input_df = self.time_lags[0]
        input_df = impute_columns(input_df,self.model)
        input_df = add_jane_multicolumn(input_df,self.label_column)
        self.input_df = input_df
        for i in range(self.n_time_lags):
            time_lag_df = self.restricted_time_lags.get(i+1)
            if time_lag_df is None:
                self.diffs[i+1] = None
            else:
                time_lag_df = impute_columns(time_lag_df,self.model)
                time_lag_df = add_jane_multicolumn(time_lag_df,self.label_column)
                input_df['Data'] = input_df['Data'].values - time_lag_df['Data'].values
                self.diffs[i+1] = input_df

    def make_predictions(self):
        """Make diff prediction."""
        predictions = pd.DataFrame()
        predictions['row_id'] = self.input_df[('Key','row_id')].values

        if (self.restricted_prediction_lags.get(1) is None) or (self.diffs.get(1) is None):
            predictions[self.label_column] = 0
        else:
            pred = self.model.predict(self.diffs[1])
            predictions[self.label_column] = self.restricted_prediction_lags.get(1)[self.label_column] + pred.values
        return predictions

    def save_prediction_cache(self, predictions):
        """Update the cache."""
        predictions_lag = predictions.copy()
        predictions_lag['symbol_id'] = self.input_df[('Key','symbol_id')].values
        self.prediction_lags[0] = predictions_lag

    def jane_predict(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
        """The prediction function used by Jane street."""

        # Prepare all input data
        self.update_properties(test, lags)
        self.restrict_time_lags()
        self.restrict_prediction_lags()
        self.compute_diffs()

        # Make prediction
        predictions = self.make_predictions()
        
        # Save
        self.save_prediction_cache(predictions)
        predictions = pl.from_pandas(predictions)
        return predictions
    
    def predict(self, input_df):
        """Predict on the full multi-index future dataframe."""
        input_df.columns = input_df.columns.droplevel('Category')
            
        # Test model
        total_pred = []
        past_date_df = None
        for date_id in tqdm(sorted(list(set(input_df['date_id'])))):
            date_df = input_df[input_df['date_id']==date_id].copy()
            for time_id in sorted(list(set(date_df['time_id']))):
                if time_id==0:
                    if past_date_df is None:
                        lags = None
                    else:
                        lags = past_date_df.copy()
                        new_columns = []
                        for c in lags.columns:
                            if '_id' in c:
                                new_columns.append(c)
                            else:
                                new_columns.append(c+'_lag_1')
                        lags.columns = new_columns
                        lags = pl.from_pandas(lags)
                else:
                    lags = None
                time_df = date_df[date_df['time_id']==time_id].copy()
                time_df['row_id'] = time_df.index.values
                time_df = pl.from_pandas(time_df)
                time_pred = self.jane_predict(time_df,lags)
                total_pred.append(time_pred.to_pandas())
            past_date_df = date_df.copy()
        total_pred = pd.concat(total_pred,axis=0)
        total_pred = total_pred.set_index('row_id')
        total_pred.columns = pd.MultiIndex.from_arrays([['Predictions'],['Prediction']],names=('Category','Column'))
    
        # Add column levels back
        input_df = add_jane_multicolumn(input_df,self.label_column)
        input_df = pd.concat([input_df,total_pred],axis=1)
        input_df = input_df[['Key','Predictions','Meta','Label']]
        return input_df

# class TertiaryForwardModel(Forward_Diff_Model):
#     """Test a combinations of a diff model and a secondary correction model together."""
    
#     def __init__(self, label_column, diff_model, tertiary_model, n_time_lags, *args, **kwargs):
#         """Sets object parameters upon initialization. Usually not used."""
#         super().__init__(label_column, diff_model, n_time_lags, *args, **kwargs)
#         self.tertiary_model = tertiary_model
        
#     def make_predictions(self):
#         """Make diff prediction."""
#         predictions = pd.DataFrame()
#         predictions['row_id'] = self.input_df[('Key','row_id')].values

#         if (self.restricted_prediction_lags.get(1) is None) or (self.diffs.get(1) is None):
#             predictions[self.label_column] = 0
#         else:
#             pred = self.model.predict(self.diffs[1])
#             predictions[self.label_column] = self.restricted_prediction_lags.get(1)[self.label_column] + pred.values
#         return predictions
    
#     def jane_predict(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
#         """The prediction function used by Jane street."""

#         # Prepare all input data
#         self.update_properties(test, lags)
#         self.restrict_time_lags()
#         self.restrict_prediction_lags()
#         self.compute_diffs()

#         # Make prediction
#         predictions = self.make_predictions()
        
#         # Save
#         self.save_prediction_cache(predictions)
#         predictions = pl.from_pandas(predictions)
#         return predictions

class SaveLags(Model):
    def __init__(self, label_column, n_time_lags, *args, **kwargs):
        """Sets object parameters upon initialization. Usually not used."""
        super().__init__(*args,**kwargs)
        self.label_column = label_column
        self.n_time_lags = n_time_lags
        self.lags = None
        self.time_lags = {}
        self.prediction_lags = {}
        # self.diffs is the difference from the current timestep to the ith one in the past
        self.diffs = {}
        for i in range(n_time_lags+1):
            self.time_lags[i] = None
            self.prediction_lags[i] = None
            self.diffs[i] = None
            
    def update_properties(self, test, lags):
        """Update the class attributes with the new information."""
        if lags is not None:
            self.lags = lags
        input_df = test.to_pandas()
        for i in range(self.n_time_lags)[::-1]:
            self.time_lags[(i+1)] = self.time_lags.get(i)
            self.prediction_lags[(i+1)] = self.prediction_lags.get(i)
        self.time_lags[0] = input_df.fillna(0)
        self.prediction_lags[0] = None
        
    def restrict_time_lags(self):
        """Limit the time lags to the symbol ids that are found in the current timestep, and set new symbols to 0."""
        self.restricted_time_lags = {}
        for i in range(self.n_time_lags+1):
            time_lag_df = pd.DataFrame(self.time_lags[0]['symbol_id'])
            saved_time_lag = self.time_lags.get(i)
            if saved_time_lag is None:
                time_lag_df = None
            else:
                time_lag_df = time_lag_df.merge(saved_time_lag,how='left',on='symbol_id').fillna(0)
            self.restricted_time_lags[i] = time_lag_df

    def restrict_prediction_lags(self):
        """Limit the prediction lags to the symbol ids that are found in the current timestep, and set new symbols to 0."""
        self.restricted_prediction_lags = {}
        for i in range(self.n_time_lags):
            prediction_lag_df = pd.DataFrame(self.time_lags[0]['symbol_id'])
            saved_prediction_lag = self.prediction_lags.get(i+1)
            if saved_prediction_lag is None:
                prediction_lag_df[self.label_column] = 0
            else:
                prediction_lag_df = prediction_lag_df.merge(saved_prediction_lag,how='left',on='symbol_id').fillna(0)
            self.restricted_prediction_lags[i+1] = prediction_lag_df

    def compute_diffs(self):
        """Compute the differences from cached timesteps. self.diffs is the difference from the i+1th timestep to the ith one in the past."""
        input_df = self.time_lags[0]
        # input_df = impute_columns(input_df,self.model)
        input_df = add_jane_multicolumn(input_df,self.label_column)
        self.input_df = input_df
        for i in range(self.n_time_lags):
            time_lag_df = self.restricted_time_lags.get(i+1)
            reference_df = self.restricted_time_lags.get(i)
            if time_lag_df is None:
                self.diffs[i] = None
            else:
                # time_lag_df = impute_columns(time_lag_df,self.model)
                time_lag_df = add_jane_multicolumn(time_lag_df,self.label_column)
                # reference_df = impute_columns(reference_df,self.model)
                reference_df = add_jane_multicolumn(reference_df,self.label_column)
                input_df['Data'] = reference_df['Data'].values - time_lag_df['Data'].values
                self.diffs[i] = input_df.copy()

    def save_prediction_cache(self, predictions):
        """Update the cache."""
        predictions_lag = predictions.copy()
        predictions_lag['symbol_id'] = self.input_df[('Key','symbol_id')].values
        self.prediction_lags[0] = predictions_lag

    def jane_predict(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
        """The prediction function used by Jane street."""
        # Prepare all input data
        self.update_properties(test, lags)
        self.restrict_time_lags()
        self.restrict_prediction_lags()
        self.compute_diffs()
    
    def predict(self, input_df):
        """Predict on the full multi-index future dataframe."""
        input_df = input_df.copy()
        input_df.columns = input_df.columns.droplevel('Category')
            
        # Test model
        total_pred = []
        past_date_df = None
        for date_id in tqdm(sorted(list(set(input_df['date_id'])))):
            date_df = input_df[input_df['date_id']==date_id].copy()
            for time_id in sorted(list(set(date_df['time_id']))):
                if time_id==0:
                    if past_date_df is None:
                        lags = None
                    else:
                        lags = past_date_df.copy()
                        new_columns = []
                        for c in lags.columns:
                            if '_id' in c:
                                new_columns.append(c)
                            else:
                                new_columns.append(c+'_lag_1')
                        lags.columns = new_columns
                        lags = pl.from_pandas(lags)
                else:
                    lags = None
                time_df = date_df[date_df['time_id']==time_id].copy()
                time_df['row_id'] = time_df.index.values
                time_df = pl.from_pandas(time_df)
                time_pred = self.jane_predict(time_df,lags)
                total_pred.append(time_pred.to_pandas())
            past_date_df = date_df.copy()
        total_pred = pd.concat(total_pred,axis=0)
        total_pred = total_pred.set_index('row_id')
        total_pred.columns = pd.MultiIndex.from_arrays([['Predictions'],['Prediction']],names=('Category','Column'))
    
        # Add column levels back
        input_df = add_jane_multicolumn(input_df,self.label_column)
        input_df = pd.concat([input_df,total_pred],axis=1)
        input_df = input_df[['Key','Predictions','Meta','Label']]
        return input_df
        
class StackedModel(SaveLags):
    """Use lists of regressive and diff models to train a new stacked model."""

    def __init__(self, label_column, regressive_model_list, diff_model_list, n_time_lags, model_class, *args, **kwargs):
        """Sets object parameters upon initialization. Usually not used."""
        #Need to add 1 for a stacked model's time_lags to account for the diff, which needs an additional timestep in the past
        super().__init__(label_column=label_column, n_time_lags=n_time_lags+1,*args,**kwargs)
        self.label_column = label_column
        self.regressive_model_list = regressive_model_list
        self.diff_model_list = diff_model_list
        self.model_class = model_class

    def transform_train_df(self, train_df):
        """Use the regressive and diff model lists to convert the train df into one that uses the outputs of these models."""
        transformed_df = train_df[['Key','Meta','Label']].copy()
        for i, model in enumerate(self.regressive_model_list):
            column_name = f"regressive_model_{i}_output"
            transformed_df[('Data',column_name)] = model.predict(train_df).values
    
        for i, model in enumerate(self.diff_model_list):
            column_name = f"diff_model_{i}_output"
            transformed_df[('Data',column_name)] = model.predict(train_df).values
        transformed_df = expand_lags(transformed_df, self.n_time_lags-1)
        return transformed_df
    
    def train(self, train_df, test_df=None, *args, **kwargs):
        """Train a new model on the stacked inputs."""
        super().train(train_df, test_df)
        transformed_df = self.transform_train_df(train_df)
        if test_df is not None:
            test_df = self.transform_train_df(test_df)
        model = self.model_class()
        model.train(transformed_df, test_df, *args, **kwargs)
        self.model = model
        
    def make_predictions(self):
        """Make stacked model prediction. Need to use the model attribute of diff models so that they don't apply the diff transformation twice."""
        predictions = pd.DataFrame()
        predictions['row_id'] = self.input_df[('Key','row_id')].values

        if (self.restricted_time_lags.get(self.n_time_lags-1) is None) or (self.diffs.get(self.n_time_lags-1) is None):
            predictions[self.label_column] = 0
        else:
            test_df = pd.DataFrame()
            for i, model in enumerate(self.regressive_model_list):
                column_name = f"regressive_model_{i}_output"
                test_df[column_name] = model.predict(impute_columns(add_jane_multicolumn(self.restricted_time_lags[0]),model))
        
            for i, model in enumerate(self.diff_model_list):
                column_name = f"diff_model_{i}_output"
                test_df[column_name] = model.model.predict(impute_columns(self.diffs[0],model))

            for lag in range(self.n_time_lags-1):
                for i, model in enumerate(self.regressive_model_list):
                    column_name = f"regressive_model_{i}_output_lag_{lag+1}"
                    test_df[column_name] = model.predict(impute_columns(add_jane_multicolumn(self.restricted_time_lags[lag+1]),model))
        
                for i, model in enumerate(self.diff_model_list):
                    column_name = f"diff_model_{i}_output_lag_{lag+1}"
                    test_df[column_name] = model.model.predict(impute_columns(self.diffs[lag+1],model))

            test_df.columns = pd.MultiIndex.from_arrays([['Data' for c in test_df.columns],test_df.columns],names=['Category','Column'])
            pred = self.model.predict(impute_columns(test_df,self.model))
            predictions[self.label_column] = pred.values
        return predictions
        
    def jane_predict(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
        """The prediction function used by Jane street."""
        super().jane_predict(test,lags)

        # Make prediction
        predictions = self.make_predictions()
        
        # Save
        self.save_prediction_cache(predictions)
        predictions = pl.from_pandas(predictions)
        return predictions

    def save(self, folder: str):
        """Save all attributes necessary to recreate a model.
        
        :param folder: Folder path will model outputs will be saved.
        :type folder: str
        """
        super().save(folder)
        model_path = path.join(folder, 'model')
        self.model.save(model_path)
        
        for i, model in enumerate(self.regressive_model_list):
            model_path = path.join(folder, f'regressive_model_{i}')
            model.save(model_path)
    
        for i, model in enumerate(self.diff_model_list):
            model_path = path.join(folder, f'diff_model_{i}')
            model.save(model_path)

    def load(self, folder: str):
        """Load all attributes of a saved model from a folder.

        :param folder: The folder path where the model will be loaded from.
        :type folder: str

        :return: None
        :rtype: None
        """
        super().load(folder)
        model_path = path.join(folder, 'model')
        self.model = self.model_class()
        self.model.load(model_path)

        for model_path in sorted(glob.glob(path.join(folder, f'regressive_model_*'))):
            model = self.model_class()
            model.load(model_path)
            self.regressive_model_list.append(model)
    
        for model_path in sorted(glob.glob(path.join(folder, f'diff_model_*'))):
            model = DiffModel(label_column=self.label_column,model_class=self.model_class)
            model.load(model_path)
            self.diff_model_list.append(model)

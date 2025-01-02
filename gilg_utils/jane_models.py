import numpy as np
import pandas as pd
from os import path, makedirs
from gilg_utils.general import save_json, load_json, save_pickle, load_pickle, add_jane_multicolumn, impute_columns
from gilg_utils.models import Model
import polars as pl
from tqdm import tqdm

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
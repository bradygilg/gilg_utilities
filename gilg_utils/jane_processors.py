import numpy as np
import pandas as pd
from os import path, makedirs
from gilg_utils.general import save_json, load_json, save_pickle, load_pickle

def feature_properties(input_df):
    """Compute statistical properties of features."""
    from scipy.stats import spearmanr
    avg_unique = input_df.groupby([('Key','date_id')])['Data'].nunique().mean()
    avg_values = input_df.groupby([('Key','date_id')])['Data'].mean()
    avg_diff = input_df.groupby([('Key','date_id')]).apply(lambda df: df['Data'].diff().mean())
    
    unique_df = avg_unique.reset_index()[['Column',0]].rename({'Column':'Feature',0:'Nunique'},axis=1)
    
    spearman_df = pd.DataFrame()
    feature_names = []
    spearman_rs = []
    for c in avg_values['Data'].columns:
        feature_names.append(c)
        s = spearmanr(avg_values[('Data',c)],avg_diff[c]).statistic
        spearman_rs.append(s)
    spearman_df['Feature'] = feature_names
    spearman_df['Value_Diff_Spearman'] = spearman_rs
    
    lag_df = pd.DataFrame()
    for lag in [1]:
        feature_names = []
        lag_spearman = []
        for c in input_df['Data'].columns:
            feature_names.append(c)
            lag_spearman.append(spearmanr(input_df[('Data',c)].iloc[lag:],input_df[('Data',c)].iloc[:-lag] ).statistic)
        lag_df['Feature'] = feature_names
        lag_df[f'Autocorrelation_lag_{lag}'] = lag_spearman
        
    for lag in [1,2,3]:
        feature_names = []
        lag_spearman = []
        for c in input_df['Data'].columns:
            feature_names.append(c)
            lag_spearman.append(spearmanr(input_df[('Data',c)].diff().iloc[1+lag:],input_df[('Data',c)].diff().iloc[1:-lag] ).statistic)
        lag_df['Feature'] = feature_names
        lag_df[f'Autocorrelation_diff_lag_{lag}'] = lag_spearman
    
    out_df = unique_df.merge(spearman_df,on='Feature').merge(lag_df,on='Feature')
    return out_df

def select_feature_parameters(feature_property_df,
                              min_unique_fraction=0.5,
                              min_log_transform_spearman=0.4,
                              max_no_diff_spearman=0.65,
                              max_one_diff_spearman=0.3):
    """Identify whether features are autoregressive, log scale, or unuseable."""
    out_df = feature_property_df.copy()
    feature_filter = (out_df['Nunique']<(len(out_df)*min_unique_fraction))
    log_transform_filter = (out_df['Value_Diff_Spearman'].abs()>=min_log_transform_spearman)
    no_diff = (out_df['Autocorrelation_lag_1'].abs()<max_no_diff_spearman)
    one_diff = (~no_diff) & (out_df['Autocorrelation_diff_lag_1'].abs()<max_one_diff_spearman)
    two_diff = (~no_diff) & (~one_diff)
    out_df['Filtered'] = feature_filter
    out_df['Log_Transform'] = log_transform_filter
    out_df['Diff_Lag'] = 0
    out_df.loc[one_diff,'Diff_Lag'] = 1
    out_df.loc[two_diff,'Diff_Lag'] = 2
    return out_df
import numpy as np
import pandas as pd
from os import path, makedirs
from gilg_utils.general import save_json, load_json, save_pickle, load_pickle, select_multicolumns

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

def apply_feature_transformation(input_df,feature_df):
    """Use the feature parameters to transform the input dataframe."""
    
    # Filter bad features
    keep_features = list(feature_df[~feature_df['Filtered']]['Feature'])
    input_df = select_multicolumns(input_df,'Data',keep_features)
    
    # # Log transform currently disabled due to negative values
    # log_transform_features = list(feature_df[~feature_df['Log_Transform']]['Feature'])
    # for c in log_transform_features:
    #     input_df[('Data',c)] = np.log2(np.abs(input_df[('Data',c)])+0.000000001)

    # Diff features
    diff_1_features = list(feature_df[feature_df['Diff_Lag']==1]['Feature'])
    diff_2_features = list(feature_df[feature_df['Diff_Lag']==2]['Feature'])
    
    categories = input_df.columns.get_level_values('Category')
    columns = input_df.columns.get_level_values('Column')
    diff_1_feature_mask = (categories=='Data') & (columns.isin(diff_1_features))
    diff_2_feature_mask = (categories=='Data') & (columns.isin(diff_2_features))
    
    lag_1 = input_df.loc[:,diff_1_feature_mask].copy()
    lag_1.iloc[1:,:] = lag_1.iloc[:-1,:]
    lag_2 = input_df.loc[:,diff_2_feature_mask].copy()
    lag_2.iloc[2:,:] = lag_2.iloc[:-2,:]
    
    input_df.loc[:,diff_1_feature_mask] = input_df.loc[:,diff_1_feature_mask] - lag_1
    input_df.loc[:,diff_2_feature_mask] = input_df.loc[:,diff_2_feature_mask] - lag_2

    # Diff label
    lag_1_label = input_df.loc[:,categories=='Label'].copy()
    lag_1_label.iloc[1:,:] = lag_1_label.iloc[:-1,:]
    input_df.loc[:,categories=='Label'] = input_df.loc[:,categories=='Label'] - lag_1_label

    # Remove timesteps that do not have enough lag
    input_df = input_df[~input_df[('Key','time_id')].isin([0,1])]
    return input_df

def three_diff_feature_transformation(input_df,feature_df):
    """Use the feature parameters to transform the input dataframe."""
    
    # Filter bad features
    keep_features = list(feature_df[~feature_df['Filtered']]['Feature'])
    input_df = select_multicolumns(input_df,'Data',keep_features)
    
    # # Log transform currently disabled due to negative values
    # log_transform_features = list(feature_df[~feature_df['Log_Transform']]['Feature'])
    # for c in log_transform_features:
    #     input_df[('Data',c)] = np.log2(np.abs(input_df[('Data',c)])+0.000000001)

    # Diff features
    categories = input_df.columns.get_level_values('Category')
    columns = input_df.columns.get_level_values('Column')
    
    lag_0 = input_df['Data'].copy()
    lag_1 = input_df['Data'].copy()
    lag_2 = input_df['Data'].copy()
    lag_3 = input_df['Data'].copy()
    input_df = input_df[['Key','Meta','Label']]
    lag_1.iloc[1:,:] = lag_1.iloc[:-1,:]
    lag_2.iloc[2:,:] = lag_2.iloc[:-2,:]
    lag_3.iloc[3:,:] = lag_3.iloc[:-3,:]

    diff_1 = lag_0 - lag_1
    diff_2 = lag_1 - lag_2
    diff_3 = lag_2 - lag_3
    del lag_1, lag_2, lag_3
    import gc
    gc.collect()

    diff_1.columns = [c+'_diff_lag_1' for c in diff_1.columns]
    diff_2.columns = [c+'_diff_lag_2' for c in diff_2.columns]
    diff_3.columns = [c+'_diff_lag_3' for c in diff_3.columns]
    new_data = pd.concat([lag_0,diff_1,diff_2,diff_3],axis=1)
    new_data.columns = pd.MultiIndex.from_arrays([['Data' for c in new_data.columns],new_data.columns],names=['Category','Column'])
    input_df = pd.concat([input_df,new_data],axis=1)

    # Diff label
    categories = input_df.columns.get_level_values('Category')
    lag_1_label = input_df.loc[:,categories=='Label'].copy()
    lag_1_label.iloc[1:,:] = lag_1_label.iloc[:-1,:]
    input_df.loc[:,categories=='Label'] = input_df.loc[:,categories=='Label'] - lag_1_label

    # Remove timesteps that do not have enough lag
    input_df = input_df[~input_df[('Key','time_id')].isin([0,1,2])]
    return input_df

def diff_transformation(input_df):
    """Convert a dataframe to a diff dataframe."""

    # Diff features
    categories = input_df.columns.get_level_values('Category')
    columns = input_df.columns.get_level_values('Column')
    
    lag_0 = input_df['Data'].copy()
    lag_1 = input_df['Data'].copy()
    input_df = input_df[['Key','Meta','Label']]
    lag_1.iloc[1:,:] = lag_1.iloc[:-1,:]

    diff_1 = lag_0 - lag_1
    del lag_0, lag_1
    import gc
    gc.collect()

    diff_1.columns = pd.MultiIndex.from_arrays([['Data' for c in diff_1.columns],diff_1.columns],names=['Category','Column'])
    input_df = pd.concat([input_df,diff_1],axis=1)

    # Diff label
    categories = input_df.columns.get_level_values('Category')
    lag_1_label = input_df.loc[:,categories=='Label'].copy()
    lag_1_label.iloc[1:,:] = lag_1_label.iloc[:-1,:]
    input_df.loc[:,categories=='Label'] = input_df.loc[:,categories=='Label'] - lag_1_label

    # Remove timesteps that do not have enough lag
    # input_df = input_df[~input_df[('Key','time_id')].isin([0])]
    return input_df

def expand_lags(input_df,n_time_lags):
    """Include data from previous rows into each row."""
    new_data = []
    for i in range(n_time_lags):
        lag = input_df['Data'].copy()
        lag.iloc[i+1:,:] = lag.iloc[:-(i+1),:]
        lag.columns = [c+f'_lag_{i+1}' for c in lag.columns]
        new_data.append(lag)
    new_data = [input_df['Data'].copy()] + new_data
    input_df = input_df.drop('Data',axis=1)
    new_data = pd.concat(new_data,axis=1)
    new_data.columns = pd.MultiIndex.from_arrays([['Data' for c in new_data.columns],new_data.columns],names=['Category','Column'])
    input_df = pd.concat([input_df,new_data],axis=1)
    return input_df
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import PredefinedSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



#General Preprocessing functions:
def clean_NaN(df):
    df = df.copy()
    df.dropna(subset=['target'], inplace=True)
    return df

def remove_long_sequences(df, col_name, seq_len):
    df = df.copy()
    # Identify sequences of zeros
    df['group'] = (df[col_name] != 0).cumsum()
    df['group_count'] = df.groupby('group')[col_name].transform('count')
    
    # Create a mask to identify rows with sequences longer than seq_len and isshadow lower than 1
    mask = (df[col_name] == 0) & (df['group_count'] > seq_len) #& (df['is_in_shadow:idx'] < 1)
    
    # Remove rows with sequences longer than seq_len and isshadow lower than 1
    df_cleaned = df[~mask].drop(columns=['group', 'group_count'])
    return df_cleaned.copy()


def remove_repeating_nonzero(df, col_name, repeat_count=5):
    df = df.copy()
    # create a mask to identify rows with repeating nonzero values in the target column
    mask = ((df[col_name] != 0) & (df[col_name].shift(1) == df[col_name]))
    # create a mask to identify rows with repeating nonzero values that occur more than repeat_count times
    repeat_mask = mask & (mask.groupby((~mask).cumsum()).cumcount() >= repeat_count)
    # create a mask to identify the complete sequence of repeating nonzero values
    seq_mask = repeat_mask | repeat_mask.shift(-5)
    # remove rows with repeating nonzero values that occur more than repeat_count times
    df = df[~seq_mask]
    return df



def remove_outliers_hourly_and_month(df, target_col):
    df = df.copy()
    for month in range (1,13):
        df_month = df[df.index.month == month]
        for i in range(24):
            df_hour=df_month[df_month.index.hour == i]
            Q1 = df_hour[target_col].quantile(0.25)
            Q3 = df_hour[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_hour_rem = df_hour[(df_hour[target_col] >= lower_bound) & (df_hour[target_col] <= upper_bound)]
            if i==0:
                df_rem=df_hour_rem
            else:
                df_rem=pd.concat([df_rem,df_hour_rem],axis=0)
        if month==1:
            df_rem_month=df_rem
        else:
            df_rem_month=pd.concat([df_rem_month,df_rem],axis=0)
    df_rem_month.sort_index(inplace=True)
    return df_rem_month

def clean(df):
    df = df.copy()
    df=clean_NaN(df)
    df=remove_long_sequences(df, 'target', 60)
    df=remove_repeating_nonzero(df, 'target')
    #df=remove_outliers_hourly_and_month(df, 'target')
    return df


def encode(data, col, max_val):
    data = data.copy()
    data = data.copy()
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def create_time_features(df):
    df = df.copy()
    df["hour"]=df.index.hour
    df["dayofyear"]=df.index.dayofyear
    df["month"]=df.index.month
    df["week"] = df.index.isocalendar().week

    #zero indexing:
    df["dayofyear"]-=1
    df["month"]-=1
    df["week"]-=1


    #Cycling the time features:
    df = encode(df, "hour", 24)
    df = encode(df, "month", 12)
    df = encode(df, "week", 53)
    df = encode(df, "dayofyear", 366)

    df.drop(columns=["hour", "month", "week", "dayofyear"], inplace=True)




    df["mult1"]=(1-df["is_in_shadow:idx"])*df['direct_rad:W']
    df["mult2"]=(1-df["is_in_shadow:idx"])*df['clear_sky_rad:W']
    df["date_calc"]=pd.to_datetime(df["date_calc"])
    df.index=pd.to_datetime(df.index)
    df["uncertainty"]=(df.index-df["date_calc"]).apply(lambda x: x.total_seconds()/3600)
    df["uncertainty"].fillna(0, inplace=True)
    return df

def create_features(df):
    df = df.copy()

    df.dropna(subset=['absolute_humidity_2m:gm3'], inplace=True)
    df["total_solar_rad"]=df["direct_rad:W"]+df["diffuse_rad:W"]
    #df["clear_sky_%"]=df["total_solar_rad"]/df["clear_sky_rad:W"]*100
    #df["clear_sky_%"].fillna(0, inplace=True)
    df["spec humid"]=df["absolute_humidity_2m:gm3"]/df["air_density_2m:kgm3"]
    df["temp*total_rad"]=df["t_1000hPa:K"]*df["total_solar_rad"]
    df["wind_angle"]=(np.arctan2(df["wind_speed_u_10m:ms"],df["wind_speed_v_10m:ms"]))*180/np.pi
    #df["total_snow_depth"] = df["snow_depth:cm"] + df["fresh_snow_1h:cm"]
    #df["total_precip_5min"] = df["precip_5min:mm"] + df["snow_melt_10min:mm"]
    #df["total_precip_type"] = df["precip_type_5min:idx"] + df["snow_water:kgm2"]
    df["total_pressure"] = df["pressure_50m:hPa"] + df["pressure_100m:hPa"]
    df["total_sun_angle"] = df["sun_azimuth:d"] + df["sun_elevation:d"]
    df["solar intensity"]=1361*np.cos(np.radians(90-df["sun_elevation:d"]))
    df["solar intensity"].clip(lower=0, inplace=True)
    return df

def shift_target(df, target_col):
    df = df.copy()
    # Ensure the DataFrame is indexed by date
    df.index = pd.to_datetime(df.index)

    # Store the original indices
    original_indices = df.index

    # Reindex the DataFrame to include all 15-minute intervals
    all_intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15T')
    df = df.reindex(all_intervals)

    # Shift the target variable by 1 period (15 minutes) forward and backward
    df[target_col + '_shifted_forward'] = df[target_col].shift(-1)
    df[target_col + '_shifted_backward'] = df[target_col].shift(1)

    # Forward fill the missing values for the forward shift
    df[target_col + '_shifted_forward'].fillna(method='ffill', inplace=True)

    # Backward fill the missing values for the backward shift
    df[target_col + '_shifted_backward'].fillna(method='bfill', inplace=True)

    # Keep only the original indices
    df = df.loc[original_indices]

    return df


def add_lagged_features(df):
    df = df.copy()
    features_to_lag = [ "total_solar_rad", "temptotal_rad", "clear_sky_radW", "diffuse_radW", "direct_radW",  "total_cloud_coverp", "solarintensity", "total_sun_angle", "pressure_100mhPa"]
    
    for feature in features_to_lag:
        df = shift_target(df, feature)

    return df


def general_read_flaml(letter):

    df = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_train_observed.parquet")
    df2=pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_train_estimated.parquet")
    y = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/train_targets.parquet")
    # set the index to date_forecast and group by hourly frequency
    df.set_index("date_forecast", inplace=True)
    df2.set_index("date_forecast", inplace=True)
    y.set_index("time", inplace=True)

    df.index = pd.to_datetime(df.index)
    df2.index = pd.to_datetime(df2.index)
    y.index = pd.to_datetime(y.index) 
    
    df=pd.concat([df,df2],axis=0)

    # truncate y to match the index of df
    y = y.truncate(before=df.index[0], after=df.index[-1])
    latest_y_time = y.index[-1]
    latest_needed_df_time = latest_y_time + pd.Timedelta(minutes=45)
    # Truncate y based on df
    y = y.truncate(before=df.index[0], after=df.index[-1])
    # Ensure df has all needed entries from the start of y to 45 minutes after the end of y
    df = df.truncate(before=y.index[0], after=latest_needed_df_time)
    y.rename(columns={"pv_measurement":"target"},inplace=True)
    X = df.copy()
    Y = y.copy()
    #drop nan rows in Y
    Y = clean(Y)
    X.index = pd.to_datetime(X.index)
    Y.index = pd.to_datetime(Y.index)

    X_filtered = X[X.index.floor('H').isin(Y.index)]

    # Step 2: Ensure there are exactly four 15-min intervals for each hour
    valid_indices = X_filtered.groupby(X_filtered.index.floor('H')).filter(lambda group: len(group) == 4).index

    # Final filtered X
    X_final = X[X.index.isin(valid_indices)]


    #Troubleshooting: Find and print the hours with a mismatch
    group_sizes = X_filtered.groupby(X_filtered.index.floor('H')).size()
    mismatch_hours = group_sizes[group_sizes != 4]

    #Additional troubleshooting: find hours in Y without four 15-min intervals in X
    missing_hours_in_x = Y.index[~Y.index.isin(X_filtered.index.floor('H'))]


    #Remove mismatched and missing hours from Y
    all_issues = mismatch_hours.index.union(missing_hours_in_x)
    Y_clean = Y[~Y.index.isin(all_issues)]

    #dropping nan columns:
    X_final = X_final.drop(columns=['cloud_base_agl:m'])
    X_final = X_final.drop(columns=['ceiling_height_agl:m'])
    X_final = X_final.drop(columns=['snow_density:kgm3'])

    X_final = create_features(X_final)
    X_final = create_time_features(X_final)
    X_final.drop(columns=['date_calc'], inplace=True)

    X_final = X_final.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    Y_clean = Y_clean.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    #X_final = add_lagged_features(X_final)

    # Split X_final into a list of 4-row DataFrames
    X_grouped = [group for _, group in X_final.groupby(X_final.index.floor('H')) if len(group) == 4]
    
    # Ensure we only take the groups of X corresponding to Y_clean
    X_list = [X_grouped[i] for i in range(len(Y_clean))]

    return X_list, Y_clean


def general_read(letter):

    df = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_train_observed.parquet")
    df2=pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_train_estimated.parquet")
    y = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/train_targets.parquet")
    # set the index to date_forecast and group by hourly frequency
    df.set_index("date_forecast", inplace=True)
    df2.set_index("date_forecast", inplace=True)
    y.set_index("time", inplace=True)

    df.index = pd.to_datetime(df.index)
    df2.index = pd.to_datetime(df2.index)
    y.index = pd.to_datetime(y.index) 
    
    df=pd.concat([df,df2],axis=0)

    # truncate y to match the index of df
    y = y.truncate(before=df.index[0], after=df.index[-1])
    latest_y_time = y.index[-1]
    latest_needed_df_time = latest_y_time + pd.Timedelta(minutes=45)
    # Truncate y based on df
    y = y.truncate(before=df.index[0], after=df.index[-1])
    # Ensure df has all needed entries from the start of y to 45 minutes after the end of y
    df = df.truncate(before=y.index[0], after=latest_needed_df_time)
    y.rename(columns={"pv_measurement":"target"},inplace=True)
    X = df.copy()
    Y = y.copy()
    #drop nan rows in Y
    Y = clean(Y)
    X.index = pd.to_datetime(X.index)
    Y.index = pd.to_datetime(Y.index)

    X_filtered = X[X.index.floor('H').isin(Y.index)]

    # Step 2: Ensure there are exactly four 15-min intervals for each hour
    valid_indices = X_filtered.groupby(X_filtered.index.floor('H')).filter(lambda group: len(group) == 4).index

    # Final filtered X
    X_final = X[X.index.isin(valid_indices)]


    #Troubleshooting: Find and print the hours with a mismatch
    group_sizes = X_filtered.groupby(X_filtered.index.floor('H')).size()
    mismatch_hours = group_sizes[group_sizes != 4]

    #Additional troubleshooting: find hours in Y without four 15-min intervals in X
    missing_hours_in_x = Y.index[~Y.index.isin(X_filtered.index.floor('H'))]


    #Remove mismatched and missing hours from Y
    all_issues = mismatch_hours.index.union(missing_hours_in_x)
    Y_clean = Y[~Y.index.isin(all_issues)]

    #dropping nan columns:
    X_final = X_final.drop(columns=['cloud_base_agl:m'])
    X_final = X_final.drop(columns=['ceiling_height_agl:m'])
    X_final = X_final.drop(columns=['snow_density:kgm3'])

    X_final = create_features(X_final)
    X_final = create_time_features(X_final)
    X_final.drop(columns=['date_calc'], inplace=True)

    X_final = X_final.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    Y_clean = Y_clean.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    X_final = add_lagged_features(X_final)

    # Split X_final into a list of 4-row DataFrames
    X_grouped = [group for _, group in X_final.groupby(X_final.index.floor('H')) if len(group) == 4]
    
    # Ensure we only take the groups of X corresponding to Y_clean
    X_list = [X_grouped[i] for i in range(len(Y_clean))]

    return X_list, Y_clean

def readRawData(letter):
    # read X_train_observed.parquet file for the current letter
    df = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_train_observed.parquet")

    df2=pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_train_estimated.parquet")

    # set the index to date_forecast and group by hourly frequency
    df.set_index("date_forecast", inplace=True)
    df.index = pd.to_datetime(df.index)



    df2.set_index("date_forecast", inplace=True)
    df2.index = pd.to_datetime(df2.index)

    # read train_targets.parquet file for the current letter
    y = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/train_targets.parquet")
    y.set_index("time", inplace=True)
    y.index = pd.to_datetime(y.index) 

    
    df=pd.concat([df,df2],axis=0)

    # truncate y to match the index of df
    y = y.truncate(before=df.index[0], after=df.index[-1])
    latest_y_time = y.index[-1]
    latest_needed_df_time = latest_y_time + pd.Timedelta(minutes=45)

    # Truncate y based on df
    y = y.truncate(before=df.index[0], after=df.index[-1])

    # Ensure df has all needed entries from the start of y to 45 minutes after the end of y
    df = df.truncate(before=y.index[0], after=latest_needed_df_time)

    y.rename(columns={"pv_measurement":"target"},inplace=True)


    X = df.copy()
    Y = y.copy()
    #drop nan rows in Y
    Y = clean(Y)
    X.index = pd.to_datetime(X.index)
    Y.index = pd.to_datetime(Y.index)

    #removing november december and january
    #Y = Y[(Y.index.month != 11) & (Y.index.month != 12) & (Y.index.month != 1)] 

    # Step 1: Keep only rows in X that are within an hour present in Y
    X_filtered = X[X.index.floor('H').isin(Y.index)]

    # Step 2: Ensure there are exactly four 15-min intervals for each hour
    valid_indices = X_filtered.groupby(X_filtered.index.floor('H')).filter(lambda group: len(group) == 4).index

    # Final filtered X
    X_final = X[X.index.isin(valid_indices)]

    #Check length conditions
    # print(f"\nExpected length of X_final: {4 * len(Y)}")
    # print(f"Actual length of X_final: {len(X_final)}")

    #Troubleshooting: Find and print the hours with a mismatch
    group_sizes = X_filtered.groupby(X_filtered.index.floor('H')).size()
    mismatch_hours = group_sizes[group_sizes != 4]

    # print("\nHours with mismatched number of 15-min intervals:")
    # print(mismatch_hours)

    #Additional troubleshooting: find hours in Y without four 15-min intervals in X
    missing_hours_in_x = Y.index[~Y.index.isin(X_filtered.index.floor('H'))]
    # if not missing_hours_in_x.empty:
    #     print("\nAdditional hours in Y without four 15-min intervals in X:")
    #     print(missing_hours_in_x)

    #Remove mismatched and missing hours from Y
    all_issues = mismatch_hours.index.union(missing_hours_in_x)
    Y_clean = Y[~Y.index.isin(all_issues)]

    #Re-check length conditions
    # print(f"\nAdjusted expected length of X_final: {4 * len(Y_clean)}")
    # print(f"Actual length of X_final: {len(X_final)}")


    #dropping nan columns:
    X_final.drop(columns=['cloud_base_agl:m'], inplace=True)
    X_final.drop(columns=['ceiling_height_agl:m'], inplace=True)
    X_final.drop(columns=['snow_density:kgm3'], inplace=True)

    X_final = create_features(X_final)
    X_final = create_time_features(X_final)
    X_final.drop(columns=['date_calc'], inplace=True)

    X_final = X_final.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    Y_clean = Y_clean.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))



    return X_final, Y_clean



def general_read_lstm(letter):

    df = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_train_observed.parquet")
    df2=pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_train_estimated.parquet")
    y = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/train_targets.parquet")
    # set the index to date_forecast and group by hourly frequency
    df.set_index("date_forecast", inplace=True)
    df2.set_index("date_forecast", inplace=True)
    y.set_index("time", inplace=True)

    df.index = pd.to_datetime(df.index)
    df2.index = pd.to_datetime(df2.index)
    y.index = pd.to_datetime(y.index) 
    
    df=pd.concat([df,df2],axis=0)

    # truncate y to match the index of df
    y = y.truncate(before=df.index[0], after=df.index[-1])
    latest_y_time = y.index[-1]
    latest_needed_df_time = latest_y_time + pd.Timedelta(minutes=45)
    # Truncate y based on df
    y = y.truncate(before=df.index[0], after=df.index[-1])
    # Ensure df has all needed entries from the start of y to 45 minutes after the end of y
    df = df.truncate(before=y.index[0], after=latest_needed_df_time)
    y.rename(columns={"pv_measurement":"target"},inplace=True)
    X = df.copy()
    Y = y.copy()
    #drop nan rows in Y
    Y = clean(Y)
    X.index = pd.to_datetime(X.index)
    Y.index = pd.to_datetime(Y.index)

    X_filtered = X[X.index.floor('H').isin(Y.index)]

    # Step 2: Ensure there are exactly four 15-min intervals for each hour
    valid_indices = X_filtered.groupby(X_filtered.index.floor('H')).filter(lambda group: len(group) == 4).index

    # Final filtered X
    X_final = X[X.index.isin(valid_indices)]


    #Troubleshooting: Find and print the hours with a mismatch
    group_sizes = X_filtered.groupby(X_filtered.index.floor('H')).size()
    mismatch_hours = group_sizes[group_sizes != 4]


    #Additional troubleshooting: find hours in Y without four 15-min intervals in X
    missing_hours_in_x = Y.index[~Y.index.isin(X_filtered.index.floor('H'))]


    #Remove mismatched and missing hours from Y
    all_issues = mismatch_hours.index.union(missing_hours_in_x)
    Y_clean = Y[~Y.index.isin(all_issues)]

    #dropping nan columns:
    X_final = X_final.drop(columns=['cloud_base_agl:m'])
    X_final = X_final.drop(columns=['ceiling_height_agl:m'])
    X_final = X_final.drop(columns=['snow_density:kgm3'])

    X_final = create_features(X_final)
    X_final = create_time_features(X_final)
    X_final.drop(columns=['date_calc'], inplace=True)

    X_final = X_final.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    Y_clean = Y_clean.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    return X_final, Y_clean


def concatenate_dfs(df_list):
    """
    Concatenates a list of DataFrames into a single DataFrame.

    Args:
    df_list (list of pd.DataFrame): List of DataFrame objects to concatenate.

    Returns:
    pd.DataFrame: A single DataFrame containing all rows from the input DataFrames in the order they appear in the list.
    """
    return pd.concat(df_list, ignore_index=False)
"""
----------------------------------------------------------------------------------------------------
"""


#Preprocessing transformers:
# class QuartersAsColumnsTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X, y=None):
#         # Ensure input is a DataFrame
#         X = X.copy()
#         assert isinstance(X, pd.DataFrame)
#         #make sure index is datetime:
#         X.index = pd.to_datetime(X.index)

#         original_index = X.index


#         X['hour'] = X.index.floor('H')
#         X['minute'] = X.index.minute

#         # Melt the DataFrame to long format
#         df_melted = pd.melt(X, id_vars=['hour', 'minute'], value_vars=X.columns[:-2]).copy()  # excluding 'hour' and 'minute'

#         # Create a multi-level column name combining variable and minute
#         df_melted['variable_minute'] = df_melted['variable'] + '_' + df_melted['minute'].astype(str) + 'min'

#         # Drop the 'variable_minute' column


#         # Pivot the data to get one row per hour and columns for each variable and minute
#         X = df_melted.pivot(index='hour', columns='variable_minute', values='value').copy()
#         #rename index to date_forecast:
#         X.index.rename("date_forecast", inplace=True)


#         #drop irrelevant columns:
#         #hour_sin	hour_cos	month_sin	month_cos	week_sin	week_cos	dayofyear_sin	dayofyear_cos	mult1	mult2	uncertainty

#         irrelevant_cols = [
#             "hour_sin", "hour_cos", "month_sin", "month_cos", "week_sin", 
#             "week_cos", "dayofyear_sin", "dayofyear_cos", "uncertainty"
#         ]
#         variantes = ["_0min", "_15min", "_30min", "_45min"]

#         # Collect columns to be dropped
#         columns_to_drop = []


#         for variant in variantes:
#             for col in irrelevant_cols:
#                 col_variant = col + variant
#                 if col_variant in X.columns:
#                     if variant == "_0min":
#                         # rename _0min from column name;
#                         X.rename(columns={col_variant: col}, inplace=True)
#                     else:
#                         # Add to the list of columns to drop
#                         columns_to_drop.append(col_variant)

#         # Drop all at once
#         X.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        

#         reindex_map = original_index.floor('H').unique()
#         X = X.reindex(reindex_map)
#         X.index = reindex_map

#         #drop hour_

#         # if "object" in X.dtypes.unique():
#         #     print("waring: object in QuarterAsColumnsTransformer")
#         #     print(X.dtypes.unique())
#         #     for col in X.columns:
#         #         print(col)

#         #X = X.select_dtypes(include=[np.number])
#         return X
class QuartersAsColumnsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Ensure input is a DataFrame
        X = X.copy()
        assert isinstance(X, pd.DataFrame)
        #make sure index is datetime:
        X.index = pd.to_datetime(X.index)

        original_index = X.index


        X['hour'] = X.index.floor('H')
        X['minute'] = X.index.minute

        # Melt the DataFrame to long format
        df_melted = pd.melt(X, id_vars=['hour', 'minute'], value_vars=X.columns[:-2]).copy()  # excluding 'hour' and 'minute'

        # Create a multi-level column name combining variable and minute
        df_melted['variable_minute'] = df_melted['variable'] + '_' + df_melted['minute'].astype(str) + 'min'

        # Drop the 'variable_minute' column


        # Pivot the data to get one row per hour and columns for each variable and minute
        X = df_melted.pivot(index='hour', columns='variable_minute', values='value').copy()
        #rename index to date_forecast:
        X.index.rename("date_forecast", inplace=True)


        #drop irrelevant columns:
        #hour_sin	hour_cos	month_sin	month_cos	week_sin	week_cos	dayofyear_sin	dayofyear_cos	mult1	mult2	uncertainty


        irrelevant_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "week_sin", "week_cos", "dayofyear_sin", "dayofyear_cos", "uncertainty"]
        variantes = ["_0min", "_15min", "_30min", "_45min"]
        for variant in variantes:
            for col in irrelevant_cols:
                if variant == "_0min":
                    #remove _0min from column name;
                    X.rename(columns={col+variant:col}, inplace=True)
                else:
                    X.drop(columns=[col+variant], inplace=True)
        

        reindex_map = original_index.floor('H').unique()
        X = X.reindex(reindex_map)
        X.index = reindex_map

        #drop hour_

        if "object" in X.dtypes.unique():
            print("waring: object in QuarterAsColumnsTransformer")
            print(X.dtypes.unique())
            for col in X.columns:
                print(col)

        #X = X.select_dtypes(include=[np.number])
        return X

class StatisticalFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy.index = pd.to_datetime(X_copy.index)
        X_copy['hour'] = X_copy.index.floor('H')
        
        # Compute mean, std
        aggregated = X_copy.groupby('hour').agg(['mean', 'std'])
        
        # Filter hours with exactly 4 data points
        valid_hours = X_copy.groupby('hour').size()
        valid_hours = valid_hours[valid_hours == 4].index
        
        X_final = aggregated.loc[valid_hours]
        
        # Flatten the multi-index to form new column names
        X_final.columns = ['_'.join(col).strip() for col in X_final.columns.values]
        # for col in X_final.columns:
        #     print(col)
        #drop minute_mean and minute_std if they exist:
        if "minute_mean" in X_final.columns:
            X_final.drop(columns=["minute_mean", "minute_std"], inplace=True)
        
        X_final = X_final.select_dtypes(include=[np.number])
        # print(X_final.dtypes.unique())
        # for col in X_final.columns:
        #     print(col)


        return X_final
    
class TrimmedMeanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Ensure the dataframe's index is a datetime type
        X = X.copy()
        X.index = pd.to_datetime(X.index)
        
        original_index = X.index

        # Create a helper column 'hour_label' 
        X['hour_label'] = X.index.floor('H')
        
        # Compute the trimmed mean for each valid hour
        def compute_trimmed_mean(group):
            if group.shape[0] != 4:  # Only process groups of size 4
                return np.nan

            # Exclude any datetime columns
            numeric_cols = group.select_dtypes(include=[np.number])
            
            min_val = np.min(numeric_cols, axis=0)
            max_val = np.max(numeric_cols, axis=0)
            total = np.sum(numeric_cols, axis=0)
            return (total - min_val - max_val) / 2  # Removing min and max

        # Group and apply the function
        X_trimmed_mean = X.groupby('hour_label').apply(compute_trimmed_mean)
        
        # Drop the helper column in the result as it's no longer needed
        if 'hour_label' in X_trimmed_mean.columns:
            X_trimmed_mean = X_trimmed_mean.drop(columns=['hour_label'])

        # Filter hours with exactly 4 data points
        valid_hours = X['hour_label'].value_counts()
        valid_hours = valid_hours[valid_hours == 4].index
        X_final = X_trimmed_mean[X_trimmed_mean.index.isin(valid_hours)]

        reindex_map = original_index.floor('H').unique()
        X_final = X_final.reindex(reindex_map)
        X_final.index = reindex_map

        X_final = X_final.select_dtypes(include=[np.number])
        
        return X_final

# class HourMonthTargetEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.encoding_map = {}

#     def fit(self, X, y):
#         X = X.copy()
#         # Ensure X's index is a datetime index
#         if not isinstance(X.index, pd.DatetimeIndex):
#             raise ValueError("Index of input X must be a pandas DatetimeIndex")



#         try:
#             # Extract hour and month from the index
#             df = pd.DataFrame({ 'target': y, 'hour': X.index.hour, 'month': X.index.month })
#         except:
#             try:
#                 y = pd.DataFrame(y)
#                 df = pd.DataFrame({ 'target': y, 'hour': X.index.hour, 'month': X.index.month })
#                 y = y["target"]
#             except:
#                 df 

#         # Compute mean target value for each hour of each month
#         self.encoding_map = df.groupby(['month', 'hour'])['target'].mean().to_dict()
#         return self

#     def transform(self, X):
#         # Ensure X's index is a datetime index
#         if not isinstance(X.index, pd.DatetimeIndex):
#             raise ValueError("Index of input X must be a pandas DatetimeIndex")

#         # Extract hour and month from the index
#         X_transformed = X.copy()
#         X_transformed['hour'] = X.index.hour
#         X_transformed['month'] = X.index.month

#         # Map the mean target values
#         X_transformed['target_encoded'] = X_transformed.apply(
#             lambda row: self.encoding_map.get((row['month'], row['hour']), np.nan), axis=1)

#         # You might want to drop the 'hour' and 'month' columns if they are not needed
#         X_transformed = X_transformed.drop(['hour', 'month'], axis=1)
#         #if it contains dtype dtype("O") print the columns:
#         if "object" in X_transformed.dtypes.unique():
#             print("waring: object in HourMonthTargetEncoder")
#             print(X_transformed.dtypes.unique())
#             print(X_transformed.columns)

#         X_transformed = X_transformed.select_dtypes(include=[np.number])
#         # print(X_transformed.dtypes.unique())

#         return X_transformed

class HourMonthTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoding_map = {}
        self.y_ = None  # To store y during fit

    def fit(self, X, y=None):
        # Ensure X's index is a datetime index
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Index of input X must be a pandas DatetimeIndex")

        if y is None:
            raise ValueError("y cannot be None for fitting the encoder")

        # Store the target values for encoding later
        self.y_ = y

        try:
            # Extract hour and month from the index and use y provided during fit
            df = pd.DataFrame({'target': self.y_, 'hour': X.index.hour, 'month': X.index.month})
        except Exception as e:
            raise e

        # Compute mean target value for each hour of each month
        self.encoding_map = df.groupby(['month', 'hour'])['target'].mean().to_dict()
        return self

    def transform(self, X):
        # Ensure X's index is a datetime index
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Index of input X must be a pandas DatetimeIndex")

        if self.y_ is None:
            raise ValueError("The encoder has not been fitted with target values")

        # Extract hour and month from the index
        X_transformed = X.copy()
        X_transformed['hour'] = X.index.hour
        X_transformed['month'] = X.index.month

        # Map the mean target values
        X_transformed['target_encoded'] = X_transformed.apply(
            lambda row: self.encoding_map.get((row['month'], row['hour']), np.nan), axis=1)

        # Optionally drop 'hour' and 'month' if they're not needed
        X_transformed.drop(['hour', 'month'], axis=1, inplace=True)

        # Check for object dtypes and print warning if any
        if "object" in X_transformed.dtypes.values:
            print("Warning: object dtype in HourMonthTargetEncoder")
            print(X_transformed.dtypes)

        # Ensure that only numeric types are returned
        X_transformed = X_transformed.select_dtypes(include=[np.number])

        return X_transformed




def apply_preprocessor(data, preprocessor_name):
    data = data.copy()
    # Assuming `pre.choose_transformer` returns a callable object that can be used to transform the data
    preprocessor = choose_transformer(preprocessor_name)
    return preprocessor.transform(data)

#Preprocessing functions for the different models:


#LSTM preprocessing:
def train_val_split_diffrent_folds(X,y,letter,fold_number):
    X = X.copy()
    y = y.copy()
    if letter == "A":
            assert fold_number in [0,1,2,3]
            year = 2019 + fold_number
    elif letter == "B":
        assert fold_number in [0,1,2]
        year = 2019 + fold_number
    elif letter == "C":
        assert fold_number in [0,1]
        year = 2020 + fold_number

    
    # Define conditions to move May and June of split_date's year from train to test
    may_june_july_condition_X = ((X.index.month == 5) | (X.index.month == 6) | (X.index.month == 7)) & ((X.index.year == year))
    may_june_july_condition_y = ((y.index.month == 5) | (y.index.month == 6) | (y.index.month == 7)) & ((y.index.year == year))
    
    X_val = X[may_june_july_condition_X]
    y_val = y[may_june_july_condition_y]

    # Remove May and June data from training set
    X_train = X[~may_june_july_condition_X]
    y_train = y[~may_june_july_condition_y]

    return X_train, y_train, X_val, y_val





def train_test_split_may_june_july(X, y,letter):
    """
    Splits the data based on a given date. Additionally, moves May, June and July data of split_date's year
    from training set to test set.
    
    Parameters:
    - X: Quarter-hourly input data with DateTime index.
    - y: Hourly target data with DateTime index.
    - split_date: Date (string or datetime object) to split the data on.
    
    Returns:
    X_train, y_train, X_test, y_test
    """

    if letter == "A":
        year = 2022
    elif letter == "B":
        year = 2019
    elif letter == "C":
        year = 2020
    
    # Define conditions to move May and June of split_date's year from train to test
    may_june_july_condition_X = ((X.index.month == 5) | (X.index.month == 6) | (X.index.month == 7)) & ((X.index.year == year))
    may_june_july_condition_y = ((y.index.month == 5) | (y.index.month == 6) | (y.index.month == 7)) & ((y.index.year == year))
    
    X_may_june_july = X[may_june_july_condition_X]
    y_may_june_july = y[may_june_july_condition_y]

    # Remove May and June data from training set
    X_train = X[~may_june_july_condition_X]
    y_train = y[~may_june_july_condition_y]

    return X_train, y_train, X_may_june_july, y_may_june_july

def train_val_blend(X, y,letter):
    X = X.copy()
    y = y.copy()
    if letter == "A":
        year = 2022
    elif letter == "B":
        year = 2019
    elif letter == "C":
        year = 2021


    if letter == "A":
        blend_year = 2021
    elif letter == "B":
        blend_year = 2020
    elif letter == "C":
        blend_year = 2020
    
    if letter == "C":
        # Define conditions to move May and June of split_date's year from train to test
        may_june_july_condition_X = ((X.index.month == 6) | (X.index.month == 5)) & ((X.index.year == year))
        may_june_july_condition_y = ((y.index.month == 6) | (y.index.month == 5)) & ((y.index.year == year))

    else:
        # Define conditions to move May and June of split_date's year from train to test
        may_june_july_condition_X = ((X.index.month == 5) | (X.index.month == 6) | (X.index.month == 7)) & ((X.index.year == year))
        may_june_july_condition_y = ((y.index.month == 5) | (y.index.month == 6) | (y.index.month == 7)) & ((y.index.year == year))
        
    X_val = X[may_june_july_condition_X]
    y_val = y[may_june_july_condition_y]

    if letter == "C":
        X_blend_condition = ((X.index.month == 7) | (X.index.month == 8)) & ((X.index.year == blend_year))
        y_blend_condition = ((y.index.month == 7) | (y.index.month == 8)) & ((y.index.year == blend_year))
    else:
        X_blend_condition = ((X.index.month == 5) | (X.index.month == 6) | (X.index.month == 7)) & ((X.index.year == blend_year))
        y_blend_condition = ((y.index.month == 5) | (y.index.month == 6) | (y.index.month == 7)) & ((y.index.year == blend_year))


    X_blend = X[X_blend_condition]
    y_blend = y[y_blend_condition]

    # Remove the data from training set
    X_train = X[~may_june_july_condition_X & ~X_blend_condition]
    y_train = y[~may_june_july_condition_y & ~y_blend_condition]


    return X_train, y_train, X_val, y_val, X_blend, y_blend





def train_test_split_on_specific_day_May_june(X, y, split_date):
    """
    Splits the data based on a given date. Additionally, moves May, June and July data of split_date's year
    from training set to test set.
    
    Parameters:
    - X: Quarter-hourly input data with DateTime index.
    - y: Hourly target data with DateTime index.
    - split_date: Date (string or datetime object) to split the data on.
    
    Returns:
    X_train, y_train, X_test, y_test
    """
    split_date = pd.Timestamp(split_date).normalize()

    # Ensure split_date is a datetime object
    if isinstance(split_date, str):
        split_date = pd.Timestamp(split_date)

    print(f"Split date: {split_date}")

    # Split the data based on the provided date
    X_train = X[X.index.normalize() < split_date]
    y_train = y[y.index.normalize() < split_date]

    X_test = X[X.index.normalize() >= split_date]
    y_test = y[y.index.normalize() >= split_date]

    # Define conditions to move May and June of split_date's year from train to test
    may_june_condition_X = ((X_train.index.month == 5) | (X_train.index.month == 6) | (X_train.index.month == 7)) & ((X_train.index.year == split_date.year))
    may_june_condition_y = ((y_train.index.month == 5) | (y_train.index.month == 6) | (y_train.index.month == 7)) & ((y_train.index.year == split_date.year))
    
    X_may_june = X_train[may_june_condition_X]
    y_may_june = y_train[may_june_condition_y]

    # Remove May and June data from training set
    X_train = X_train[~may_june_condition_X]
    y_train = y_train[~may_june_condition_y]

    # Append May and June data to test set
    X_test = pd.concat([X_may_june, X_test])
    y_test = pd.concat([y_may_june, y_test])

    return X_train, y_train, X_test, y_test








def time_series_split(X, split_date = "2022-10-29"):
    
    if not isinstance(X.index, pd.DatetimeIndex):
        X.index = pd.to_datetime(X.index)
    
    split_date = pd.to_datetime(split_date)
    
    mask_val = (X.index >= split_date)
    
    split_year = split_date.year
    mask_may_june_july = (X.index.month.isin([5, 6, 7])) & (X.index.year == split_year)
    
    mask_val = mask_val | mask_may_june_july

    
    test_fold = np.where(mask_val, 0, -1)
    
    
    return PredefinedSplit(test_fold)

def split_df_on_alternate_days(x_df, y_df):
    # Convert index to datetime if it's not already
    x_df.index = pd.to_datetime(x_df.index)
    y_df.index = pd.to_datetime(y_df.index)
    
    # Check if both dataframes are aligned
    assert all(x_df.index == y_df.index), "Indexes of x_df and y_df do not match!"

    # Extract day from the index
    days = x_df.index.day

    # Split into even and odd days
    x_even_days = x_df[days % 2 == 0]
    y_even_days = y_df[days % 2 == 0]

    x_odd_days = x_df[days % 2 != 0]
    y_odd_days = y_df[days % 2 != 0]

    return x_even_days, y_even_days, x_odd_days, y_odd_days

def lstm_train_test_split(X, y,letter, split_date):
    """
    Splits the data based on a given date. Additionally, moves May, June and July data of split_date's year
    from training set to test set.
    
    Parameters:
    - X: Quarter-hourly input data with DateTime index.
    - y: Hourly target data with DateTime index.
    - split_date: Date (string or datetime object) to split the data on.
    
    Returns:
    X_train, y_train, X_test, y_test
    """
    split_date = pd.Timestamp(split_date).normalize()

    if isinstance(split_date, str):
        split_date = pd.Timestamp(split_date)
    if letter == "A":
        year = 2022
    elif letter == "B":
        year = 2019
    elif letter == "C":
        year = 2020

    X_train = X[X.index.normalize() < split_date]
    y_train = y[y.index.normalize() < split_date]

    X_test = X[X.index.normalize() >= split_date]
    y_test = y[y.index.normalize() >= split_date]
    
    # Define conditions to move May and June of split_date's year from train to test
    may_june_july_condition_X = ((X.index.month == 5) | (X.index.month == 6) | (X.index.month == 7)) & ((X.index.year == year))
    may_june_july_condition_y = ((y.index.month == 5) | (y.index.month == 6) | (y.index.month == 7)) & ((y.index.year == year))
    
    X_may_june_july = X[may_june_july_condition_X]
    y_may_june_july = y[may_june_july_condition_y]

    # Remove May and June data from training set
    X_train = X[~may_june_july_condition_X]
    y_train = y[~may_june_july_condition_y]

    # Append May and June data to test set
    X_test = pd.concat([X_may_june_july, X_test])
    y_test = pd.concat([y_may_june_july, y_test])

    return X_train, y_train, X_test, y_test


def remove_winter_months(df):
    """
    Removes the winter months (December, January, February) from a DataFrame that
    has a DateTime index.

    Parameters:
    - df: DataFrame with DateTime index.

    Returns:
    - DataFrame with winter months removed.
    """
    # Ensure the index is a DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DateTimeIndex.")

    # Define condition to filter out the winter months
    winter_condition = (df.index.month == 12) | (df.index.month == 1) | (df.index.month == 2) | (df.index.month == 11)

    # Filter out the winter months
    df_no_winter = df[~winter_condition]
    return df_no_winter


def new_train_test_split(X, y,letter, split_date):
    """
    Splits the data based on a given date. Additionally, moves May, June and July data of split_date's year
    from training set to test set.
    
    Parameters:
    - X: Quarter-hourly input data with DateTime index.
    - y: Hourly target data with DateTime index.
    - split_date: Date (string or datetime object) to split the data on.
    
    Returns:
    X_train, y_train, X_test, y_test
    """
    split_date = pd.Timestamp(split_date).normalize()
    print(f"Split date: {split_date}")

    if isinstance(split_date, str):
        split_date = pd.Timestamp(split_date)
    if letter == "A":
        year = 2022
    elif letter == "B":
        year = 2019
    elif letter == "C":
        year = 2020

    X_train = X[X.index.normalize() < split_date]
    y_train = y[y.index.normalize() < split_date]

    X_test = X[X.index.normalize() >= split_date]
    y_test = y[y.index.normalize() >= split_date]
    
    # Define conditions to move May and June of split_date's year from train to test
    may_june_july_condition_X = ((X.index.month == 5) | (X.index.month == 6) | (X.index.month == 7)) & ((X.index.year == year))
    may_june_july_condition_y = ((y.index.month == 5) | (y.index.month == 6) | (y.index.month == 7)) & ((y.index.year == year))
    
    X_may_june_july = X[may_june_july_condition_X]
    y_may_june_july = y[may_june_july_condition_y]

    # Remove May and June data from training set
    X_train = X[~may_june_july_condition_X]
    y_train = y[~may_june_july_condition_y]

    # Append May and June data to test set
    X_test = pd.concat([X_may_june_july, X_test])
    y_test = pd.concat([y_may_june_july, y_test])

    return X_train, y_train, X_test, y_test


def choose_scaler(scaler_string):
    if scaler_string == "minmax":
        return MinMaxScaler()
    elif scaler_string == "standard":
        return StandardScaler()
    elif scaler_string == "robust":
        return RobustScaler()

def choose_transformer(transformer_string):
    if transformer_string == "quarters":
        return QuartersAsColumnsTransformer()
    elif transformer_string == "statistical":
        return StatisticalFeaturesTransformer()
    elif transformer_string == "trimmedMean":
        return TrimmedMeanTransformer()

def choose_encoder(encoder_boolian):
    if encoder_boolian == True:
        return HourMonthTargetEncoder()
    else:
        return None

def generate_predefined_split(X_train, X_val, y_train, y_val):
    """
    This function takes in separate training and validation datasets, combines them,
    and creates a PredefinedSplit object that can be used with sklearn's GridSearchCV
    or other model selection utilities. This allows for specifying which samples are
    used for training and which are used for validation.

    Parameters:
    X_train (array-like): Training features.
    X_val (array-like): Validation features.
    y_train (array-like): Training labels.
    y_val (array-like): Validation labels.

    Returns:
    X (array-like): The combined dataset of features.
    y (array-like): The combined dataset of labels.
    split_index (PredefinedSplit): An instance of PredefinedSplit with the indices set.
    """

    # Combine the training and validation sets
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    # Generate the indices array where -1 indicates the sample is part of the training set,
    # and 0 indicates the sample is part of the validation set.
    train_indices = -1 * np.ones(len(X_train))
    val_indices = 0 * np.ones(len(X_val))
    test_fold = np.concatenate((train_indices, val_indices))

    # Create the PredefinedSplit object
    predefined_split = PredefinedSplit(test_fold)

    return X, y, predefined_split

def printhei():
    print("hei")
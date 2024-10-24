import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame(): # type: ignore
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    if df is None:
        file_path = input("Please enter the dataset file path (e.g., 'dataset-2.csv'): ")
        df = pd.read_csv(file_path)

    toll_ids = np.sort(pd.unique(df[['id_start', 'id_end']].values.ravel('K')))
    
    distance_matrix = pd.DataFrame(np.inf, index=toll_ids, columns=toll_ids)# type: ignore

    np.fill_diagonal(distance_matrix.values, 0)

    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance  

    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                distance_matrix.at[i, j] = min(
                    distance_matrix.at[i, j],
                    distance_matrix.at[i, k] + distance_matrix.at[k, j]
                )

    print("\nDistance Matrix:")
    print(distance_matrix)
    return distance_matrix
    return df
calculate_distance_matrix()




def unroll_distance_matrix(df)->pd.DataFrame(): # type: ignore
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []

    for i in df.index: # type: ignore
        for j in df.columns: # type: ignore
            if i != j: 
                unrolled_data.append((i, j, df.at[i, j])) # type: ignore

    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    return unrolled_df
    return df
unrolled_df = unroll_distance_matrix(distance_matrix) # type: ignore
print(unrolled_df)

    


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame(): # type: ignore
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    reference_avg_distance = (
        df[df['id_start'] == reference_id]['distance'].mean() # type: ignore
    )

    lower_bound = reference_avg_distance * 0.9  
    upper_bound = reference_avg_distance * 1.1  

    valid_ids = (
        df.groupby('id_start')['distance']
        .mean()
        .reset_index()
    )

    within_threshold = valid_ids[
        (valid_ids['distance'] >= lower_bound) & (valid_ids['distance'] <= upper_bound)
    ]['id_start']

    return sorted(within_threshold.tolist())

    return df
result = find_ids_within_ten_percentage_threshold(unrolled_df, 1001400)
print(result)



def calculate_toll_rate(df)->pd.DataFrame(): # type: ignore
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient

    return df
updated_df = calculate_toll_rate(unrolled_df)
print(updated_df)




def calculate_time_based_toll_rates(df)->pd.DataFrame(): # type: ignore
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    weekday_factors = [
        (dt.time(0, 0, 0), dt.time(10, 0, 0), 0.8),
        (dt.time(10, 0, 0), dt.time(18, 0, 0), 1.2),
        (dt.time(18, 0, 0), dt.time(23, 59, 59), 0.8)
    ]
    weekend_factor = 0.7


    results = []

    for index, row in df.iterrows(): # type: ignore
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        for day in weekdays + weekends:
            for start_time, end_time, factor in weekday_factors:
                if day in weekends:  
                    factor = weekend_factor


                new_rates = {
                    'moto': row['moto'] * factor,
                    'car': row['car'] * factor,
                    'rv': row['rv'] * factor,
                    'bus': row['bus'] * factor,
                    'truck': row['truck'] * factor,
                }

                results.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **new_rates
                })

    time_based_toll_df = pd.DataFrame(unrolled_df)

    return time_based_toll_df

    return df
time_based_toll_df = calculate_time_based_toll_rates(updated_df)
print(time_based_toll_df)


import pandas as pd
import numpy as np


def calculate_distance_matrix(df)->pd.DataFrame():  # type: ignore
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    if df is None:
        file_path = input("Please enter the dataset file path (e.g., 'dataset-2.csv'): ")
        df = pd.read_csv(file_path)

    toll_ids = np.sort(pd.unique(df[['id_start', 'id_end']].values.ravel('K')))
    
    distance_matrix = pd.DataFrame(np.inf, index=toll_ids, columns=toll_ids)# type: ignore

    np.fill_diagonal(distance_matrix.values, 0)

    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance  

    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                distance_matrix.at[i, j] = min(
                    distance_matrix.at[i, j],
                    distance_matrix.at[i, k] + distance_matrix.at[k, j]
                )

    print("\nDistance Matrix:")
    print(distance_matrix)
    return distance_matrix
    return df
calculate_distance_matrix('C:\Users\Madara uchiha\Documents\GitHub\MapUp-DA-Assessment-2024\datasets\dataset-2.csv')

def unroll_distance_matrix(df)->pd.DataFrame(): # type: ignore
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []

    for i in df.index: # type: ignore
        for j in df.columns: # type: ignore
            if i != j: 
                unrolled_data.append((i, j, df.at[i, j])) # type: ignore

    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    return unrolled_df
    return df
unrolled_df = unroll_distance_matrix(distance_matrix) # type: ignore
print(unrolled_df)



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame(): # type: ignore
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    reference_avg_distance = (
        df[df['id_start'] == reference_id]['distance'].mean() # type: ignore
    )

    lower_bound = reference_avg_distance * 0.9  
    upper_bound = reference_avg_distance * 1.1  

    valid_ids = (
        df.groupby('id_start')['distance']
        .mean()
        .reset_index()
    )

    within_threshold = valid_ids[
        (valid_ids['distance'] >= lower_bound) & (valid_ids['distance'] <= upper_bound)
    ]['id_start']

    return sorted(within_threshold.tolist())

    return df
result = find_ids_within_ten_percentage_threshold(unrolled_df, 1001400)
print(result)


def calculate_toll_rate(df)->pd.DataFrame(): # type: ignore
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient

    return df
updated_df = calculate_toll_rate(unrolled_df)
print(updated_df)

import datetime as dt
def calculate_time_based_toll_rates(df)->pd.DataFrame(): # type: ignore
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    weekday_factors = [
        (dt.time(0, 0, 0), dt.time(10, 0, 0), 0.8),
        (dt.time(10, 0, 0), dt.time(18, 0, 0), 1.2),
        (dt.time(18, 0, 0), dt.time(23, 59, 59), 0.8)
    ]
    weekend_factor = 0.7


    results = []

    for index, row in df.iterrows(): # type: ignore
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        for day in weekdays + weekends:
            for start_time, end_time, factor in weekday_factors:
                if day in weekends:  
                    factor = weekend_factor


                new_rates = {
                    'moto': row['moto'] * factor,
                    'car': row['car'] * factor,
                    'rv': row['rv'] * factor,
                    'bus': row['bus'] * factor,
                    'truck': row['truck'] * factor,
                }

                results.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **new_rates
                })

    time_based_toll_df = pd.DataFrame(unrolled_df)

    return time_based_toll_df

    return df
time_based_toll_df = calculate_time_based_toll_rates(updated_df)
print(time_based_toll_df)

import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    locations = pd.unique(df[['From', 'To']].values.ravel())
    locations.sort()  
    n = len(locations)
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    for _, row in df.iterrows():
        distance_matrix.at[row["From"], row["To"]] = row["Distance"]
        distance_matrix.at[row["To"], row["From"]] = row["Distance"]  
    np.fill_diagonal(distance_matrix.values, 0)
    for k in locations:
        for i in locations:
            for j in locations:
                distance_matrix.at[i, j] = min(
                    distance_matrix.at[i, j],
                    distance_matrix.at[i, k] + distance_matrix.at[k, j]
                )
    return distance_matrix
data = {
    "From": [
        1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446,
        1001448, 1001450, 1001452, 1001454, 1001456, 1001458, 1001460,
        1001460, 1001461, 1001462, 1001464, 1001466, 1001468, 1001470
    ],
    "To": [
        1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 1001448,
        1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001461,
        1001462, 1001462, 1001464, 1001466, 1001468, 1001470, 1001472
    ],
    "Distance": [
        4.5, 4, 2, 2, 0.7, 6.6, 9.6, 15.7, 9.9, 11.3, 13.6, 8.9, 5.1, 
        12.8, 17.9, 5.1, 26.7, 8.5, 10.7, 10.6, 16
    ]
}
df = pd.DataFrame(data)
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)





import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a cumulative distance matrix using the Floyd-Warshall algorithm.
    """
    locations = pd.unique(df[['From', 'To']].values.ravel())
    locations.sort() 
    n = len(locations)
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    for _, row in df.iterrows():
        distance_matrix.at[row["From"], row["To"]] = row["Distance"]
        distance_matrix.at[row["To"], row["From"]] = row["Distance"] 
    np.fill_diagonal(distance_matrix.values, 0)
    for k in locations:
        for i in locations:
            for j in locations:
                distance_matrix.at[i, j] = min(
                    distance_matrix.at[i, j],
                    distance_matrix.at[i, k] + distance_matrix.at[k, j]
                )

    return distance_matrix

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix into a DataFrame with 'id_start', 'id_end', and 'distance' columns.
    """
    unrolled_df = df.reset_index().melt(id_vars='index', var_name='id_end', value_name='distance')
    unrolled_df = unrolled_df.rename(columns={'index': 'id_start'})
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]

    return unrolled_df
data = {
    "From": [
        1001442, 1001488, 1004356, 1004354, 1004355, 1001444, 1001446,
        1001448, 1001450, 1001452, 1001454, 1001456, 1001458, 1001460,
        1001460, 1001461, 1001462, 1001464, 1001466, 1001468, 1001470
    ],
    "To": [
        1001488, 1004356, 1004354, 1004355, 1001444, 1001446, 1001448,
        1001450, 1001452, 1001454, 1001456, 1001458, 1001460, 1001461,
        1001462, 1001462, 1001464, 1001466, 1001468, 1001470, 1001472
    ],
    "Distance": [
        4.5, 4, 2, 2, 0.7, 6.6, 9.6, 15.7, 9.9, 11.3, 13.6, 8.9, 5.1, 
        12.8, 17.9, 5.1, 26.7, 8.5, 10.7, 10.6, 16
    ]
}
df = pd.DataFrame(data)
distance_matrix = calculate_distance_matrix(df)
unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df.head())






import pandas as pd

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing id_start and distance columns.
        reference_id (int): The ID for which the average distance is used for comparison.
        
    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified 
                          percentage threshold of the reference ID's average distance.
    """
    if 'id_start' not in df.columns or 'distance' not in df.columns:
        raise ValueError("DataFrame must contain 'id_start' and 'distance' columns.")
    avg_distance_ref = df.loc[df['id_start'] == reference_id, 'distance'].mean()
    if pd.isna(avg_distance_ref):
        return pd.DataFrame(columns=['id_start', 'avg_distance'])
    lower_bound = avg_distance_ref * 0.9
    upper_bound = avg_distance_ref * 1.1
    avg_distances = df.groupby('id_start', as_index=False)['distance'].mean()
    avg_distances.columns = ['id_start', 'avg_distance']
    result = avg_distances[(avg_distances['avg_distance'] >= lower_bound) & 
                           (avg_distances['avg_distance'] <= upper_bound)]
    result = result.sort_values(by='id_start')
    return result
df = pd.DataFrame({
    'id_start': [1, 1, 2, 2, 3, 3],
    'distance': [10, 12, 11, 13, 15, 14]
})
result = find_ids_within_ten_percentage_threshold(df, 1)
print(result)  





import pandas as pd

def calculate_toll_rate(df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df
data = {
    'id_start': [1001400] * 9,
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418],
    'distance': [9.7, 29.9, 45.9, 67.6, 78.7, 94.3, 112.5, 125.7, 139.3]
}
df = pd.DataFrame(data)
result_df = calculate_toll_rate(df)
print(result_df)





import pandas as pd
from datetime import time

def apply_discount(rate, discount_factor):
    """Apply a discount factor to the original rate."""
    return rate * discount_factor

def calculate_time_based_toll_rates(df):
    def get_discount_factor(day, start_time):
        if day in ['Saturday', 'Sunday']:
            return 0.7 

        if time(0, 0, 0) <= start_time < time(10, 0, 0):
            return 0.8 
        elif time(10, 0, 0) <= start_time < time(18, 0, 0):
            return 1.2  
        elif time(18, 0, 0) <= start_time <= time(23, 59, 59):
            return 0.8  
        return 1.0 
    updated_rows = []
    for _, row in df.iterrows():
        start_day = row['start_day']
        start_time = row['start_time']
        discount_factor = get_discount_factor(start_day, start_time)
        updated_row = row.copy()
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            updated_row[vehicle] = apply_discount(row[vehicle], discount_factor)
        updated_rows.append(updated_row)
    result_df = pd.DataFrame(updated_rows)
    return result_df
data = {
    'id_start': [1001400, 1001400, 1001408, 1001408],
    'id_end': [1001402, 1001402, 1001410, 1001410],
    'distance': [9.7, 9.7, 11.1, 11.1],
    'start_day': ['Monday', 'Tuesday', 'Saturday', 'Sunday'],
    'start_time': [time(0, 0, 0), time(10, 0, 0), time(0, 0, 0), time(18, 0, 0)],
    'end_day': ['Friday', 'Saturday', 'Sunday', 'Sunday'],
    'end_time': [time(10, 0, 0), time(18, 0, 0), time(23, 59, 59), time(23, 59, 59)],
    'moto': [6.21, 9.31, 5.43, 6.22],
    'car': [9.31, 13.97, 8.15, 9.32],
    'rv': [11.64, 17.46, 10.19, 11.66],
    'bus': [17.07, 25.61, 14.94, 17.09],
    'truck': [27.94, 41.90, 24.44, 27.97]
}
df = pd.DataFrame(data)
result_df = calculate_time_based_toll_rates(df)
print(result_df)

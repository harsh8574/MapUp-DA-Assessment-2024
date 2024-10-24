# import pandas as pd
# import numpy as np
# from datetime import datetime, time, timedelta

# def calculate_distance_matrix(df)->pd.DataFrame:
#     """
#     Calculate a distance matrix based on the dataframe, df.
#     """
#     # Create a set of all unique IDs
#     all_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    
#     # Initialize the distance matrix with zeros
#     dist_matrix = pd.DataFrame(0.0, index=all_ids, columns=all_ids)
    
#     # Fill the matrix with direct distances
#     for _, row in df.iterrows():
#         dist_matrix.loc[row['id_start'], row['id_end']] = row['distance']
#         dist_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Make symmetric
    
#     # Floyd-Warshall algorithm for shortest paths
#     for k in all_ids:
#         for i in all_ids:
#             for j in all_ids:
#                 if dist_matrix.loc[i,j] == 0 and i != j:
#                     dist_matrix.loc[i,j] = dist_matrix.loc[i,k] + dist_matrix.loc[k,j]
#                 else:
#                     new_dist = dist_matrix.loc[i,k] + dist_matrix.loc[k,j]
#                     if new_dist < dist_matrix.loc[i,j] or dist_matrix.loc[i,j] == 0:
#                         if i != j and new_dist > 0:
#                             dist_matrix.loc[i,j] = new_dist
    
#     return dist_matrix

# def unroll_distance_matrix(df)->pd.DataFrame:
#     """
#     Unroll a distance matrix to a DataFrame with id_start, id_end, and distance.
#     """
#     # Create all combinations of IDs
#     id_start = []
#     id_end = []
#     distances = []
    
#     for i in df.index:
#         for j in df.columns:
#             if i != j:  # Exclude same ID combinations
#                 id_start.append(i)
#                 id_end.append(j)
#                 distances.append(df.loc[i,j])
    
#     # Create the unrolled DataFrame
#     unrolled_df = pd.DataFrame({
#         'id_start': id_start,
#         'id_end': id_end,
#         'distance': distances
#     })
    
#     return unrolled_df

# def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame:
#     """
#     Find IDs within 10% threshold of reference ID's average distance.
#     """
#     # Calculate average distance for reference ID
#     ref_avg = df[df['id_start'] == reference_id]['distance'].mean()
    
#     # Calculate threshold bounds
#     lower_bound = ref_avg * 0.9
#     upper_bound = ref_avg * 1.1
    
#     # Calculate average distance for each id_start
#     avg_distances = df.groupby('id_start')['distance'].mean()
    
#     # Find IDs within threshold
#     within_threshold = avg_distances[
#         (avg_distances >= lower_bound) & 
#         (avg_distances <= upper_bound)
#     ].index.tolist()
    
#     # Sort the result
#     within_threshold.sort()
    
#     return within_threshold

# def calculate_toll_rate(df)->pd.DataFrame:
#     """
#     Calculate toll rates for different vehicle types.
#     """
#     # Define rate coefficients
#     rate_coefficients = {
#         'moto': 0.8,
#         'car': 1.2,
#         'rv': 1.5,
#         'bus': 2.2,
#         'truck': 3.6
#     }
    
#     # Calculate toll rates for each vehicle type
#     result_df = df.copy()
#     for vehicle, rate in rate_coefficients.items():
#         result_df[vehicle] = result_df['distance'] * rate
        
#     return result_df

# def calculate_time_based_toll_rates(df)->pd.DataFrame:
#     """
#     Calculate time-based toll rates for different time intervals.
#     """
#     # Create time intervals for weekdays
#     weekday_intervals = [
#         (time(0, 0, 0), time(10, 0, 0), 0.8),
#         (time(10, 0, 0), time(18, 0, 0), 1.2),
#         (time(18, 0, 0), time(23, 59, 59), 0.8)
#     ]
    
#     # Create the result DataFrame
#     rows = []
    
#     # Get unique (id_start, id_end) pairs
#     unique_pairs = df[['id_start', 'id_end', 'distance']].drop_duplicates()
    
#     days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
#     for _, pair in unique_pairs.iterrows():
#         for day in days:
#             if day in ['Saturday', 'Sunday']:
#                 # Weekend rates
#                 row = {
#                     'id_start': pair['id_start'],
#                     'id_end': pair['id_end'],
#                     'distance': pair['distance'],
#                     'start_day': day,
#                     'start_time': time(0, 0, 0),
#                     'end_day': day,
#                     'end_time': time(23, 59, 59)
#                 }
                
#                 # Calculate vehicle rates with weekend discount
#                 for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
#                     base_rate = df[df['id_start'] == pair['id_start']][vehicle].iloc[0]
#                     row[vehicle] = base_rate * 0.7
                
#                 rows.append(row)
            
#             else:
#                 # Weekday rates
#                 for start_time, end_time, factor in weekday_intervals:
#                     row = {
#                         'id_start': pair['id_start'],
#                         'id_end': pair['id_end'],
#                         'distance': pair['distance'],
#                         'start_day': day,
#                         'start_time': start_time,
#                         'end_day': day,
#                         'end_time': end_time
#                     }
                    
#                     # Calculate vehicle rates with time-based discount
#                     for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
#                         base_rate = df[df['id_start'] == pair['id_start']][vehicle].iloc[0]
#                         row[vehicle] = base_rate * factor
                    
#                     rows.append(row)
    
#     result_df = pd.DataFrame(rows)
#     return result_df


import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

def calculate_distance_matrix(df)->pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.
    """
    # Create a set of all unique IDs
    ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    
    # Initialize the distance matrix with infinity
    n = len(ids)
    dist_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
    np.fill_diagonal(dist_matrix.values, 0)  # Set diagonal to 0
    
    # Fill known distances
    for _, row in df.iterrows():
        dist_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        dist_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Symmetric matrix
    
    # Floyd-Warshall algorithm
    for k in ids:
        for i in ids:
            for j in ids:
                if dist_matrix.loc[i,k] != np.inf and dist_matrix.loc[k,j] != np.inf:
                    dist_matrix.loc[i,j] = min(
                        dist_matrix.loc[i,j],
                        dist_matrix.loc[i,k] + dist_matrix.loc[k,j]
                    )
    
    # Replace any remaining inf values with 0
    dist_matrix = dist_matrix.replace(np.inf, 0)
    
    return dist_matrix

def unroll_distance_matrix(df)->pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame with id_start, id_end, and distance.
    """
    # Get all combinations of IDs where distance > 0
    unrolled_data = []
    
    for id_start in df.index:
        for id_end in df.columns:
            distance = df.loc[id_start, id_end]
            if id_start != id_end and distance > 0:  # Exclude same IDs and zero distances
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance
                })
    
    return pd.DataFrame(unrolled_data)

def find_ids_within_ten_percentage_threshold(df, reference_id)->list:
    """
    Find IDs within 10% threshold of reference ID's average distance.
    """
    # Calculate average distance for each id_start
    avg_distances = df.groupby('id_start')['distance'].mean()
    
    # Get reference ID's average distance
    reference_avg = avg_distances[reference_id]
    
    # Calculate thresholds
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1
    
    # Find IDs within threshold
    valid_ids = avg_distances[
        (avg_distances >= lower_bound) & 
        (avg_distances <= upper_bound)
    ].index.tolist()
    
    return sorted(valid_ids)

def calculate_toll_rate(df)->pd.DataFrame:
    """
    Calculate toll rates for different vehicle types.
    """
    # Create a copy of input DataFrame
    result_df = df.copy()
    
    # Define vehicle rate coefficients
    vehicle_rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate rates for each vehicle type
    for vehicle, rate in vehicle_rates.items():
        result_df[vehicle] = result_df['distance'] * rate
    
    return result_df

def calculate_time_based_toll_rates(df)->pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals.
    """
    def get_time_factor(day, time_obj):
        if day in ['Saturday', 'Sunday']:
            return 0.7
        else:
            if time(0,0,0) <= time_obj < time(10,0,0):
                return 0.8
            elif time(10,0,0) <= time_obj < time(18,0,0):
                return 1.2
            else:
                return 0.8
    
    # Create time intervals
    time_intervals = [
        (time(0,0,0), time(10,0,0)),
        (time(10,0,0), time(18,0,0)),
        (time(18,0,0), time(23,59,59))
    ]
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Get unique id pairs
    unique_pairs = df[['id_start', 'id_end', 'distance']].drop_duplicates()
    
    rows = []
    for _, pair in unique_pairs.iterrows():
        for day in days:
            if day in ['Saturday', 'Sunday']:
                # Weekend - single entry for whole day
                row = {
                    'id_start': pair['id_start'],
                    'id_end': pair['id_end'],
                    'distance': pair['distance'],
                    'start_day': day,
                    'end_day': day,
                    'start_time': time(0,0,0),
                    'end_time': time(23,59,59)
                }
                
                # Calculate vehicle rates
                factor = 0.7  # Weekend discount
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    base_rate = df[df['id_start'] == pair['id_start']][vehicle].iloc[0]
                    row[vehicle] = base_rate * factor
                
                rows.append(row)
            
            else:
                # Weekday - separate entries for different time intervals
                for start_t, end_t in time_intervals:
                    row = {
                        'id_start': pair['id_start'],
                        'id_end': pair['id_end'],
                        'distance': pair['distance'],
                        'start_day': day,
                        'end_day': day,
                        'start_time': start_t,
                        'end_time': end_t
                    }
                    
                    # Calculate vehicle rates
                    factor = get_time_factor(day, start_t)
                    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                        base_rate = df[df['id_start'] == pair['id_start']][vehicle].iloc[0]
                        row[vehicle] = base_rate * factor
                    
                    rows.append(row)
    
    result_df = pd.DataFrame(rows)
    return result_df
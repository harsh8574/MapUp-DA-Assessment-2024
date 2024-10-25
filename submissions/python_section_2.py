# Question 9:

import pandas as pd

import pandas as pd

file_path = r'C:\Users\DELL\MapUp-DA-Assessment-2024\datasets\dataset-2.csv'

data = pd.read_csv(file_path)

print(data.head())

import numpy as np

# Get unique toll location IDs
ids = sorted(pd.unique(df[['id_start', 'id_end']].values.ravel('K')))

# Initialize a distance matrix with infinity (large value)
distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)

# Set diagonal values (distance to itself) to 0
np.fill_diagonal(distance_matrix.values, 0)

# Populate the matrix with direct distances from the CSV
for _, row in df.iterrows():
    distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
    distance_matrix.at[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

# Display the initial matrix with direct distances
distance_matrix

# Apply the Floyd-Warshall algorithm to find the shortest paths
for k in ids:
    for i in ids:
        for j in ids:
            # Update the distance if a shorter path is found through intermediary point k
            distance_matrix.at[i, j] = min(distance_matrix.at[i, j], distance_matrix.at[i, k] + distance_matrix.at[k, j])

# Display the final updated distance matrix
distance_matrix





# Question 10 :

def unroll_distance_matrix(distance_matrix):
    # Initialize a list to store the rows
    unrolled_data = []
    
    # Loop through the matrix to extract the id_start, id_end, and distance values
    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            if i != j:  # Exclude the diagonal (same start and end points)
                unrolled_data.append({'id_start': i, 'id_end': j, 'distance': distance_matrix.at[i, j]})
    
    # Convert the list to a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df

# Apply the function to the distance matrix from question 9
unrolled_df = unroll_distance_matrix(distance_matrix_image)

# Display the unrolled DataFrame
unrolled_df.head(10)  # Show the first 10 rows to match the format in the attached sample image

# Question 11 :
def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Filter the rows where id_start matches the reference ID
    ref_distances = df[df['id_start'] == reference_id]['distance']
    
    # Calculate the average distance for the reference ID
    ref_avg_distance = ref_distances.mean()
    
    # Calculate the 10% threshold range
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1
    
    # Find all unique id_start values
    ids_within_threshold = []
    for toll_id in df['id_start'].unique():
        # Calculate the average distance for each id_start
        avg_distance = df[df['id_start'] == toll_id]['distance'].mean()
        
        # Check if the average distance falls within the 10% threshold range
        if lower_bound <= avg_distance <= upper_bound:
            ids_within_threshold.append(toll_id)
    
    # Return the sorted list of ids within the threshold
    return sorted(ids_within_threshold)

# Example usage with reference ID 1001400
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, 1001400)

# Display the result
ids_within_threshold


#Question 12 :
import pandas as pd

import pandas as pd

file_path = r'C:\Users\DELL\MapUp-DA-Assessment-2024\datasets\dataset-2.csv'

data = pd.read_csv(file_path)

print(data.head())

# Define rate coefficients for each vehicle type
rate_coefficients = {
    'moto': 0.8,
    'car': 1.2,
    'rv': 1.5,
    'bus': 2.2,
    'truck': 3.6
}

# Calculate toll rates for each vehicle type by multiplying the distance with the corresponding rate coefficient
df['moto'] = df['distance'] * rate_coefficients['moto']
df['car'] = df['distance'] * rate_coefficients['car']
df['rv'] = df['distance'] * rate_coefficients['rv']
df['bus'] = df['distance'] * rate_coefficients['bus']
df['truck'] = df['distance'] * rate_coefficients['truck']

# Display the updated DataFrame
df.head()


#Question 13:

import datetime as dt

# Sample data for start_day, start_time, end_day, and end_time based on user inputs
# This would usually come from the CSV, but since it's not there, we create a mock dataset with similar structure

# Define a function to apply the time-based toll adjustments
def calculate_time_based_toll_rates(df):
    def get_discount_factor(day, time):
        """
        Determine the discount factor based on day and time.
        Weekdays (Monday - Friday) have different time-based discounts, while
        weekends (Saturday and Sunday) have a constant discount factor.
        """
        if day in ['Saturday', 'Sunday']:
            return 0.7
        else:
            if dt.time(0, 0) <= time <= dt.time(10, 0):
                return 0.8
            elif dt.time(10, 0) < time <= dt.time(18, 0):
                return 1.2
            else:
                return 0.8
    
    # Create time objects from start_time and end_time strings
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time
    
    # Apply discount factors to the vehicle columns based on day and time
    for index, row in df.iterrows():
        discount_factor = get_discount_factor(row['start_day'], row['start_time'])
        df.at[index, 'moto'] *= discount_factor
        df.at[index, 'car'] *= discount_factor
        df.at[index, 'rv'] *= discount_factor
        df.at[index, 'bus'] *= discount_factor
        df.at[index, 'truck'] *= discount_factor
    
    return df

# Sample dataframe from previous steps (mock data)
df_sample = pd.DataFrame({
    'id_start': [1001400, 1001400, 1001408, 1001408],
    'id_end': [1001402, 1001402, 1001410, 1001410],
    'distance': [9.7, 9.7, 11.1, 11.1],
    'start_day': ['Monday', 'Tuesday', 'Monday', 'Wednesday'],
    'start_time': ['00:00:00', '10:00:00', '00:00:00', '18:00:00'],
    'end_day': ['Friday', 'Saturday', 'Friday', 'Sunday'],
    'end_time': ['10:00:00', '18:00:00', '10:00:00', '23:59:59'],
    'moto': [6.21, 9.31, 7.10, 6.22],
    'car': [9.31, 13.97, 10.66, 9.32],
    'rv': [11.64, 17.46, 13.32, 11.66],
    'bus': [17.07, 25.61, 19.54, 17.09],
    'truck': [27.94, 41.90, 31.97, 27.97]
})

# Apply the time-based toll rate calculation function to the DataFrame
df_time_based = calculate_time_based_toll_rates(df_sample)
df_time_based.head()

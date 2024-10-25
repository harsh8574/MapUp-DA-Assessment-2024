from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    
    # Process the list in chunks of size n
    for i in range(0, len(lst), n):
        # Create a temporary list for the current chunk
        temp = []
        for j in range(min(n, len(lst) - i)):
            # Add elements to the temporary list
            temp.append(lst[i + j])
        # Reverse the temporary list and add to the result
        for k in range(len(temp) - 1, -1, -1):
            result.append(temp[k])
    
    return result

# Example test cases
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3)) 
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))    
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    
    # Iterate over each string in the list
    for string in lst:
        length = len(string)
        
        # If the length is not a key in the result, add it
        if length not in result:
            result[length] = []
        
        # Add the string to the corresponding length list
        result[length].append(string)
    
    # Sort the dictionary by keys (lengths) in ascending order
    sorted_result = dict(sorted(result.items()))
    
    return sorted_result

# Example test cases
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))  
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}



from typing import Dict, Any

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten(current, parent_key=''):
        items = {}
        
        # Iterate over the dictionary or list
        if isinstance(current, dict):
            for key, value in current.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                # Recursively flatten nested dictionaries
                items.update(flatten(value, new_key))
        elif isinstance(current, list):
            for index, value in enumerate(current):
                new_key = f"{parent_key}[{index}]"
                # Recursively flatten nested lists
                items.update(flatten(value, new_key))
        else:
            # Base case: no more nesting, add the value
            items[parent_key] = current
        
        return items

    # Call the recursive function and return the flattened dictionary
    return flatten(nested_dict)


print(flatten_dict(input_data))  
# Output: {'road.name': 'Highway 1', 'road.length': 350, 'road.sections[0].id': 1, 'road.sections[0].condition.pavement': 'good', 'road.sections[0].condition.traffic': 'moderate'}



from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            # If we reach the end of the list, add the current permutation to the results
            result.append(nums[:])
            return
        
        seen = set()  # To track used numbers at this level
        for i in range(start, len(nums)):
            if nums[i] in seen:  # Skip duplicates
                continue
            seen.add(nums[i])
            # Swap the current number with the start index
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)  # Recurse on the next index
            # Backtrack: swap back
            nums[start], nums[i] = nums[i], nums[start]

    result = []
    nums.sort()  # Sort to handle duplicates
    backtrack(0)
    return result

# Example test case
print(unique_permutations([1, 1, 2]))  
# Output: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]


import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Regular expression pattern to match the different date formats
    pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    # Find all matching dates using the regex pattern
    matches = re.findall(pattern, text)
    
    return matches

# Example test case
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))  
# Output: ['23-08-1994', '08/23/1994', '1994.08.23']


import pandas as pd
import polyline
import numpy as np

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth 
    specified in decimal degrees using the Haversine formula.
    
    Args:
        coord1 (tuple): The first coordinate (latitude, longitude).
        coord2 (tuple): The second coordinate (latitude, longitude).

    Returns:
        float: Distance between the two coordinates in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in meters
    r = 6371000
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string to a list of coordinates
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate the distance for each point
    distances = [0]  # The first point has a distance of 0
    for i in range(1, len(coordinates)):
        distance = haversine(coordinates[i - 1], coordinates[i])
        distances.append(distance)
    
    # Add the distance column to the DataFrame
    df['distance'] = distances
    
    return df

# Example test case
polyline_str = "u~_vF`n~uC?@??_C??o@??J???J@??XyD??P@??^H???N???D??o@??E???D??"
df = polyline_to_dataframe(polyline_str)
print(df)




from typing import List

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all elements in its row and column (excluding itself).
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Create a new matrix to hold the final values
    final_matrix = [[0] * n for _ in range(n)]
    
    # Calculate the sum of rows and columns in the rotated matrix
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  # Sum of the current row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the current column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude the element itself

    return final_matrix

# Example test case
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)
# Output: [[22, 19, 16], [23, 20, 17], [24, 21, 18]]


import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use the given dataset to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the dataset.

    Returns:
        pd.Series: A boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    # Ensure the timestamp columns are datetime objects
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Create a multi-index based on (id, id_2)
    df.set_index(['id', 'id_2'], inplace=True)

    # Group by (id, id_2)
    grouped = df.groupby(level=[0, 1])
    
    # Function to check completeness for each group
    def check_completeness(group):
        # Check if the start and end timestamps cover all days of the week and a full 24 hours
        all_days = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        days_covered = set(group['start_timestamp'].dt.day_name())
        time_covered = group['end_timestamp'].max() - group['start_timestamp'].min()

        # Check if all days are covered and the time covers 24 hours
        return (days_covered == all_days) and (time_covered >= pd.Timedelta(days=1))

    # Apply the check completeness function to each group and return a boolean series
    result = grouped.apply(check_completeness)
    
    return result

# Example usage
# df = pd.read_csv('dataset-1.csv')
# result_series = time_check(df)
# print(result_series)
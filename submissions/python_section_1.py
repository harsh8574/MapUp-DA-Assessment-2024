from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []  
    i = 0  

    while i < len(lst):
        group = []
        for j in range(n):
            if i + j < len(lst):  
                group.append(lst[i + j])

        reversed_group = []
        for k in range(len(group) - 1, -1, -1):
            reversed_group.append(group[k])

        result.extend(reversed_group)

        i += n

    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    
    for word in lst:
        length = len(word)  
        
        if length not in result:
            result[length] = []
        
        result[length].append(word)
    
    return dict(sorted(result.items()))


def flatten_dict(nested_dict: Dict,parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flat_dict = {}
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, new_key, sep))
        
        elif isinstance(value, list):
            for i, item in enumerate(value):
                indexed_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    flat_dict.update(flatten_dict(item, indexed_key, sep))
                else:
                    flat_dict[indexed_key] = item
        

        else:
            flat_dict[new_key] = value
    
    return flat_dict

from itertools import permutations
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    all_perms = permutations(nums)
    unique_perms = set(all_perms)
    
    return [list(perm) for perm in unique_perms]   
    pass

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

    pattern = r'\b(\d{2}-\d{2}-\d{4})\b|\b(\d{2}/\d{2}/\d{4})\b|\b(\d{4}\.\d{2}\.\d{2})\b'
    
    matches = re.findall(pattern, text)
    
    dates = [match for group in matches for match in group if match]
    
    return dates
    pass

import polyline
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two latitude-longitude points in meters.
    """
    R = 6371000  # Radius of the Earth in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, 
    and distance between consecutive points.

    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=["latitude", "longitude"])

    distances = [0.0]

    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1][["latitude", "longitude"]]
        lat2, lon2 = df.iloc[i][["latitude", "longitude"]]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        distances.append(dist)

    df["distance"] = distances

    return df




def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) 
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  
    
    return final_matrix
    return []


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()

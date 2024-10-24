from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    i = 0
    
    while i < len(lst):
        temp = []
        for j in range(i, min(i + n, len(lst))):
            temp.append(lst[j])
        
        for j in range(len(temp)):
            result.append(temp[len(temp) - 1 - j])
        
        i += n
    
    return result
# --------------------------------------------------------------------------------------------

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}

    for word in lst:
        length = len(word)
        
        if length not in length_dict:
            length_dict[length] = []
        
        length_dict[length].append(word)
    
    return dict(sorted(length_dict.items()))
# ------------------------------------------------------------------------------------------------
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    items.append((list_key, item))
        else:
            items.append((new_key, v))
    
    return dict(items)
    return dict
# --------------------------------------------------------------------------------------------------------
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    if len(path) == len(nums):
            result.append(path[:])  
            return
        
    for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            
            used[i] = True
            path.append(nums[i])
            
            backtrack(path, used)
            
            path.pop()
            used[i] = False
    
    nums.sort()
    result = []
    used = [False] * len(nums) 
    
    backtrack([], used)
    
    return result
    pass
# -----------------------------------------------------------------------------------------------------------------

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
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b', 
        r'\b\d{4}\.\d{2}\.\d{2}\b'
    ]
    
    combined_pattern = '|'.join(date_patterns)
    
    matches = re.findall(combined_pattern, text)
    
    return matches
    pass
# ----------------------------------------------------------------------------------------
import polyline
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 

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
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    distances = [0]
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1]
        lat2, lon2 = df.iloc[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)

    df['distance'] = distances
    return df
    return pd.Dataframe()

# ----------------------------------------------------------------------------------------
from typing import List
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
            original_sum = i + j
            final_matrix[i][j] = rotated_matrix[i][j] * original_sum
            
    return final_matrix
    return []
# ----------------------------------------------------------------------------------------
import pandas as pd
def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    

    return pd.Series()
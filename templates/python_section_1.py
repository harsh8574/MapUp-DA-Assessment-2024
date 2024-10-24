from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    If fewer than n elements remain, reverse all of them.
    """
    result = []
    i = 0

    while i < len(lst):
        group = []
        for j in range(i, min(i + n, len(lst))):
            group.append(lst[j])
        reversed_group = []
        for k in range(len(group) - 1, -1, -1):
            reversed_group.append(group[k])
        result.extend(reversed_group)
        i += n

    return result
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))           # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]





from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary sorted by length.
    """
    length_dict = {}
    for s in lst:
        length = len(s)
        if length not in length_dict:
            length_dict[length] = []  
        length_dict[length].append(s)
    return dict(sorted(length_dict.items()))
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))





from typing import Any, Dict

def flatten_dict(nested_dict: Dict[Any, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary to flatten
    :param parent_key: The prefix for the current level (used during recursion)
    :param sep: Separator between keys (default is '.')
    :return: A flattened dictionary
    """
    flattened = {}

    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict): 
            flattened.update(flatten_dict(value, new_key, sep))
        elif isinstance(value, list): 
            for index, item in enumerate(value):
                flattened.update(flatten_dict({f"{key}[{index}]": item}, parent_key, sep))
        else:  
            flattened[new_key] = value

    return flattened
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)





from typing import List
from itertools import permutations

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.

    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    unique_perms = set(permutations(nums))
    return [list(perm) for perm in unique_perms]
print(unique_permutations([1, 1, 2]))






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
    date_pattern = re.compile(
        r'\b(\d{2}-\d{2}-\d{4})\b|' 
        r'\b(\d{2}/\d{2}/\d{4})\b|' 
        r'\b(\d{4}\.\d{2}\.\d{2})\b' 
    )
    matches = date_pattern.findall(text)
    return [match for group in matches for match in group if match]
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))





import pandas as pd
import polyline
from math import radians, sin, cos, sqrt, atan2
from typing import List, Tuple

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points on the Earth (in meters).
    """
    R = 6371000  

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c  

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude,
    and distance between consecutive points.

    Args:
        polyline_str (str): The encoded polyline string.
    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates: List[Tuple[float, float]] = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=["latitude", "longitude"])
    distances = [0.0] 

    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1][["latitude", "longitude"]]
        lat2, lon2 = df.iloc[i][["latitude", "longitude"]]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    df["distance"] = distances

    return df
polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
df = polyline_to_dataframe(polyline_str)
print(df)





from typing import List

def rotate_matrix_90_degrees(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotates the given matrix by 90 degrees clockwise.
    """
    n = len(matrix)
    rotated = [[matrix[n - 1 - j][i] for j in range(n)] for i in range(n)]
    return rotated
def sum_row_and_column(matrix: List[List[int]], row: int, col: int) -> int:
    """
    Calculates the sum of elements in the same row and column, excluding the element at (row, col).
    """
    row_sum = sum(matrix[row]) - matrix[row][col]  
    col_sum = sum(matrix[i][col] for i in range(len(matrix))) - matrix[row][col]  
    return row_sum + col_sum

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the matrix by 90 degrees clockwise, then replace each element with the
    sum of elements in the same row and column, excluding itself.
    
    Args:
        matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
        List[List[int]]: A new 2D list representing the transformed matrix.
    """
    rotated_matrix = rotate_matrix_90_degrees(matrix)
    n = len(rotated_matrix)
    transformed_matrix = [[0] * n for _ in range(n)] 

    for i in range(n):
        for j in range(n):
            transformed_matrix[i][j] = sum_row_and_column(rotated_matrix, i, j)

    return transformed_matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

result = rotate_and_multiply_matrix(matrix)
print(result)





import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verifies if each (id, id_2) pair covers a full 24-hour period and all 7 days of the week.
    
    Args:
        df (pd.DataFrame): DataFrame containing id, id_2, and timestamp (startDay, startTime, endDay, endTime).
    
    Returns:
        pd.Series: Boolean series with a MultiIndex (id, id_2) indicating whether the timestamps are complete.
    """
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    df['start_weekday'] = df['start_datetime'].dt.dayofweek
    df['end_weekday'] = df['end_datetime'].dt.dayofweek
    expected_days = set(range(7)) 
    expected_hours = pd.date_range('00:00', '23:59', freq='1min').time
    def validate_group(group):
        days_covered = set(group['start_weekday']).union(group['end_weekday'])
        if days_covered != expected_days:
            return False
        for day in expected_days:
            day_entries = group[(group['start_weekday'] <= day) & (group['end_weekday'] >= day)]
            time_covered = pd.Series(False, index=expected_hours)

            for _, row in day_entries.iterrows():
                start_time = row['start_datetime'].time()
                end_time = row['end_datetime'].time()
                time_range = pd.date_range(start_time, end_time, freq='1min').time
                time_covered.loc[time_range] = True

            if not time_covered.all():
                return False

        return True
    result = df.groupby(['id', 'id_2']).apply(validate_group)
    return pd.Series(result, index=result.index)
print(time_check)






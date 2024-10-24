from typing import Dict, List
import pandas as pd
import re
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
import polyline
from datetime import datetime, timedelta

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        # Get the current group
        group = lst[i:min(i + n, len(lst))]
        # Reverse the group manually
        left, right = 0, len(group) - 1
        while left < right:
            group[left], group[right] = group[right], group[left]
            left += 1
            right -= 1
        result.extend(group)
    return result

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = defaultdict(list)
    for s in lst:
        result[len(s)].append(s)
    return dict(sorted(result.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.', prefix: str = '') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    """
    result = {}
    for key, value in nested_dict.items():
        new_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_dict(value, sep, f"{new_key}{sep}"))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    result.update(flatten_dict(item, sep, f"{new_key}[{i}]{sep}"))
                else:
                    result[f"{new_key}[{i}]"] = item
        else:
            result[new_key] = value
    return result

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    """
    def backtrack(counter, perm):
        if len(perm) == len(nums):
            result.add(tuple(perm))
            return
            
        for num in counter:
            if counter[num] > 0:
                counter[num] -= 1
                perm.append(num)
                backtrack(counter, perm)
                perm.pop()
                counter[num] += 1
    
    counter = defaultdict(int)
    for num in nums:
        counter[num] += 1
    result = set()
    backtrack(counter, [])
    return [list(p) for p in result]

def find_all_dates(text: str) -> List[str]:
    """
    Finds all valid dates in the given text.
    """
    patterns = [
        r'\d{2}-\d{2}-\d{4}',  # dd-mm-yyyy
        r'\d{2}/\d{2}/\d{4}',  # mm/dd/yyyy
        r'\d{4}\.\d{2}\.\d{2}'  # yyyy.mm.dd
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return dates

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance between two points on Earth."""
    R = 6371000  # Earth's radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with distances.
    """
    # Decode polyline
    coords = polyline.decode(polyline_str)
    
    # Create DataFrame
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    
    # Calculate distances
    distances = [0]  # First point has distance 0
    for i in range(1, len(df)):
        dist = haversine_distance(
            df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
            df.iloc[i]['latitude'], df.iloc[i]['longitude']
        )
        distances.append(dist)
    
    df['distance'] = distances
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate matrix 90 degrees clockwise and transform elements.
    
    The transformation process:
    1. First rotate the matrix 90 degrees clockwise
    2. For each element in the rotated matrix, replace it with the sum of all 
       elements in its row and column, excluding itself
    
    Args:
    - matrix (List[List[int]]): Input matrix to be transformed
    
    Returns:
    - List[List[int]]: Transformed matrix
    """
    n = len(matrix)
    
    # Step 1: Rotate matrix 90 degrees clockwise
    rotated = [[matrix[n-1-j][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Calculate result using row and column sums from rotated matrix
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Calculate row sum excluding current element
            row_sum = sum(rotated[i][k] for k in range(n) if k != j)
            
            # Calculate column sum excluding current element
            col_sum = sum(rotated[k][j] for k in range(n) if k != i)
            
            # Sum of row and column sums
            result[i][j] = row_sum + col_sum
    
    return result

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of time data for each (id, id_2) pair.
    
    Args:
        df (pandas.DataFrame): DataFrame with columns id, id_2, startDay, endDay, startTime, endTime
        
    Returns:
        pd.Series: Boolean series with multi-index (id, id_2)
    """
    def check_coverage(group):
        # Map days to numbers
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        
        # Convert days to numbers
        start_days = [day_map[day] for day in group['startDay']]
        end_days = [day_map[day] for day in group['endDay']]
        
        # Check if all days are covered
        days_covered = set()
        for start_day, end_day in zip(start_days, end_days):
            if end_day >= start_day:
                days_covered.update(range(start_day, end_day + 1))
            else:
                # Handle week wraparound
                days_covered.update(range(start_day, 7))
                days_covered.update(range(0, end_day + 1))
        
        # Convert times to minutes for easier comparison
        def time_to_minutes(time_str):
            h, m, s = map(int, time_str.split(':'))
            return h * 60 + m + (s / 60)
        
        start_times = [time_to_minutes(t) for t in group['startTime']]
        end_times = [time_to_minutes(t) for t in group['endTime']]
        
        # Check time coverage
        time_ranges = list(zip(start_times, end_times))
        total_coverage = set()
        for start, end in time_ranges:
            if end >= start:
                total_coverage.update(range(int(start), int(end) + 1))
            else:
                # Handle day wraparound
                total_coverage.update(range(int(start), 24 * 60))
                total_coverage.update(range(0, int(end) + 1))
        
        return len(days_covered) == 7 and len(total_coverage) == 24 * 60
    
    return df.groupby(['id', 'id_2']).apply(check_coverage)
# Databricks notebook source
# MAGIC %md
# MAGIC **Question 1: Reverse List by N Elements**
# MAGIC

# COMMAND ----------

# Question 1: Reverse List by N Elements


def reverse_in_groups(lst, n):
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        end = min(i + n, length)
        temp = []
        
        for j in range(i, end):
            temp.append(lst[j])
        

        for j in range(len(temp) - 1, -1, -1):
            result.append(temp[j])
    
    return result

print(reverse_in_groups([1, 2, 3, 4, 5, 6, 7, 8], 3)) 
print(reverse_in_groups([1, 2, 3, 4, 5], 2))        
print(reverse_in_groups([10, 20, 30, 40, 50, 60, 70], 4))  



# COMMAND ----------

# MAGIC %md
# MAGIC **Question 2: Lists & Dictionaries**
# MAGIC

# COMMAND ----------

#Question 2: Lists & Dictionaries

def group_strings_by_length(strings):
    length_dict = {}

    for s in strings:
        length = len(s)
        if length not in length_dict:
            length_dict[length] = []  
        length_dict[length].append(s)  

    sorted_length_dict = dict(sorted(length_dict.items()))

    return sorted_length_dict

print(group_strings_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_strings_by_length(["one", "two", "three", "four"]))


# COMMAND ----------

# MAGIC %md
# MAGIC **Question 3: Flatten a Nested Dictionary**
# MAGIC

# COMMAND ----------

#Question 3: Flatten a Nested Dictionary

def flatten_dictionary(nested_dict, parent_key='', separator='.'):
    items = {}

    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            items.update(flatten_dictionary(value, new_key, separator))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dictionary(item, f"{new_key}[{index}]", separator))
                else:
                    items[f"{new_key}[{index}]"] = item
        else:
            items[new_key] = value

    return items

nested_input = {
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

flattened_output = flatten_dictionary(nested_input)
print(flattened_output)


# COMMAND ----------

# MAGIC %md
# MAGIC **Question 4: Generate Unique Permutations**
# MAGIC

# COMMAND ----------

#Question 4: Generate Unique Permutations

def unique_permutations(nums):
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:]) 
            return
        
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[start]:
                continue
            
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)  
            nums[start], nums[i] = nums[i], nums[start]  

    nums.sort()  
    result = []
    backtrack(0)
    return result

input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question 5: Find All Dates in a Text**
# MAGIC

# COMMAND ----------

#Question 5: Find All Dates in a Text

import re

def find_all_dates(text):
    date_patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',  
        r'\b(\d{2})/(\d{2})/(\d{4})\b',  
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'  
    ]
    
    combined_pattern = '|'.join(date_patterns)
    
    matches = re.findall(combined_pattern, text)
    
    valid_dates = [''.join(match) for match in matches if any(match)]
    
    return valid_dates

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)


# COMMAND ----------

# MAGIC %md
# MAGIC **Question 6: Decode Polyline, Convert to DataFrame with Distances**
# MAGIC

# COMMAND ----------

#Question 6: Decode Polyline, Convert to DataFrame with Distances

import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371000
    return c * r

def decode_polyline(polyline_str):
    coordinates = []
    index = 0
    lat = 0
    lng = 0
    length = len(polyline_str)

    while index < length:
        b = 0
        shift = 0
        result = 0
        
        while True:
            if index >= length:  
                return coordinates
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            if b < 0x20:
                break
            shift += 5
            
        dlat = (result >> 1) ^ -(result & 1)
        lat += dlat
        
        shift = 0
        result = 0
        
        while True:
            if index >= length: 
                return coordinates
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            if b < 0x20:
                break
            shift += 5
        
        dlng = (result >> 1) ^ -(result & 1)
        lng += dlng
        
        coordinates.append((lat * 1e-5, lng * 1e-5))

    return coordinates

def create_dataframe(coordinates):
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    distances = [0]  
    for i in range(1, len(df)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                         df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)
    
    df['distance'] = distances
    return df

polyline_str = "gfo}EtohhU~@j@w@qCqA"
coordinates = decode_polyline(polyline_str)
df = create_dataframe(coordinates)
print(df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question 7: Matrix Rotation and Transformation**
# MAGIC
# MAGIC

# COMMAND ----------

#Question 7: Matrix Rotation and Transformation

def rotate_and_transform_matrix(matrix):
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

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform_matrix(matrix)
for row in result:
    print(row)


# COMMAND ----------

# MAGIC %md
# MAGIC **Question 8: Time Check**
# MAGIC

# COMMAND ----------

#Question 8: Time Check

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local").appName("TimeCheck").getOrCreate()

file_path = 'dbfs:/FileStore/shared_uploads/shubhrasingh1423@gmail.com/dataset_1-1.csv'
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

day_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

day_mapping_expr = F.create_map([F.lit(x) for x in sum(day_mapping.items(), ())])

df_spark = df_spark.withColumn('startDay_num', day_mapping_expr[F.col('startDay')]) \
                   .withColumn('endDay_num', day_mapping_expr[F.col('endDay')])

df_spark = df_spark.withColumn('startTime_full', F.concat(F.col('startDay_num'), F.lit(' '), F.col('startTime'))) \
                   .withColumn('endTime_full', F.concat(F.col('endDay_num'), F.lit(' '), F.col('endTime')))

df_spark = df_spark.withColumn('startTime_full', F.to_timestamp('startTime_full', 'd HH:mm:ss')) \
                   .withColumn('endTime_full', F.to_timestamp('endTime_full', 'd HH:mm:ss'))

def check_timestamps(df):
    window_spec = Window.partitionBy("id", "id_2")

    df = df.withColumn('total_duration', F.sum(F.unix_timestamp('endTime_full') - F.unix_timestamp('startTime_full')).over(window_spec))

    full_week_seconds = 7 * 24 * 60 * 60

    df = df.withColumn('is_incomplete', F.col('total_duration') < full_week_seconds)

    return df.select('id', 'id_2', 'is_incomplete').dropDuplicates()

result_df = check_timestamps(df_spark)

result_df.display()


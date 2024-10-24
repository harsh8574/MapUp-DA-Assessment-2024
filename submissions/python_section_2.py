# Databricks notebook source
# MAGIC %md
# MAGIC **Question 9: Distance Matrix Calculation**
# MAGIC

# COMMAND ----------

pip install graphframes


# COMMAND ----------

# Question 9

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType
from graphframes import GraphFrame

spark = SparkSession.builder \
    .appName("Distance Matrix Calculation") \
    .getOrCreate()

file_path = 'dbfs:/FileStore/shared_uploads/shubhrasingh1423@gmail.com/dataset_2.csv'
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

df_spark = df_spark.withColumnRenamed("id_start", "src") \
                   .withColumnRenamed("id_end", "dst") \
                   .withColumnRenamed("distance", "dist") \
                   .withColumn("dist", F.col("dist").cast(LongType())) \
                   .withColumn("src", F.col("src").cast(LongType())) \
                   .withColumn("dst", F.col("dst").cast(LongType()))

vertices = df_spark.select(F.col("src").alias("id")).union(df_spark.select(F.col("dst").alias("id"))).distinct()

graph = GraphFrame(vertices, df_spark)

shortest_paths = graph.shortestPaths(landmarks=vertices.select("id").rdd.flatMap(lambda x: x).collect())

exploded_paths = shortest_paths.select("id", F.explode("distances").alias("dst", "distance")) \
                               .withColumn("dst", F.col("dst").cast(LongType())) \
                               .withColumn("distance", F.col("distance").cast(LongType()))

distance_matrix_spark = exploded_paths.groupBy("id").pivot("dst").agg(F.first("distance")).fillna(float('inf'))

distance_matrix_spark.display()


# COMMAND ----------

# MAGIC %md
# MAGIC **Question 10: Unroll Distance Matrix**
# MAGIC

# COMMAND ----------

# Question 10

from pyspark.sql import functions as F

def unroll_distance_matrix(distance_matrix):
    toll_ids = distance_matrix.columns
    
    map_expr = []
    for c in toll_ids[1:]:  
        map_expr.append(F.lit(c))  
        map_expr.append(F.col(c))  
    
    unrolled_df = distance_matrix.select(
        F.col(toll_ids[0]).alias("id_start"),  
        F.explode(F.create_map(*map_expr)).alias("id_end", "distance")  
    )
    
    unrolled_df = unrolled_df.filter(F.col("id_start") != F.col("id_end"))
    
    return unrolled_df

unrolled_df_spark = unroll_distance_matrix(distance_matrix_spark)

unrolled_df_spark.display()


# COMMAND ----------

# MAGIC %md
# MAGIC **Question 11: Finding IDs within Percentage Threshold**

# COMMAND ----------

# Question 11

from pyspark.sql import functions as F

def find_ids_within_ten_percentage_threshold(unrolled_df, reference_value):
    reference_avg = unrolled_df.filter(F.col("id_start") == reference_value).agg(F.avg("distance")).collect()[0][0]
    
    lower_threshold = reference_avg * 0.9  
    upper_threshold = reference_avg * 1.1  
    
    filtered_df = unrolled_df.groupBy("id_start").agg(F.avg("distance").alias("avg_distance"))
    
    result_df = filtered_df.filter(
        (F.col("avg_distance") >= lower_threshold) & (F.col("avg_distance") <= upper_threshold)
    )
    
    result_ids = result_df.select("id_start").orderBy("id_start").rdd.flatMap(lambda x: x).collect()
    
    return result_ids

reference_value = 1001400
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df_spark, reference_value)

print(result_ids)


# COMMAND ----------

# MAGIC %md
# MAGIC **Question 12: Calculate Toll Rate**

# COMMAND ----------

# Question 12

from pyspark.sql import functions as F

def calculate_toll_rate(unrolled_df):
    unrolled_df = unrolled_df.withColumn("moto", F.col("distance") * 0.8)
    unrolled_df = unrolled_df.withColumn("car", F.col("distance") * 1.2)
    unrolled_df = unrolled_df.withColumn("rv", F.col("distance") * 1.5)
    unrolled_df = unrolled_df.withColumn("bus", F.col("distance") * 2.2)
    unrolled_df = unrolled_df.withColumn("truck", F.col("distance") * 3.6)
    
    return unrolled_df

toll_rate_df = calculate_toll_rate(unrolled_df_spark)

toll_rate_df.display()


# COMMAND ----------

# MAGIC %md
# MAGIC **Question 13: Calculate Time-Based Toll Rates**

# COMMAND ----------

# Question 13

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import pandas as pd
from datetime import timedelta

spark = SparkSession.builder \
    .appName("Calculate Time Based Toll Rates") \
    .getOrCreate()

def calculate_time_based_toll_rates(toll_rate_df):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekend_days = ['Saturday', 'Sunday']
    
    weekday_discount = [
        (0.8, '00:00:00', '10:00:00'),
        (1.2, '10:00:00', '18:00:00'),
        (0.8, '18:00:00', '23:59:59')
    ]
    
    weekend_discount = 0.7
    
    time_intervals = []
    for day_index in range(7):
        for hour in range(24):
            for minute in [0, 30]:  
                start_time = f"{hour:02d}:{minute:02d}:00"
                end_hour = (hour + 1) % 24
                end_time = f"{end_hour:02d}:{minute:02d}:00"
                current_day = (pd.Timestamp('2021-01-04') + timedelta(days=day_index)).strftime('%A')
                time_intervals.append((current_day, start_time, end_time))
    
    time_df = spark.createDataFrame(time_intervals, ["start_day", "start_time", "end_time"])

    joined_df = toll_rate_df.crossJoin(time_df)

    def calculate_discount(start_day, start_time):
        if start_day in weekdays:
            for discount, start, end in weekday_discount:
                if start_time >= start and start_time < end:
                    return discount
            return 1.0  
        else:
            return weekend_discount  

    calculate_discount_udf = F.udf(calculate_discount, FloatType())

    result_df = joined_df.withColumn("discount_factor", calculate_discount_udf(
        F.col("start_day"),
        F.col("start_time")
    ))

    result_df = result_df.withColumn("moto", F.col("moto") * F.col("discount_factor"))
    result_df = result_df.withColumn("car", F.col("car") * F.col("discount_factor"))
    result_df = result_df.withColumn("rv", F.col("rv") * F.col("discount_factor"))
    result_df = result_df.withColumn("bus", F.col("bus") * F.col("discount_factor"))
    result_df = result_df.withColumn("truck", F.col("truck") * F.col("discount_factor"))

    result_df = result_df.select(
        "id_start", "id_end",
        "start_day", "start_time", "end_time",
        "moto", "car", "rv", "bus", "truck"
    )
    
    return result_df

sample_data = [(1, 2, 10.0, 12.0, 15.0, 22.0, 36.0),
               (2, 3, 15.5, 18.6, 23.2, 34.1, 55.8)]
schema = StructType([
    StructField("id_start", StringType(), True),
    StructField("id_end", StringType(), True),
    StructField("moto", FloatType(), True),
    StructField("car", FloatType(), True),
    StructField("rv", FloatType(), True),
    StructField("bus", FloatType(), True),
    StructField("truck", FloatType(), True),
])

toll_rates_df = spark.createDataFrame(sample_data, schema)

result_df = calculate_time_based_toll_rates(toll_rates_df)

result_df.display(truncate=False)


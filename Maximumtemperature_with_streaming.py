from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, regexp_extract
from pyspark.sql.types import *

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Rainfall Analysis") \
    .getOrCreate()

# Define the schema for the CSV data
schema = StructType([
    StructField("Month", StringType(), True),
    StructField("Measure", StringType(), True),
    StructField("Year", IntegerType(), True),
    StructField("Meteorological Conditions", StringType(), True),
    StructField("Value", StringType(), True)
])

# Read the CSV data with the defined schema
streaming_df = spark.readStream \
    .format("csv") \
    .schema(schema) \
    .option("header", "true") \
    .option("delimiter", ";") \
    .load("path")

# Apply event time watermarking on the 'Year' column with a threshold of 1 minute
streaming_df_with_watermark = streaming_df \
    .withColumn("timestamp", col("Year").cast("timestamp")) \
    .withWatermark("timestamp", "1 minute")

# Extract temperature from the "Value" column
temperature_df = streaming_df_with_watermark \
    .filter(streaming_df_with_watermark["Measure"].contains("°C")) \
    .withColumn("Temperature", regexp_extract(col("Value"), r'(\d+(\.\d+)?)', 1).cast(FloatType()))

# Calculate maximum temperature for each month
max_temp_month = temperature_df.groupBy("Month").agg(max("Temperature").alias("MaxTemperature"))

# Sort the result in descending order of MaxTemperature and select top 3 rows
top_3_max_temp_month = max_temp_month.orderBy(col("MaxTemperature").desc()).limit(3)

# Define the streaming query to write the result to the console
query = top_3_max_temp_month.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# Wait for the streaming query to terminate
query.awaitTermination()

# Stop the SparkSession
spark.stop()

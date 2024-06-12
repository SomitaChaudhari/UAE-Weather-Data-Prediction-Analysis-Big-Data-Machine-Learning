from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Create a SparkSession
spark = SparkSession.builder \
    .appName("WeatherDataAnalysis") \
    .getOrCreate()

# Load the dataset
file_path = 'path'
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Define feature columns
feature_columns = ['Maximum Temperature', 'Mean Rainfall by Month and Year', 'Mean Temperature', 'Mean of Relative Humidity']

# VectorAssembler to combine features into a single vector column
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
data = assembler.transform(data)

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=123)

# Create RandomForestRegressor models
rf_humidity = RandomForestRegressor(featuresCol='features', labelCol='Mean of Relative Humidity', seed=42)
rf_rainfall = RandomForestRegressor(featuresCol='features', labelCol='Mean Rainfall by Month and Year', seed=42)
rf_temperature = RandomForestRegressor(featuresCol='features', labelCol='Maximum Temperature', seed=42)

# Train the models
rf_humidity_model = rf_humidity.fit(train_data)
rf_rainfall_model = rf_rainfall.fit(train_data)
rf_temperature_model = rf_temperature.fit(train_data)

# Make predictions on test data
predictions_humidity = rf_humidity_model.transform(test_data)
predictions_rainfall = rf_rainfall_model.transform(test_data)
predictions_temperature = rf_temperature_model.transform(test_data)

# Evaluate the models
evaluator_humidity = RegressionEvaluator(labelCol='Mean of Relative Humidity', predictionCol='prediction', metricName='rmse')
evaluator_rainfall = RegressionEvaluator(labelCol='Mean Rainfall by Month and Year', predictionCol='prediction', metricName='rmse')
evaluator_temperature = RegressionEvaluator(labelCol='Maximum Temperature', predictionCol='prediction', metricName='rmse')

rmse_humidity = evaluator_humidity.evaluate(predictions_humidity)
rmse_rainfall = evaluator_rainfall.evaluate(predictions_rainfall)
rmse_temperature = evaluator_temperature.evaluate(predictions_temperature)

print("Root Mean Squared Error (RMSE) for Mean of Relative Humidity:", rmse_humidity)
print("Root Mean Squared Error (RMSE) for Mean Rainfall by Month and Year:", rmse_rainfall)
print("Root Mean Squared Error (RMSE) for Maximum Temperature:", rmse_temperature)

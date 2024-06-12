from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
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

# Create a DecisionTreeRegressor for Mean of Relative Humidity
dt_humidity = DecisionTreeRegressor(featuresCol='features', labelCol='Mean of Relative Humidity')

# Train the DecisionTreeRegressor for Mean of Relative Humidity
dt_humidity_model = dt_humidity.fit(train_data)

# Make predictions on test data for Mean of Relative Humidity
predictions_humidity = dt_humidity_model.transform(test_data)

# Evaluate the model for Mean of Relative Humidity
evaluator_humidity = RegressionEvaluator(labelCol='Mean of Relative Humidity', predictionCol='prediction', metricName='rmse')
rmse_humidity = evaluator_humidity.evaluate(predictions_humidity)
print("Root Mean Squared Error (RMSE) for Mean of Relative Humidity:", rmse_humidity)

# Create a DecisionTreeRegressor for Mean Rainfall by Month and Year
dt_rainfall = DecisionTreeRegressor(featuresCol='features', labelCol='Mean Rainfall by Month and Year')

# Train the DecisionTreeRegressor for Mean Rainfall by Month and Year
dt_rainfall_model = dt_rainfall.fit(train_data)

# Make predictions on test data for Mean Rainfall by Month and Year
predictions_rainfall = dt_rainfall_model.transform(test_data)

# Evaluate the model for Mean Rainfall by Month and Year
evaluator_rainfall = RegressionEvaluator(labelCol='Mean Rainfall by Month and Year', predictionCol='prediction', metricName='rmse')
rmse_rainfall = evaluator_rainfall.evaluate(predictions_rainfall)
print("Root Mean Squared Error (RMSE) for Mean Rainfall by Month and Year:", rmse_rainfall)

# Create a DecisionTreeRegressor for Maximum Temperature
dt_temperature = DecisionTreeRegressor(featuresCol='features', labelCol='Maximum Temperature')

# Train the DecisionTreeRegressor for Maximum Temperature
dt_temperature_model = dt_temperature.fit(train_data)

# Make predictions on test data for Maximum Temperature
predictions_temperature = dt_temperature_model.transform(test_data)

# Evaluate the model for Maximum Temperature
evaluator_temperature = RegressionEvaluator(labelCol='Maximum Temperature', predictionCol='prediction', metricName='rmse')
rmse_temperature = evaluator_temperature.evaluate(predictions_temperature)
print("Root Mean Squared Error (RMSE) for Maximum Temperature:", rmse_temperature)





















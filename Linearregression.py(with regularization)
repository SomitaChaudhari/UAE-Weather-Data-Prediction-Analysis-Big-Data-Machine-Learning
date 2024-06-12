from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Climate Analysis") \
    .getOrCreate()

# Read CSV file into Spark DataFrame
df = spark.read.csv("path to csv file", sep=',', header=True)

# Print DataFrame schema
df.printSchema()

# Preprocess the data
# Convert string columns to float
df = df.withColumn("Maximum Temperature", df["Maximum Temperature"].cast("float"))
df = df.withColumn("Mean Rainfall by Month and Year", df["Mean Rainfall by Month and Year"].cast("float"))
df = df.withColumn("Mean Temperature", df["Mean Temperature"].cast("float"))
df = df.withColumn("Mean of Relative Humidity", df["Mean of Relative Humidity"].cast("float"))
df = df.withColumn("Minimum Temperature", df["Minimum Temperature"].cast("float"))

# Convert "month_year" string column to DateType
df = df.withColumn("month_year", to_date(df["month_year"], "yyyy-MM-dd"))

# Preprocess the data
# Drop rows with missing values in any of the required columns
required_columns = ["month_year", "Maximum Temperature", "Mean Rainfall by Month and Year", "Mean Temperature", "Mean of Relative Humidity", "Minimum Temperature"]
df = df.dropna(subset=required_columns)

# Selecting relevant columns for training
required_columns = ["month_year", "Mean Rainfall by Month and Year", "Mean Temperature", "Mean of Relative Humidity", "Minimum Temperature", "Maximum Temperature"]
df = df.select(required_columns)

# Drop rows with NULL values in any of the feature columns
df = df.dropna()

# List of feature column names
feature_cols = ["Mean Rainfall by Month and Year", "Mean Temperature", "Mean of Relative Humidity", "Minimum Temperature"]

# Assemble the features into a single vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Transform the data
df = assembler.transform(df)

# Split the data into training and test sets
(trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)

# Create a LinearRegression model with regularization
lr = LinearRegression(labelCol="Maximum Temperature", featuresCol="features", elasticNetParam=0.5, regParam=0.1)

# Train the model
lr_model = lr.fit(trainingData)

# Make predictions on the test data
predictions = lr_model.transform(testData)

# Print a sample of predictions
predictions.select("month_year", "Maximum Temperature", "prediction").show()

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Maximum Temperature", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = {:.2f}".format(rmse))

# Stop the SparkSession
spark.stop()

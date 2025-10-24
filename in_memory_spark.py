from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

spark = SparkSession.builder.appName("InMemoryProcessing").getOrCreate()

# Load data into memory
df = spark.read.csv("sample_data.csv", header=True, inferSchema=True).cache()

# Perform in-memory analytics
df.select(avg(col("salary"))).show()

spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()

# Load data
df = spark.read.csv("sample_data.csv", header=True, inferSchema=True)

# Handle missing values
df = df.fillna({'age': 0, 'salary': 0})

# Remove duplicates
df = df.dropDuplicates()

# Convert data types
df = df.withColumn("age", col("age").cast("integer"))

# Normalize salary
assembler = VectorAssembler(inputCols=["salary"], outputCol="salary_vec")
df_vec = assembler.transform(df)
scaler = MinMaxScaler(inputCol="salary_vec", outputCol="salary_scaled")
df_scaled = scaler.fit(df_vec).transform(df_vec)

# Feature engineering
df_scaled = df_scaled.withColumn("age_salary_ratio", col("age") / (col("salary") + 1))

df_scaled.show()
df_scaled.write.csv("preprocessed_data.csv", header=True)
spark.stop()

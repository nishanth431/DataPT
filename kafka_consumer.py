from kafka import KafkaConsumer
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

consumer = KafkaConsumer('sensor_topic',
                         bootstrap_servers='localhost:9092',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

spark = SparkSession.builder.appName("RealTimeProcessing").getOrCreate()

for message in consumer:
    data = message.value
    df = spark.createDataFrame([data])
    avg_temp = df.agg({"temperature": "avg"}).collect()[0][0]
    print(f"Current Temperature: {data['temperature']}, Rolling Avg Temp: {avg_temp}")

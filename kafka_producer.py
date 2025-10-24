from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

while True:
    data = {"sensor_id": 1, "temperature": random.randint(20, 40), "humidity": random.randint(30, 80)}
    producer.send('sensor_topic', data)
    print(f"Sent: {data}")
    time.sleep(2)

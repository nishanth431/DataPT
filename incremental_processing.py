from kafka import KafkaConsumer
import json
from sklearn.linear_model import LinearRegression
import numpy as np

consumer = KafkaConsumer('dbserver1.dpt_cdc.transactions',
                         bootstrap_servers='localhost:9092',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

model = LinearRegression()
X, y = [], []

for message in consumer:
    data = message.value['payload']['after']
    X.append([data['feature1'], data['feature2']])
    y.append(data['target'])
    if len(X) > 10:
        model.fit(np.array(X), np.array(y))
        print(f"Updated Model Coefficients: {model.coef_}")

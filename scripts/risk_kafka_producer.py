import json
import time
from kafka import KafkaProducer 
from pyspark.sql import SparkSession 

# Kafka configuration
KAFKA_BROKER = "localhost:9092"
TOPIC = "risk_data"

# Initialize Spark session with memory optimizations
spark = SparkSession.builder \
    .appName("KafkaProducer") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.memory.fraction", "0.5") \
    .config("spark.memory.storageFraction", "0.4") \
    .config("spark.driver.memory", "512m") \
    .config("spark.executor.memory", "512m") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

# Read dataset from HDFS in partitions 
DATA_PATH = "hdfs://hadoop-master:9000/data/pandemics/annee=2023"
df = spark.read.parquet(DATA_PATH).select("state", "county", "date", "cases", "deaths", "mois", "jour").repartition(2)

# Initialize Kafka Producer
producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER, value_serializer=lambda v: json.dumps(v).encode("utf-8"))
print("Starting Kafka Producer...")

try:
    # Stream data row by row using an iterator to save memory
    for row in df.toLocalIterator():
        message = {
            "state": row["state"],
            "county": row["county"],
            "date": str(row["date"]),
            "cases": int(row["cases"]),
            "deaths": int(row["deaths"]),
            "mois": int(row["mois"]),
            "jour": int(row["jour"])
        }
        producer.send(TOPIC, value=message)
        print(f"Produced: {message}")
        time.sleep(1)  

except KeyboardInterrupt:
    print("\nStopping Producer...")
finally:
    producer.close()
    spark.stop()
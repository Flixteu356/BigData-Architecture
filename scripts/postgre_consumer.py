from pyspark.sql import SparkSession 
from pyspark.sql.functions import from_json, col, to_json, struct, udf 
from pyspark.sql.types import StructType, StructField, StringType, IntegerType 
from pyspark.ml.classification import RandomForestClassificationModel 
from pyspark.ml.feature import VectorAssembler, Bucketizer 
from pyspark.sql.functions import to_date 
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Spark session with memory optimizations
spark = SparkSession.builder \
    .appName("KafkaConsumerStreaming") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.5") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.cores", "1") \
    .config("spark.sql.streaming.checkpointLocation", 
            "hdfs://hadoop-master:9000/checkpoints/pandemic_v2") \
    .getOrCreate()

# Set log level to reduce noise
spark.sparkContext.setLogLevel("WARN")

# Kafka configurations
KAFKA_BROKER = "localhost:9092"
INPUT_TOPIC = "risk_data"

# Define schema for incoming data
schema = StructType([
    StructField("state", StringType(), True),
    StructField("county", StringType(), True),
    StructField("date", StringType(), True),
    StructField("cases", IntegerType(), True),
    StructField("deaths", IntegerType(), True),
    StructField("mois", IntegerType(), True),
    StructField("jour", IntegerType(), True)
])

try:
    logger.info("Starting Structured Streaming Consumer for High-Risk Zones")

    # Read stream from Kafka
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", INPUT_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    logger.info("Connected to Kafka input stream")

    # Deserialize JSON messages
    data_stream = raw_stream.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    def process_batch(batch_df, batch_id):
        logger.info("Preparing data for high-risk zone prediction")

        # Calculate risk score
        batch_df = batch_df.withColumn("risk_score", col("cases") + col("deaths") * 10)

        # Define risk categories using Bucketizer
        bucket_splits = [-float("inf"), 0, 1000, 10000, 100000, float("inf")]
        bucketizer = Bucketizer(splits=bucket_splits, inputCol="risk_score", outputCol="risk_category")
        batch_df = bucketizer.transform(batch_df)

        # Assemble features
        assembler = VectorAssembler(
            inputCols=["mois", "jour", "risk_score"],
            outputCol="features",
            handleInvalid="skip"
        )
        batch_df = assembler.transform(batch_df).dropna()

        # Load and apply the trained classification model
        logger.info("Loading and applying high-risk zone prediction model")
        model_path = "hdfs://hadoop-master:9000/models/GeoRisk_model"
        model = RandomForestClassificationModel.load(model_path)
        predictions = model.transform(batch_df)

        # Select and format the output
        logger.info("Formatting output for PostgreSQL")
        output_df = predictions.select(
            col("state"),
            col("county"),
            col("date"),
            col("risk_category").cast("integer"),
            col("prediction").cast("integer").alias("predicted_risk_category")
        ).select("state", "county", "date", "risk_category", "predicted_risk_category")

        # converting date as a DATE format
        output_df = output_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

        # Write to PostgreSQL
        logger.info("Writing high-risk zone predictions to PostgreSQL")
        logger.info(f"Number of rows in the batch: {output_df.count()}")

        output_df.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://localhost:5432/pandemic_db") \
            .option("dbtable", "risk_predictions") \
            .option("user", "spark_user") \
            .option("password", "1234") \
            .option("driver", "org.postgresql.Driver") \
            .mode("append") \
            .save()

        logger.info("Predictions successfully saved to PostgreSQL!")
        logger.info(f"Number of rows saved: {output_df.count()}")

        # Write stream to Kafka using foreachBatch
    query = data_stream.writeStream \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", "hdfs://hadoop-master:9000/checkpoints/pandemic_v2") \
        .start()

    logger.info("Stream processing started, awaiting termination...")
    query.awaitTermination()

except Exception as e:
    logger.error(f"Error in Kafka consumer: {str(e)}")
    raise
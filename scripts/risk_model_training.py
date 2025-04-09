from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier 
from pyspark.ml.feature import VectorAssembler, Bucketizer

# Initialize Spark session
spark = SparkSession.builder \
    .appName("PandemicPredictionModelTraining") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.5") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

# Load dataset from HDFS
DATA_PATH = "hdfs://hadoop-master:9000/data/pandemics/annee=2023"
df = spark.read.parquet(DATA_PATH).select("state", "county", "date", "cases", "deaths", "mois", "jour")
df = df.dropna()

# a) Calculate risk score
df = df.withColumn("risk_score", df["cases"] + df["deaths"] * 10)

# b) Define risk categories using Bucketizer
bucket_splits = [-float("inf"), 0, 1000, 10000, 100000, float("inf")]
bucketizer = Bucketizer(splits=bucket_splits, inputCol="risk_score", outputCol="risk_category")
df = bucketizer.transform(df)

# c) Assemble features
assembler = VectorAssembler(inputCols=["mois", "jour", "risk_score"], outputCol="features", handleInvalid="skip")
data = assembler.transform(df).dropna()

# Step 2: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 3: Train RandomForest Classifier (Predict risk_category)
rf = RandomForestClassifier(featuresCol="features", labelCol="risk_category", numTrees=20, maxDepth=10)
model = rf.fit(train_data)

# Step 4: Save the model
MODEL_PATH = "hdfs://hadoop-master:9000/models/GeoRisk_model"
model.write().overwrite().save(MODEL_PATH)

print("High-risk zone prediction model saved successfully at:", MODEL_PATH)

# Stop Spark session
spark.stop()
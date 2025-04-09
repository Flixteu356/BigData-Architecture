from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, to_date, count
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  

def create_spark_session():
    """Initialise et retourne une session Spark"""
    return SparkSession.builder \
        .appName("US_Covid_CSV_to_Parquet") \
        .master("local[*]") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.5") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .getOrCreate()

def define_schema():
    """Définit le schéma des données pour US_covid-19-2023.csv"""
    return StructType([
        StructField("date", StringType(), True),
        StructField("county", StringType(), True),
        StructField("state", StringType(), True),
        StructField("cases", IntegerType(), True),
        StructField("deaths", IntegerType(), True)
    ])

def process_data(spark, input_path, output_path):
    """Traite les données et les sauvegarde en format Parquet"""
    try:
        # Lecture du CSV avec schéma prédéfini
        logger.info("Lecture du fichier CSV...")
        df = spark.read.csv(input_path, header=True, schema=define_schema())

        # Identification et suppression des colonnes avec plus de 50% de valeurs NULL
        logger.info("Analyse des colonnes nulles...")
        total_rows = df.count()
        null_counts = df.select([count(col(c)).alias(c) for c in df.columns]).collect()[0]
        cols_to_drop = [c for c in df.columns if null_counts[c] < total_rows * 0.5]

        if cols_to_drop:
            logger.warning(f"Colonnes supprimées (>50% NULL): {cols_to_drop}")
            df = df.drop(*cols_to_drop)

        # Convert 'date' to DateType
        df = df.withColumn("date", to_date("date", "M/d/yyyy"))

        # Ajout des colonnes de partitionnement
        logger.info("Ajout des colonnes de partitionnement...")
        df = df.withColumn("annee", year(col("date")).cast("string")) \
               .withColumn("mois", month(col("date")).cast("string").rjust(2, '0')) \
               .withColumn("jour", dayofmonth(col("date")).cast("string").rjust(2, '0'))

        # Tri des données par année, mois et jour
        df = df.orderBy("annee", "mois", "jour")

        # Écriture en Parquet
        logger.info("Écriture des données en format Parquet...")
        (df.write
         .mode("overwrite") 
         .partitionBy("annee", "mois", "jour")
         .parquet(output_path))

        logger.info(f"Données stockées avec succès dans {output_path}")

        # Statistiques sur les données
        logger.info("Statistiques finales:")
        logger.info(f"Nombre total d'enregistrements: {df.count()}")
        logger.info(f"Nombre de colonnes: {len(df.columns)}")

    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}")
        raise

def main():
    """Fonction principale"""
    input_path = "hdfs://hadoop-master:9000/data/pandemics/us-covid_19-2023.csv"
    output_path = "hdfs://hadoop-master:9000/data/pandemics"

    spark = None
    try:
        spark = create_spark_session()
        process_data(spark, input_path, output_path)
    except Exception as e:
        logger.error(f"Erreur critique: {str(e)}")
        sys.exit(1)
    finally:
        if spark:
            spark.stop()
            logger.info("Session Spark fermée")

if __name__ == "__main__":
    main()
# COVID-19 Risk Prediction System using Big Data Architecture

## Overview
This project implements a comprehensive Big Data architecture to predict pandemic risk levels, focusing on COVID-19 data analysis. The system processes historical COVID-19 data, trains a machine learning model, and provides real-time risk predictions through an interactive dashboard.

## Architecture
The solution leverages several key Big Data technologies:
- **Storage**: Hadoop HDFS (partitioned Parquet files)
- **Processing**: Apache Spark for batch processing and machine learning
- **Streaming**: Kafka and Spark Streaming for real-time data pipelines
- **Database**: PostgreSQL for prediction storage
- **Visualization**: Streamlit dashboard and Grafana monitoring

![Architecture Overview](https://via.placeholder.com/800x400?text=Big+Data+Architecture+Diagram)

## Features
- Conversion of CSV data to optimized Parquet format with time-based partitioning
- Machine learning model (RandomForest) for risk classification
- Real-time data streaming pipeline with Kafka
- Interactive dashboards for risk visualization
- Geographic risk distribution with choropleth maps
- Time-series analysis of pandemic trends

## Components

### Data Processing
- `csv_to_parquet.py`: Converts raw COVID-19 CSV data to partitioned Parquet format in HDFS
- `risk_model_training.py`: Trains and saves a RandomForest classification model for risk prediction

### Real-time Pipeline
- `risk_kafka_producer.py`: Reads data from HDFS and streams to Kafka topic "risk_data"
- `risk_kafka_consumer.py`: Consumes data stream, applies ML model, and stores predictions in PostgreSQL

### Visualization
- `streamlit_dashboard.py`: Interactive web dashboard for data exploration and visualization
- Grafana dashboards for monitoring and analytics

## Dataset
The project uses US COVID-19 data from 2023 with the following structure:
```
date, county, state, cases, deaths
```

Data is processed and augmented with risk scores and categories.

## Getting Started

### Prerequisites
- Apache Hadoop
- Apache Spark
- Apache Kafka
- PostgreSQL
- Python 3.x with required packages (pyspark, kafka-python, streamlit, pandas, plotly)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/covid-risk-prediction.git
cd covid-risk-prediction
```

2. Set up your Hadoop environment:
```bash
hdfs dfs -mkdir -p /data/pandemics
hdfs dfs -mkdir -p /models
hdfs dfs -mkdir -p /checkpoints/pandemic_v2
```

3. Upload your COVID-19 data:
```bash
hdfs dfs -put us-covid_19-2023.csv /data/pandemics/
```

4. Install Python dependencies:
```bash
pip install pyspark kafka-python streamlit pandas plotly psycopg2-binary us
```

5. Set up PostgreSQL database:
```sql
CREATE DATABASE pandemic_db;
CREATE USER spark_user WITH PASSWORD '1234';
GRANT ALL PRIVILEGES ON DATABASE pandemic_db TO spark_user;

\c pandemic_db
CREATE TABLE risk_predictions (
    state TEXT,
    county TEXT,
    date DATE,
    risk_category INTEGER,
    predicted_risk_category INTEGER
);
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO spark_user;
```

### Running the Pipeline

1. Process CSV data to Parquet:
```bash
spark-submit csv_to_parquet.py
```

2. Train the risk prediction model:
```bash
spark-submit risk_model_training.py
```

3. Start Kafka and create necessary topics:
```bash
kafka-topics.sh --create --topic risk_data --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

4. Run the Kafka producer:
```bash
python risk_kafka_producer.py
```

5. Run the Spark Streaming consumer:
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1,org.postgresql:postgresql:42.2.27 risk_kafka_consumer.py
```

6. Launch the Streamlit dashboard:
```bash
streamlit run streamlit_dashboard.py
```

## Results

The final system provides:
- Risk classification with 96% accuracy
- Identification of high-risk pandemic zones
- Geographic visualization of risk distribution
- Time-based analysis of pandemic trends

![Dashboard Preview](https://via.placeholder.com/800x400?text=Dashboard+Preview)

## Future Improvements
- Integration with external data sources (weather, population density)
- Enhanced prediction models with deep learning
- Mobile application for real-time alerts
- Deployment to cloud infrastructure for scalability

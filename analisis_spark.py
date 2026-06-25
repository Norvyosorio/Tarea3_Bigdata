import os
import sys
import logging

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# Configure logging instead of bare print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Expected schema for input validation
EXPECTED_SCHEMA = StructType([
    StructField("age", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("academic_year", IntegerType(), True),
    StructField("academic_performance", DoubleType(), True),
    StructField("burnout_score", DoubleType(), True),
    StructField("risk_level", StringType(), True),
    StructField("mental_health_index", DoubleType(), True),
    StructField("dropout_risk", DoubleType(), True),
    StructField("stress_level", DoubleType(), True),
    StructField("exam_pressure", DoubleType(), True),
    StructField("sleep_hours", DoubleType(), True),
])

# Required columns that must exist in the dataset
REQUIRED_COLUMNS = [
    'age', 'gender', 'academic_year', 'academic_performance',
    'burnout_score', 'risk_level', 'mental_health_index',
    'dropout_risk', 'stress_level', 'exam_pressure', 'sleep_hours'
]


def get_file_path():
    """Retrieve data file path from environment variable or CLI argument."""
    file_path = os.environ.get('SPARK_DATA_PATH')
    if not file_path and len(sys.argv) > 1:
        file_path = sys.argv[1]
    if not file_path:
        file_path = 'hdfs://localhost:9000/Tarea3/student_mental_health_burnout_1M.csv'
        logger.warning(
            "No SPARK_DATA_PATH env var or CLI argument provided. "
            "Using default path: %s", file_path
        )
    return file_path


def validate_dataframe(df, required_columns):
    """Validate that the DataFrame contains all required columns."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Available columns: {df.columns}"
        )


def create_spark_session():
    """Create a Spark session with safe configuration."""
    return SparkSession.builder \
        .appName('AnalisisSaludMentalPro') \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()


def main():
    spark = None
    try:
        # 1. Session setup
        spark = create_spark_session()

        # 2. Load data with configurable path
        file_path = get_file_path()
        logger.info("Loading data from: %s", file_path)

        df_raw = spark.read.format('csv') \
            .option('header', 'true') \
            .option('inferSchema', 'true') \
            .load(file_path)

        # 3. Validate required columns exist
        validate_dataframe(df_raw, REQUIRED_COLUMNS)

        # 4. Data cleaning and transformation
        float_cols = [c for c, t in df_raw.dtypes if t == 'double']
        df_clean = df_raw.select(
            [F.round(F.col(c), 2).alias(c) if c in float_cols else F.col(c) for c in df_raw.columns]
        )

        # 5. High Performers vs High Burnout analysis
        logger.info("Top 10 students with high performance and high burnout:")
        df_clean.filter(
            (F.col('academic_performance') > 80) & (F.col('burnout_score') > 7)
        ) \
            .select('age', 'gender', 'academic_year', 'academic_performance', 'burnout_score', 'risk_level') \
            .orderBy(F.col('burnout_score').desc()) \
            .show(10)

        # 6. Aggregation by academic year and gender
        logger.info("Wellbeing statistics by year and gender:")
        stats_df = df_clean.groupBy('academic_year', 'gender') \
            .agg(
                F.avg('mental_health_index').alias('Promedio_Salud_Mental'),
                F.avg('dropout_risk').alias('Riesgo_Desercion_Promedio'),
                F.count('*').alias('Total_Estudiantes')
            ).orderBy('academic_year', 'gender')

        stats_df.show()

        # 7. Window functions - rank students by stress per academic year
        window_spec = Window.partitionBy("academic_year").orderBy(F.col("stress_level").desc())

        logger.info("Student with highest stress per academic year (Top 1):")
        df_clean.withColumn("rank_estres", F.rank().over(window_spec)) \
            .filter(F.col("rank_estres") == 1) \
            .select('academic_year', 'age', 'gender', 'stress_level', 'exam_pressure') \
            .show()

        # 8. Spark SQL - parameterized view (no user input in SQL)
        df_clean.createOrReplaceTempView("estudiantes")
        query_sql = spark.sql("""
            SELECT risk_level, COUNT(*) as cantidad, ROUND(AVG(sleep_hours), 2) as promedio_sueno
            FROM estudiantes
            GROUP BY risk_level
            ORDER BY cantidad DESC
        """)
        logger.info("Distribution by risk level (via SQL):")
        query_sql.show()

    except FileNotFoundError:
        logger.error("Data file not found. Set SPARK_DATA_PATH or pass the path as argument.")
        sys.exit(1)
    except ValueError as e:
        logger.error("Data validation error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error during analysis: %s", e)
        sys.exit(1)
    finally:
        if spark:
            spark.stop()


if __name__ == '__main__':
    main()

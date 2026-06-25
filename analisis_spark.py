import sys
import logging

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "age", "gender", "academic_year", "academic_performance",
    "burnout_score", "risk_level", "mental_health_index",
    "dropout_risk", "stress_level", "exam_pressure", "sleep_hours",
]


def create_spark_session():
    """Create and return a SparkSession, raising on failure."""
    try:
        spark = SparkSession.builder \
            .appName('AnalisisSaludMentalPro') \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
        logger.info("SparkSession created successfully.")
        return spark
    except Exception as exc:
        logger.error("Failed to create SparkSession: %s", exc)
        raise


def load_data(spark, file_path):
    """Load CSV data from *file_path* and return the raw DataFrame."""
    try:
        df = spark.read.format('csv') \
            .option('header', 'true') \
            .option('inferSchema', 'true') \
            .load(file_path)
        row_count = df.count()
        logger.info("Loaded %d rows from %s", row_count, file_path)
        if row_count == 0:
            raise ValueError(f"The dataset at {file_path} is empty.")
        return df
    except AnalysisException as exc:
        logger.error("Could not read the file at '%s'. "
                      "Check that the path exists and HDFS is reachable: %s",
                      file_path, exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error while loading data: %s", exc)
        raise


def validate_schema(df, required_columns):
    """Raise ValueError if any required column is missing from *df*."""
    existing = set(df.columns)
    missing = [c for c in required_columns if c not in existing]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {sorted(existing)}"
        )
    logger.info("Schema validation passed — all required columns present.")


def clean_data(df):
    """Round all double-type columns to 2 decimals."""
    float_cols = [c for c, t in df.dtypes if t == 'double']
    df_clean = df.select(
        [F.round(F.col(c), 2).alias(c) if c in float_cols
         else F.col(c) for c in df.columns]
    )
    logger.info("Data cleaning complete — rounded %d float columns.",
                len(float_cols))
    return df_clean


def analyze_high_performers(df_clean):
    """High performers with high burnout."""
    logger.info("Running high-performers vs high-burnout analysis...")
    try:
        result = df_clean.filter(
            (F.col('academic_performance') > 80) &
            (F.col('burnout_score') > 7)
        ).select(
            'age', 'gender', 'academic_year',
            'academic_performance', 'burnout_score', 'risk_level',
        ).orderBy(F.col('burnout_score').desc())

        count = result.count()
        if count == 0:
            logger.warning("No students matched the high-performance / "
                           "high-burnout filter.")
        else:
            logger.info("Found %d matching students.", count)
            print("\n>>> Top 10 Estudiantes con Alto Rendimiento y Mayor Burnout:")
            result.show(10)
    except AnalysisException as exc:
        logger.error("High-performers analysis failed (column issue?): %s", exc)
        raise


def analyze_wellbeing_stats(df_clean):
    """Aggregated stats by academic year and gender."""
    logger.info("Running wellbeing statistics by year and gender...")
    try:
        stats_df = df_clean.groupBy('academic_year', 'gender') \
            .agg(
                F.avg('mental_health_index').alias('Promedio_Salud_Mental'),
                F.avg('dropout_risk').alias('Riesgo_Desercion_Promedio'),
                F.count('*').alias('Total_Estudiantes'),
            ).orderBy('academic_year', 'gender')

        print("\n>>> Estadísticas de Bienestar por Año y Género:")
        stats_df.show()
    except AnalysisException as exc:
        logger.error("Wellbeing stats analysis failed (column issue?): %s", exc)
        raise


def analyze_top_stress(df_clean):
    """Rank students by stress level within each academic year."""
    logger.info("Running top-stress window analysis...")
    try:
        window_spec = Window.partitionBy("academic_year") \
            .orderBy(F.col("stress_level").desc())

        result = df_clean.withColumn(
            "rank_estres", F.rank().over(window_spec)
        ).filter(
            F.col("rank_estres") == 1
        ).select(
            'academic_year', 'age', 'gender',
            'stress_level', 'exam_pressure',
        )

        print("\n>>> Estudiante con más estrés por cada Año Académico (Top 1):")
        result.show()
    except AnalysisException as exc:
        logger.error("Top-stress analysis failed (column issue?): %s", exc)
        raise


def analyze_risk_distribution(df_clean):
    """Risk-level distribution via Spark SQL."""
    logger.info("Running risk-level distribution via Spark SQL...")
    try:
        df_clean.createOrReplaceTempView("estudiantes")
        query_sql = df_clean.sparkSession.sql("""
            SELECT risk_level,
                   COUNT(*) as cantidad,
                   ROUND(AVG(sleep_hours), 2) as promedio_sueno
            FROM estudiantes
            GROUP BY risk_level
            ORDER BY cantidad DESC
        """)
        print("\n>>> Distribución por Nivel de Riesgo (vía SQL):")
        query_sql.show()
    except AnalysisException as exc:
        logger.error("SQL risk-distribution query failed: %s", exc)
        raise


def main():
    file_path = 'hdfs://localhost:9000/Tarea3/student_mental_health_burnout_1M.csv'

    spark = create_spark_session()
    try:
        df_raw = load_data(spark, file_path)
        validate_schema(df_raw, REQUIRED_COLUMNS)
        df_clean = clean_data(df_raw)

        analyze_high_performers(df_clean)
        analyze_wellbeing_stats(df_clean)
        analyze_top_stress(df_clean)
        analyze_risk_distribution(df_clean)

        logger.info("All analyses completed successfully.")
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
    finally:
        spark.stop()
        logger.info("SparkSession stopped.")


if __name__ == '__main__':
    main()

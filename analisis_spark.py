from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def create_spark_session(app_name='AnalisisSaludMentalPro', shuffle_partitions='10'):
    """Create and return a configured SparkSession."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", shuffle_partitions) \
        .getOrCreate()


def load_data(spark, file_path):
    """Load a CSV file into a Spark DataFrame with header and schema inference."""
    return spark.read.format('csv') \
        .option('header', 'true') \
        .option('inferSchema', 'true') \
        .load(file_path)


def clean_dataframe(df):
    """Round all double-type columns to 2 decimal places."""
    float_cols = [c for c, t in df.dtypes if t == 'double']
    return df.select(
        [F.round(F.col(c), 2).alias(c) if c in float_cols else F.col(c) for c in df.columns]
    )


def filter_high_performers_high_burnout(df, perf_threshold=80, burnout_threshold=7):
    """Return students with academic_performance > perf_threshold and burnout_score > burnout_threshold,
    ordered by burnout_score descending."""
    return df.filter(
        (F.col('academic_performance') > perf_threshold) &
        (F.col('burnout_score') > burnout_threshold)
    ).select(
        'age', 'gender', 'academic_year', 'academic_performance', 'burnout_score', 'risk_level'
    ).orderBy(F.col('burnout_score').desc())


def compute_wellbeing_stats(df):
    """Aggregate mental health index, dropout risk, and student count by academic_year and gender."""
    return df.groupBy('academic_year', 'gender') \
        .agg(
            F.avg('mental_health_index').alias('Promedio_Salud_Mental'),
            F.avg('dropout_risk').alias('Riesgo_Desercion_Promedio'),
            F.count('*').alias('Total_Estudiantes')
        ).orderBy('academic_year', 'gender')


def rank_stress_by_year(df):
    """Rank students by stress_level within each academic_year using a window function.
    Returns only students ranked #1 (highest stress)."""
    window_spec = Window.partitionBy("academic_year").orderBy(F.col("stress_level").desc())
    return df.withColumn("rank_estres", F.rank().over(window_spec)) \
        .filter(F.col("rank_estres") == 1) \
        .select('academic_year', 'age', 'gender', 'stress_level', 'exam_pressure')


def risk_level_distribution(spark, df):
    """Use Spark SQL to compute count and average sleep hours per risk_level."""
    df.createOrReplaceTempView("estudiantes")
    return spark.sql("""
        SELECT risk_level, COUNT(*) as cantidad, ROUND(AVG(sleep_hours), 2) as promedio_sueno
        FROM estudiantes
        GROUP BY risk_level
        ORDER BY cantidad DESC
    """)


def main():
    # 1. Configuracion de Sesion Optimizada
    spark = create_spark_session()

    # 2. Carga de Datos con Inferencia de Esquema
    file_path = 'hdfs://localhost:9000/Tarea3/student_mental_health_burnout_1M.csv'
    df_raw = load_data(spark, file_path)

    # 3. Limpieza y Transformacion (Casteo y Redondeo)
    df_clean = clean_dataframe(df_raw)

    # 4. Analisis de "High Performers vs High Burnout"
    print("\n>>> Top 10 Estudiantes con Alto Rendimiento y Mayor Burnout:")
    filter_high_performers_high_burnout(df_clean).show(10)

    # 5. Agregacion Avanzada por Ano Academico y Genero
    print("\n>>> Estadisticas de Bienestar por Ano y Genero:")
    compute_wellbeing_stats(df_clean).show()

    # 6. Uso de Window Functions
    print("\n>>> Estudiante con mas estres por cada Ano Academico (Top 1):")
    rank_stress_by_year(df_clean).show()

    # 7. Spark SQL
    print("\n>>> Distribucion por Nivel de Riesgo (via SQL):")
    risk_level_distribution(spark, df_clean).show()


if __name__ == '__main__':
    main()

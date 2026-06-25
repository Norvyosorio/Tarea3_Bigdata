from pyspark.sql import Window
from pyspark.sql import functions as F

from spark_utils import (
    create_spark_session,
    filter_and_rank,
    load_csv,
    round_float_columns,
    show_analysis,
)

# 1. Configuracion de Sesion Optimizada
spark = create_spark_session('AnalisisSaludMentalPro')

# 2. Carga de Datos con Inferencia de Esquema
file_path = 'hdfs://localhost:9000/Tarea3/student_mental_health_burnout_1M.csv'
df_raw = load_csv(spark, file_path)

# 3. Limpieza y Transformacion (Casteo y Redondeo)
df_clean = round_float_columns(df_raw)

# 4. Analisis de "High Performers vs High Burnout" (El Insight de Oro)
high_perf_burnout = filter_and_rank(
    df_clean,
    filter_expr=(F.col('academic_performance') > 80) & (F.col('burnout_score') > 7),
    select_cols=['age', 'gender', 'academic_year', 'academic_performance', 'burnout_score', 'risk_level'],
    order_col='burnout_score',
)
show_analysis("Top 10 Estudiantes con Alto Rendimiento y Mayor Burnout:", high_perf_burnout, n=10)

# 5. Agregacion Avanzada por Ano Academico y Genero
stats_df = df_clean.groupBy('academic_year', 'gender') \
    .agg(
        F.avg('mental_health_index').alias('Promedio_Salud_Mental'),
        F.avg('dropout_risk').alias('Riesgo_Desercion_Promedio'),
        F.count('*').alias('Total_Estudiantes')
    ).orderBy('academic_year', 'gender')

show_analysis("Estadisticas de Bienestar por Ano y Genero:", stats_df)

# 6. Uso de Window Functions (Funciones de Ventana)
window_spec = Window.partitionBy("academic_year").orderBy(F.col("stress_level").desc())

ranked_stress = df_clean.withColumn("rank_estres", F.rank().over(window_spec)) \
    .filter(F.col("rank_estres") == 1) \
    .select('academic_year', 'age', 'gender', 'stress_level', 'exam_pressure')

show_analysis("Estudiante con mas estres por cada Ano Academico (Top 1):", ranked_stress)

# 7. Spark SQL (Para demostrar versatilidad)
df_clean.createOrReplaceTempView("estudiantes")
query_sql = spark.sql("""
    SELECT risk_level, COUNT(*) as cantidad, ROUND(AVG(sleep_hours), 2) as promedio_sueno
    FROM estudiantes
    GROUP BY risk_level
    ORDER BY cantidad DESC
""")
show_analysis("Distribucion por Nivel de Riesgo (via SQL):", query_sql)

# Finalizar sesion
# spark.stop()

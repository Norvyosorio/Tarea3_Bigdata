from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# 1. Configuración de Sesión Optimizada
spark = SparkSession.builder \
    .appName('AnalisisSaludMentalPro') \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

# 2. Carga de Datos con Inferencia de Esquema
file_path = 'hdfs://localhost:9000/Tarea3/student_mental_health_burnout_1M.csv'
df_raw = spark.read.format('csv') \
    .option('header', 'true') \
    .option('inferSchema', 'true') \
    .load(file_path)

# 3. Limpieza y Transformación (Casteo y Redondeo)
# Redondeamos todos los floats a 2 decimales para que el .show() no sea un caos
float_cols = [c for c, t in df_raw.dtypes if t == 'double']
df_clean = df_raw.select([F.round(F.col(c), 2).alias(c) if c in float_cols else F.col(c) for c in df_raw.columns])

# 4. Análisis de "High Performers vs High Burnout" (El Insight de Oro)
# Queremos ver quiénes tienen notas altas pero están a punto de explotar
print("\n>>> Top 10 Estudiantes con Alto Rendimiento y Mayor Burnout:")
df_clean.filter((F.col('academic_performance') > 80) & (F.col('burnout_score') > 7)) \
    .select('age', 'gender', 'academic_year', 'academic_performance', 'burnout_score', 'risk_level') \
    .orderBy(F.col('burnout_score').desc()) \
    .show(10)

# 5. Agregación Avanzada por Año Académico y Género
# Calculamos promedios de salud mental y riesgo de deserción
print("\n>>> Estadísticas de Bienestar por Año y Género:")
stats_df = df_clean.groupBy('academic_year', 'gender') \
    .agg(
        F.avg('mental_health_index').alias('Promedio_Salud_Mental'),
        F.avg('dropout_risk').alias('Riesgo_Desercion_Promedio'),
        F.count('*').alias('Total_Estudiantes')
    ).orderBy('academic_year', 'gender')

stats_df.show()

# 6. Uso de Window Functions (Funciones de Ventana)
# Vamos a rankear a los estudiantes con mayor estrés dentro de cada año académico
window_spec = Window.partitionBy("academic_year").orderBy(F.col("stress_level").desc())

print("\n>>> Estudiante con más estrés por cada Año Académico (Top 1):")
df_clean.withColumn("rank_estres", F.rank().over(window_spec)) \
    .filter(F.col("rank_estres") == 1) \
    .select('academic_year', 'age', 'gender', 'stress_level', 'exam_pressure') \
    .show()

# 7. Spark SQL (Para demostrar versatilidad)
# Si prefieres queries tradicionales, registramos como tabla temporal
df_clean.createOrReplaceTempView("estudiantes")
query_sql = spark.sql("""
    SELECT risk_level, COUNT(*) as cantidad, ROUND(AVG(sleep_hours), 2) as promedio_sueño
    FROM estudiantes
    GROUP BY risk_level
    ORDER BY cantidad DESC
""")
print("\n>>> Distribución por Nivel de Riesgo (vía SQL):")
query_sql.show()

# Finalizar sesión
# spark.stop()
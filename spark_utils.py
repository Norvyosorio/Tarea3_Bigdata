from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def create_spark_session(app_name, shuffle_partitions="10"):
    """Crea y retorna una SparkSession con configuracion estandar."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", shuffle_partitions) \
        .getOrCreate()


def load_csv(spark, file_path):
    """Carga un CSV desde HDFS con header e inferencia de esquema."""
    return spark.read.format('csv') \
        .option('header', 'true') \
        .option('inferSchema', 'true') \
        .load(file_path)


def round_float_columns(df, decimals=2):
    """Redondea todas las columnas double a la cantidad de decimales indicada."""
    float_cols = [c for c, t in df.dtypes if t == 'double']
    return df.select([
        F.round(F.col(c), decimals).alias(c) if c in float_cols else F.col(c)
        for c in df.columns
    ])


def show_analysis(title, df, n=20):
    """Imprime un encabezado descriptivo y muestra las primeras n filas del DataFrame."""
    print(f"\n>>> {title}")
    df.show(n)


def filter_and_rank(df, filter_expr, select_cols, order_col, ascending=False, limit=None):
    """Filtra, selecciona columnas y ordena un DataFrame.

    Args:
        df: DataFrame de entrada.
        filter_expr: Expresion de filtro PySpark.
        select_cols: Lista de nombres de columnas a seleccionar.
        order_col: Nombre de la columna para ordenar.
        ascending: Orden ascendente (default: descendente).
        limit: Cantidad de filas a mostrar (None = todas).
    """
    result = df.filter(filter_expr) \
        .select(*select_cols) \
        .orderBy(F.col(order_col).asc() if ascending else F.col(order_col).desc())
    return result

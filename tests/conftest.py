import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Shared SparkSession for all tests, using local mode."""
    session = SparkSession.builder \
        .master("local[1]") \
        .appName("TestAnalisisSaludMental") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
    yield session
    session.stop()

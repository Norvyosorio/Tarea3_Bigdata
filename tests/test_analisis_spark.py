"""Unit tests for analisis_spark.py functions."""

import sys
import os

import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analisis_spark import (
    create_spark_session,
    clean_dataframe,
    filter_high_performers_high_burnout,
    compute_wellbeing_stats,
    rank_stress_by_year,
    risk_level_distribution,
)


STUDENT_SCHEMA = StructType([
    StructField("age", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("academic_year", StringType(), True),
    StructField("academic_performance", DoubleType(), True),
    StructField("burnout_score", DoubleType(), True),
    StructField("risk_level", StringType(), True),
    StructField("mental_health_index", DoubleType(), True),
    StructField("dropout_risk", DoubleType(), True),
    StructField("stress_level", DoubleType(), True),
    StructField("exam_pressure", DoubleType(), True),
    StructField("sleep_hours", DoubleType(), True),
])

SAMPLE_DATA = [
    (20, "Male",   "Year_1", 85.123, 8.567, "High",   3.141, 0.789, 9.1, 8.0, 5.5),
    (21, "Female", "Year_1", 90.999, 9.111, "High",   2.718, 0.654, 7.5, 6.0, 6.0),
    (22, "Male",   "Year_2", 70.456, 5.234, "Medium", 5.555, 0.333, 4.2, 3.0, 7.5),
    (23, "Female", "Year_2", 60.789, 3.876, "Low",    7.890, 0.111, 2.1, 2.0, 8.0),
    (19, "Male",   "Year_1", 95.321, 6.543, "Medium", 4.444, 0.555, 8.8, 7.0, 6.5),
    (24, "Female", "Year_3", 88.888, 7.999, "High",   1.234, 0.999, 6.0, 5.0, 4.0),
]


@pytest.fixture
def sample_df(spark):
    """Create a sample DataFrame matching the expected schema."""
    return spark.createDataFrame(SAMPLE_DATA, schema=STUDENT_SCHEMA)


# --- Tests for create_spark_session ---

class TestCreateSparkSession:
    def test_returns_spark_session(self):
        session = create_spark_session(app_name="TestSession", shuffle_partitions="2")
        assert session is not None
        assert session.conf.get("spark.sql.shuffle.partitions") == "2"

    def test_custom_app_name(self):
        session = create_spark_session(app_name="CustomApp")
        assert "CustomApp" in session.conf.get("spark.app.name")


# --- Tests for clean_dataframe ---

class TestCleanDataframe:
    def test_rounds_double_columns(self, spark, sample_df):
        result = clean_dataframe(sample_df)
        rows = result.collect()
        first = rows[0]
        assert first["academic_performance"] == 85.12
        assert first["burnout_score"] == 8.57
        assert first["mental_health_index"] == 3.14

    def test_preserves_non_double_columns(self, spark, sample_df):
        result = clean_dataframe(sample_df)
        rows = result.collect()
        first = rows[0]
        assert first["age"] == 20
        assert first["gender"] == "Male"
        assert first["academic_year"] == "Year_1"
        assert first["risk_level"] == "High"

    def test_preserves_row_count(self, spark, sample_df):
        result = clean_dataframe(sample_df)
        assert result.count() == sample_df.count()

    def test_preserves_column_names(self, spark, sample_df):
        result = clean_dataframe(sample_df)
        assert result.columns == sample_df.columns

    def test_no_double_columns(self, spark):
        df = spark.createDataFrame(
            [(1, "a"), (2, "b")],
            schema=StructType([
                StructField("id", IntegerType()),
                StructField("name", StringType()),
            ])
        )
        result = clean_dataframe(df)
        assert result.collect() == df.collect()

    def test_empty_dataframe(self, spark):
        df = spark.createDataFrame([], schema=STUDENT_SCHEMA)
        result = clean_dataframe(df)
        assert result.count() == 0
        assert result.columns == df.columns


# --- Tests for filter_high_performers_high_burnout ---

class TestFilterHighPerformersHighBurnout:
    def test_default_thresholds(self, spark, sample_df):
        result = filter_high_performers_high_burnout(sample_df)
        rows = result.collect()
        for row in rows:
            assert row["academic_performance"] > 80
            assert row["burnout_score"] > 7

    def test_result_columns(self, spark, sample_df):
        result = filter_high_performers_high_burnout(sample_df)
        expected_cols = ["age", "gender", "academic_year", "academic_performance", "burnout_score", "risk_level"]
        assert result.columns == expected_cols

    def test_ordered_by_burnout_desc(self, spark, sample_df):
        result = filter_high_performers_high_burnout(sample_df)
        rows = result.collect()
        burnout_scores = [row["burnout_score"] for row in rows]
        assert burnout_scores == sorted(burnout_scores, reverse=True)

    def test_custom_thresholds(self, spark, sample_df):
        result = filter_high_performers_high_burnout(sample_df, perf_threshold=60, burnout_threshold=3)
        rows = result.collect()
        for row in rows:
            assert row["academic_performance"] > 60
            assert row["burnout_score"] > 3
        assert len(rows) >= 1

    def test_no_matches(self, spark, sample_df):
        result = filter_high_performers_high_burnout(sample_df, perf_threshold=99, burnout_threshold=99)
        assert result.count() == 0

    def test_excludes_boundary_values(self, spark):
        data = [
            (20, "Male", "Year_1", 80.0, 7.0, "Medium", 5.0, 0.5, 5.0, 5.0, 6.0),
            (21, "Female", "Year_1", 81.0, 7.1, "High", 4.0, 0.4, 4.0, 4.0, 7.0),
        ]
        df = spark.createDataFrame(data, schema=STUDENT_SCHEMA)
        result = filter_high_performers_high_burnout(df)
        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["academic_performance"] == 81.0


# --- Tests for compute_wellbeing_stats ---

class TestComputeWellbeingStats:
    def test_groups_by_year_and_gender(self, spark, sample_df):
        result = compute_wellbeing_stats(sample_df)
        result_cols = result.columns
        assert "academic_year" in result_cols
        assert "gender" in result_cols
        assert "Promedio_Salud_Mental" in result_cols
        assert "Riesgo_Desercion_Promedio" in result_cols
        assert "Total_Estudiantes" in result_cols

    def test_correct_student_count(self, spark, sample_df):
        result = compute_wellbeing_stats(sample_df)
        rows = result.collect()
        total = sum(row["Total_Estudiantes"] for row in rows)
        assert total == 6

    def test_aggregation_values(self, spark):
        data = [
            (20, "Male", "Year_1", 80.0, 5.0, "Low", 4.0, 0.2, 3.0, 3.0, 7.0),
            (21, "Male", "Year_1", 85.0, 6.0, "Low", 6.0, 0.4, 4.0, 4.0, 6.0),
        ]
        df = spark.createDataFrame(data, schema=STUDENT_SCHEMA)
        result = compute_wellbeing_stats(df)
        rows = result.collect()
        assert len(rows) == 1
        row = rows[0]
        assert row["Total_Estudiantes"] == 2
        assert abs(row["Promedio_Salud_Mental"] - 5.0) < 0.01
        assert abs(row["Riesgo_Desercion_Promedio"] - 0.3) < 0.01

    def test_ordered_by_year_and_gender(self, spark, sample_df):
        result = compute_wellbeing_stats(sample_df)
        rows = result.collect()
        keys = [(row["academic_year"], row["gender"]) for row in rows]
        assert keys == sorted(keys)

    def test_empty_dataframe(self, spark):
        df = spark.createDataFrame([], schema=STUDENT_SCHEMA)
        result = compute_wellbeing_stats(df)
        assert result.count() == 0


# --- Tests for rank_stress_by_year ---

class TestRankStressByYear:
    def test_returns_top_stress_per_year(self, spark, sample_df):
        result = rank_stress_by_year(sample_df)
        rows = result.collect()
        years = [row["academic_year"] for row in rows]
        assert "Year_1" in years
        assert "Year_2" in years
        assert "Year_3" in years

    def test_result_columns(self, spark, sample_df):
        result = rank_stress_by_year(sample_df)
        expected_cols = ["academic_year", "age", "gender", "stress_level", "exam_pressure"]
        assert result.columns == expected_cols

    def test_highest_stress_selected(self, spark):
        data = [
            (20, "Male",   "Year_1", 80.0, 5.0, "Low", 4.0, 0.2, 10.0, 8.0, 5.0),
            (21, "Female", "Year_1", 85.0, 6.0, "Low", 6.0, 0.4,  3.0, 2.0, 7.0),
        ]
        df = spark.createDataFrame(data, schema=STUDENT_SCHEMA)
        result = rank_stress_by_year(df)
        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["stress_level"] == 10.0
        assert rows[0]["age"] == 20

    def test_ties_return_all_tied(self, spark):
        data = [
            (20, "Male",   "Year_1", 80.0, 5.0, "Low", 4.0, 0.2, 9.0, 8.0, 5.0),
            (21, "Female", "Year_1", 85.0, 6.0, "Low", 6.0, 0.4, 9.0, 7.0, 6.0),
        ]
        df = spark.createDataFrame(data, schema=STUDENT_SCHEMA)
        result = rank_stress_by_year(df)
        rows = result.collect()
        assert len(rows) == 2

    def test_empty_dataframe(self, spark):
        df = spark.createDataFrame([], schema=STUDENT_SCHEMA)
        result = rank_stress_by_year(df)
        assert result.count() == 0


# --- Tests for risk_level_distribution ---

class TestRiskLevelDistribution:
    def test_groups_by_risk_level(self, spark, sample_df):
        result = risk_level_distribution(spark, sample_df)
        result_cols = result.columns
        assert "risk_level" in result_cols
        assert "cantidad" in result_cols
        assert "promedio_sueno" in result_cols

    def test_correct_counts(self, spark, sample_df):
        result = risk_level_distribution(spark, sample_df)
        rows = result.collect()
        total = sum(row["cantidad"] for row in rows)
        assert total == 6

    def test_ordered_by_count_desc(self, spark, sample_df):
        result = risk_level_distribution(spark, sample_df)
        rows = result.collect()
        counts = [row["cantidad"] for row in rows]
        assert counts == sorted(counts, reverse=True)

    def test_sleep_hours_average(self, spark):
        data = [
            (20, "Male",   "Year_1", 80.0, 5.0, "High", 4.0, 0.2, 5.0, 5.0, 6.0),
            (21, "Female", "Year_1", 85.0, 6.0, "High", 6.0, 0.4, 4.0, 4.0, 8.0),
        ]
        df = spark.createDataFrame(data, schema=STUDENT_SCHEMA)
        result = risk_level_distribution(spark, df)
        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["risk_level"] == "High"
        assert rows[0]["cantidad"] == 2
        assert rows[0]["promedio_sueno"] == 7.0

    def test_all_risk_levels_present(self, spark, sample_df):
        result = risk_level_distribution(spark, sample_df)
        rows = result.collect()
        levels = {row["risk_level"] for row in rows}
        assert levels == {"High", "Medium", "Low"}

    def test_empty_dataframe(self, spark):
        df = spark.createDataFrame([], schema=STUDENT_SCHEMA)
        result = risk_level_distribution(spark, df)
        assert result.count() == 0

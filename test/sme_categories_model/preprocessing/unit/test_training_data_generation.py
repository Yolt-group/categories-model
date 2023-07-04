import pandas as pd
import pyspark.sql.functions as f

from categories_model.preprocessing.create_training_data import (
    match_column,
    match_category_rule,
    remove_diacritics,
)


def test_remove_diacritics(spark_session):
    input = pd.DataFrame(["privé", "garçon", "meeëten"], columns=["test"])
    expected = pd.DataFrame(["prive", "garcon", "meeeten"], columns=["test"])
    spark_input_df = spark_session.createDataFrame(input)
    spark_output_df = spark_input_df.withColumn(
        "test", remove_diacritics(f.col("test"))
    )
    output = spark_output_df.toPandas()
    pd.testing.assert_frame_equal(output, expected)
    assert True


def test_clean_column(cleaned_df):
    df = cleaned_df.toPandas()
    pd.testing.assert_series_equal(df["description"], df["expected"], check_names=False)


def test_match_column(cleaned_df):
    match = match_column(column_name="description", word_list=["prive", "etentje"])
    match_df = cleaned_df.where(match).toPandas()
    assert len(match_df) == 2


def test_match_category_rule(cleaned_df, training_data_test_rule):
    df = match_category_rule(cleaned_df, training_data_test_rule)
    assert df.count() == cleaned_df.where(f.col("should_match")).count()

from typing import Optional

import pyspark.sql.functions as sf
from pyspark.ml import Model, Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame


class _BTransformer(Transformer):
    def __init__(self, inputCol: str):
        self.inputCol = inputCol
        super(_BTransformer, self).__init__()

    def _transform(self, dataset: DataFrame) -> DataFrame:
        pass


class _IdentityTransformer(_BTransformer):
    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset


class _LogTransformer(_BTransformer):
    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.withColumn(self.inputCol, sf.log(dataset[self.inputCol]))


class _FXTransformer(_BTransformer):
    def _transform(self, dataset):
        return dataset.withColumn(self.inputCol, dataset[self.inputCol] * (1.0 - dataset[self.inputCol]))


class _LogFXTransformer(_BTransformer):
    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.withColumn(self.inputCol, sf.log(dataset[self.inputCol] * (1.0 - dataset[self.inputCol])))


class _SqrtTransformer(_BTransformer):
    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.withColumn(self.inputCol, sf.sqrt(dataset[self.inputCol]))


class _SqrtExpTransformer(_BTransformer):
    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.withColumn(self.inputCol, sf.sqrt(sf.exp(dataset[self.inputCol])))


class BModel(PipelineModel):
    def __init__(self, stages: list[_BTransformer], linreg_idx: int):
        super().__init__(stages=stages)
        self.linreg_idx = linreg_idx

    def _linreg(self):
        return self.stages[self.linreg_idx]

    def transform(self, dataset: DataFrame) -> DataFrame:
        prediction = super().transform(dataset).select("estimated_b")
        prediction = prediction.withColumn("__row_id", sf.monotonically_increasing_id())

        dataset = dataset.withColumn("__row_id", sf.monotonically_increasing_id())
        df_combined = dataset.join(prediction, on="__row_id", how="inner").drop("__row_id")
        return df_combined

    def predict(self, dataset: DataFrame) -> DataFrame:
        return self.transform(dataset)

    def summary(self):
        return self._linreg().summary

    def r2(self) -> float:
        return self.summary().r2

    def coefs(self) -> list[float]:
        return self._linreg().coefficients


class BPipeline(Pipeline):
    def __init__(
        self,
        x_col_name: str,
        y_col_name: str,
        X_Transform: _BTransformer = _IdentityTransformer,
        Y_Transform: _BTransformer = _IdentityTransformer,
    ):
        if Y_Transform == _LogTransformer:
            ResponseTranform = _SqrtExpTransformer
        else:
            ResponseTranform = _SqrtTransformer
        self.linreg_idx = 3
        stages = [
            Y_Transform(inputCol=y_col_name),
            X_Transform(inputCol=x_col_name),
            VectorAssembler(inputCols=[x_col_name], outputCol="__features"),
            LinearRegression(featuresCol="__features", labelCol=y_col_name, predictionCol="estimated_b"),
            ResponseTranform(inputCol="estimated_b"),
        ]
        super().__init__(stages=stages)

    def _fit(self, dataset: DataFrame) -> BModel:
        model = super()._fit(dataset)
        return BModel(model.stages[1:], self.linreg_idx - 1)


def get_best_b_model(dataset: DataFrame, x_col_name: str, y_col_name: str) -> BModel:
    """train a model to predict the effect size B based on the MAF.

    Args:
        dataset (DataFrame): _description_
        x_col_name (str): _description_
        y_col_name (str): _description_

    Returns:
        BModel: _description_
    """
    best_model = None
    best_r2 = -1.0
    for X_Transform in [_IdentityTransformer, _LogTransformer, _FXTransformer, _LogFXTransformer]:
        for Y_Transform in [_IdentityTransformer, _LogTransformer]:
            model = BPipeline(x_col_name, y_col_name, X_Transform, Y_Transform).fit(dataset)
            r2 = model.r2()
            if r2 > best_r2:
                best_model = model
                best_r2 = r2
    return best_model


def estimate_b(
    test_df: DataFrame,
    maf_col_name: str,
    train_binary: bool,
    test_binary: bool,
    model: Optional[Model] = None,
    train_df: Optional[DataFrame] = None,
    beta_col_name: Optional[str] = None,
    test_se: Optional[str] = None,
    test_prevalence: Optional[float] = None,
    train_z: Optional[str] = None,
    train_n: Optional[str] = None,
) -> DataFrame:
    """predict an estimation for B, the effect size, based on the MAF

    Args:
        test_df (DataFrame): test data, what the prediction is based off
        maf_col_name (str): column containing the MAF data in both data sets
        train_binary (bool): whether the training trait was binary
        test_binary (bool): whether the trait of interest is binary
        model (Model, optional): fitted model to estimate B. If not given, one will be trained with `get_best_b_model` and `train_df`
        train_df (DataFrame, optional): training data. required if no model is given.
        beta_col_name (str, optional): column containing the known value of beta in the training data. required if no model is given.
        test_se (str, optional): column containing the standard error of the MAF for the test data. required if `train_binary != test_binary`. Defaults to None.
        test_prevalence (float, optional): prevalence of the trait in the test data. required if `train_binary and not test_binary`. Defaults to None.
        train_z (str, optional): column containing Z scores of the training data. required if `train_binary != test_binary`. Defaults to None.
        train_n (str, optional): column containing the sample size in the training data. required if `train_binary != test_binary`. Defaults to None.

    Returns:
        DataFrame: the test dataset with a new column `estimated_b`
    """
    if model is None:
        if train_binary == test_binary:
            y = train_df[beta_col_name] ** 2.0
        else:
            y = (
                (train_df[train_z] ** 2.0)
                / (2.0 * train_df[train_n] * train_df[maf_col_name])
                * (1.0 - train_df[maf_col_name])
            )
        x = train_df[maf_col_name]
        train_df = train_df.withColumns({'__y': y, '__x': x}).select("__x", "__y")
        model = get_best_b_model(train_df, "__x", "__y")
    test_df_x = test_df.select(test_df[maf_col_name].alias("__x"))
    prediction = model.transform(test_df_x)
    if train_binary != test_binary:
        if not test_binary:
            prediction['estimated_b'] = prediction['estimated_b'] * test_df[test_se]
        else:
            prediction['estimated_b'] = prediction['estimated_b'] / sf.sqrt(
                test_df[test_prevalence] * (1.0 - test_df[test_se])
            )

    return prediction

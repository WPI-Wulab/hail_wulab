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
    def __init__(self, stages: list[_BTransformer], linreg_idx: int, prediction_col: str = "estimated_b"):
        """The fitted model to predict effect size B based on the MAF

        Args:
            stages (list[_BTransformer]): transformations and predictors used to transform the MAF, predict B, and transform the output
            linreg_idx (int): index of the actuall predictor in the stages list
            prediction_col (str, optional): what to call the result. Defaults to "estimated_b".
        """
        super().__init__(stages=stages)
        self.linreg_idx = linreg_idx
        self.prediction_col = prediction_col

    def _linreg(self):
        return self.stages[self.linreg_idx]

    def transform(self, dataset: DataFrame) -> DataFrame:
        prediction = super().transform(dataset).select(self.prediction_col)
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
    """class containing the pipeline to fit a linear model relating the MAF to the effect size B"""

    def __init__(
        self,
        x_col_name: str,
        y_col_name: str,
        X_Transform: _BTransformer = _IdentityTransformer,
        Y_Transform: _BTransformer = _IdentityTransformer,
        prediction_col: str = "estimated_b",
    ):
        """Class containing pipeline to fit a linear model relating the MAF to the effect size B

        Args:
            x_col_name (str): column name of the X values, the MAF
            y_col_name (str): column name of the Y values, B the effect size
            X_Transform (_BTransformer, optional): transformation to be applied to the MAF. Defaults to _IdentityTransformer.
            Y_Transform (_BTransformer, optional): transformation to be applied to B. Defaults to _IdentityTransformer.
            prediction_col (str, optional): what to call the values predicted by the eventual model. Defaults to "estimated_b".
        """
        if Y_Transform == _LogTransformer:
            ResponseTranform = _SqrtExpTransformer
        else:
            ResponseTranform = _SqrtTransformer
        self.linreg_idx = 3
        self.prediction_col = prediction_col
        stages = [
            Y_Transform(inputCol=y_col_name),
            X_Transform(inputCol=x_col_name),
            VectorAssembler(inputCols=[x_col_name], outputCol="__features"),
            LinearRegression(featuresCol="__features", labelCol=y_col_name, predictionCol=prediction_col),
            ResponseTranform(inputCol=prediction_col),
        ]
        super().__init__(stages=stages)

    def _fit(self, dataset: DataFrame) -> BModel:
        model = super()._fit(dataset)
        print(self.prediction_col)
        dataset.show()
        return BModel(model.stages[1:], linreg_idx=self.linreg_idx - 1, prediction_col=self.prediction_col)


def get_best_b_model(
    dataset: DataFrame,
    x_col_name: str,
    y_col_name: str,
    prediction_col: str = "estimated_b",
) -> BModel:
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
            model = BPipeline(x_col_name, y_col_name, X_Transform, Y_Transform, prediction_col=prediction_col).fit(
                dataset
            )
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
    prediction_col: str = "estimated_b",
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
        prediction_col (str, optional): what to call the estimated B in the resulting DataFrame

    Returns:
        DataFrame: the test dataset with a new column `estimated_b`
    """
    if model is None:
        if train_binary == test_binary:
            y = train_df[beta_col_name] ** 2.0
        else:
            y = (
                (train_df[train_z] ** 2.0)
                / (2.0 * train_df[train_n] * train_df[beta_col_name])
                * (1.0 - train_df[beta_col_name])
            )
        train_df = train_df.withColumn(beta_col_name, y)
        model = get_best_b_model(
            train_df, x_col_name=maf_col_name, y_col_name=beta_col_name, prediction_col=prediction_col
        )
    prediction = model.transform(test_df)
    if train_binary != test_binary:
        if not test_binary:
            prediction[prediction_col] = prediction[prediction_col] * test_df[test_se]
        else:
            prediction[prediction_col] = prediction[prediction_col] / sf.sqrt(
                test_df[test_prevalence] * (1.0 - test_df[test_se])
            )

    return prediction

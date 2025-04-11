import warnings
from typing import Any, Optional

import pyspark.sql.functions as sf
from numpy import logspace
from pyspark.ml import Model, Pipeline, PipelineModel, Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from hail import ir
from hail.expr import expr_any, expr_float64, matrix_table_source
from hail.table import Table
from hail.typecheck import sequenceof, typecheck
from hail.utils.java import Env


@typecheck(
    key=expr_any,
    b=expr_float64,
    pi=expr_float64,
    geno=expr_float64,
    covariates=sequenceof(expr_float64),
    pheno=expr_float64,
    logistic=bool,
    method=str,
    use_t=bool,
)
def glow(key, b, pi, geno, covariates, pheno, logistic: bool, method: str, use_t: bool = False) -> Table:
    # TODO check that fields are correct type and from the correct mt

    mt = matrix_table_source("gfisher", key)
    key_name_out = mt._fields_inverse[key] if key in mt._fields_inverse else "id"

    row_exprs = {'__key': key, '__b': b, '__pi': pi}

    # FIXME: remove this logic when annotation is better optimized
    if geno in mt._fields_inverse:
        x_field_name = mt._fields_inverse[geno]
        entry_expr = {}
    else:
        x_field_name = Env.get_uid()
        entry_expr = {x_field_name: geno}
    method = method.upper()
    if method not in ["FISHER", "SKAT", "BURDEN", "OMNI"]:
        raise ValueError(f"Method must be one of (FISHER, SKAT, BURDEN, OMNI). Got {method}")

    y_field_name = '__y'
    cov_field_names = list(f'__cov{i}' for i in range(len(covariates)))
    mt = mt._select_all(
        col_exprs=dict(**{y_field_name: pheno}, **dict(zip(cov_field_names, covariates))),
        row_exprs=row_exprs,
        entry_exprs=entry_expr,
    )
    config = {
        "name": "GLOW",
        "keyField": "__key",
        "keyFieldOut": key_name_out,
        "bField": "__b",
        "piField": '__pi',
        'genoField': x_field_name,
        'covFields': cov_field_names,
        "phenoField": y_field_name,
        "logistic": logistic,
        "method": method,
        "useT": use_t,
    }
    return Table(ir.MatrixToTableApply(mt._mir, config)).persist()


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
    model: Optional[Model | Any] = None,
    train_df: Optional[DataFrame] = None,
    beta_col_name: Optional[str] = None,
    test_se: Optional[str] = None,
    test_prevalence: Optional[float] = None,
    train_z: Optional[str] = None,
    train_n: Optional[str] = None,
    prediction_col: Optional[str] = "estimated_b",
) -> DataFrame:
    """predict an estimation for B, the effect size, based on the MAF

    Args:
        test_df (DataFrame): test data, what the prediction is based off
        maf_col_name (str): column containing the MAF data in both data sets
        train_binary (bool): whether the training trait was binary
        test_binary (bool): whether the trait of interest is binary
        model (Model | Any, optional): fitted model to estimate B. must implement a `transform` method that accepts and returns a pyspark.sql.DataFrame. If not given, one will be trained with `get_best_b_model` and `train_df`
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


def get_pi_models(
    train_df: DataFrame,
    x_col_name: Optional[str | list[str]] = "features",
    y_col_name: Optional[str] = "class",
    prediction_col: Optional[str] = "estimated_pi",
    n_controls: Optional[int] = None,
    n_models: Optional[int] = None,
    l1_reg: Optional[bool] = True,
):
    """train models to predict causal likelihood pi based on annotations

    Args:
        train_df (DataFrame): training data.
        x_col_name (str | list[str], optional): name of the vector features column, or list of the features to be vectorized. if a list, the resulting models will have a feature column of "__features". Defaults to "features".
        y_col_name (str, optional): name of the column indicating whether it is a case or control. Column must contain only binary values, 0 for control and 1 for case. Defaults to "case".
        prediction_col (str, optional): what the models should call their prediction. Defaults to "estimated_pi".
        n_controls (int, optional): maximum number of controls to use when fitting models. If not provided, all will be used.
        n_models (int, optional): number of models to fit. Defaults to 5.
    """
    if n_models is None:
        raise ValueError("n_models must be provided")

    train = train_df
    need_assembler = isinstance(x_col_name, list)
    lr_feature_name = "__features" if need_assembler else x_col_name
    logreg = LogisticRegression(featuresCol=lr_feature_name, labelCol=y_col_name, probabilityCol=prediction_col)

    if l1_reg:
        n = train.count()
        logreg.setElasticNetParam(1.0)
        if need_assembler:
            n_features = len(x_col_name)
        else:
            n_features = train.schema[x_col_name].metadata["ml_attr"]["num_attrs"]

        xyabs = [
            (
                train.agg(
                    sf.sum(
                        sf.abs(
                            sf.col(y_col_name)
                            * (sf.col(x_col_name[i]) if need_assembler else sf.get(sf.col(x_col_name), i))
                        )
                    )
                )
            ).collect()[0][0]
            for i in range(n_features)
        ]

        lambda_max = max(xyabs) / n
        lambda_min = lambda_max * (0.01 if n < n_features else 1e-4)

        grid = ParamGridBuilder().addGrid(logreg.regParam, logspace(lambda_max, lambda_min, 100)).build()
        evaluator = BinaryClassificationEvaluator(labelCol=y_col_name, rawPredictionCol=logreg.getRawPredictionCol())
        logreg = CrossValidator(estimator=logreg, estimatorParamMaps=grid, evaluator=evaluator)
    # replace later
    if need_assembler:
        # have to combine the columns into a single vector.
        # train = VectorAssembler(inputCols=x_col_name, outputCol="__features").transform(train)
        model = Pipeline(stages=[VectorAssembler(inputCols=x_col_name, outputCol="__features"), logreg])
        # model will now be fit on a column called __features
        # print("warning, the models will now expect a column called __features.")
        # x_col_name = "__features"
    else:
        model = logreg

    if n_controls is None:
        warnings.warn(
            "n_controls not provided, using all controls. This means that each model will be trained on the same data"
        )
        ctrl_size = 1.0
    else:
        ctrl_size = n_controls / train.filter(f"{y_col_name} == 0").count()
    models = [
        model.fit(train.sampleBy(y_col_name, fractions={0: ctrl_size, 1: 1.0}).sample(1.0)) for _ in range(n_models)
    ]
    return models


def estimate_pi(
    test_df: DataFrame,
    models: Optional[list[Model | Any]] = None,
    prediction_col: Optional[str] = "estimated_pi",
    **kwargs,
):
    """estimate the causal likelihood pi based on annotations from testing data using the best model from training

    Args:
        test_df (DataFrame): testing data to estimate pi for
        models (list[Model], optional): trained models to predict pi with. models must implement a `transform` method that accepts and returns a pyspark.sql.DataFrame. if not given, models will be trained with `get_pi_models`.
        **kwargs: arguments to pass to `get_pi_models`

    Returns:
        DataFrame: the test data with new columns `prediction_{i}` for each model
    """

    if models is None:
        models = get_pi_models(prediction_col=prediction_col, **kwargs)

    colnames = test_df.columns
    test = test_df
    for name in colnames:
        test = test.withColumnRenamed(name, name.replace(".", "_"))

    test_df = test_df.withColumns({
        prediction_col: sf.lit(0.0),
        "id": sf.row_number().over(Window().orderBy(sf.lit('A'))),
    })
    for i, model in enumerate(models):
        pred_i = f"{prediction_col}_{i}"
        preds = (
            model.transform(test)
            .withColumn(prediction_col, sf.get(vector_to_array(prediction_col), 1))
            .select(prediction_col)
            .withColumnRenamed(prediction_col, pred_i)
            .withColumn("id", sf.row_number().over(Window().orderBy(sf.lit('A'))))
        )
        test_df = test_df.join(preds, on="id")
        test_df = test_df.withColumn(prediction_col, test_df[prediction_col] + test_df[pred_i]).drop(pred_i)
    test_df = test_df.withColumn(prediction_col, test_df[prediction_col] / len(models)).drop("id")
    return test_df

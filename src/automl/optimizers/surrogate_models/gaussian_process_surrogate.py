from automl.optimizers.surrogate_models.base_surrogate_model import BaseSurrogateModel
from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from automl.enums import Task
from automl.config import TIME_SERIES_CONFIG
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from automl.functions import createPipeline


class GaussianProcessSurrogate(BaseSurrogateModel):
    def __init__(self, task: Task):
        """
        A unified Gaussian Process surrogate that branches to GPR or GPC:
            - Classification -> GaussianProcessClassifier with Matern kernel.
            - Regression    -> GaussianProcessRegressor with RBF kernel.
        - Time Series -> Placeholder for future implementation.
        """
        self.task = task

        if self.task == Task.CLASSIFICATION:
            kernel = C(1.0, (1e-3, 1e5)) * Matern(
                length_scale=1.0,
                nu=1.5,
                length_scale_bounds=(1e-3, 1e5)
            )
            self.model = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=20
            )
        elif self.task == Task.REGRESSION:
            kernel = C(1.0, (1e-3, 1e5)) * RBF(
                length_scale=1.0,
                length_scale_bounds=(1e-3, 1e5)
            )
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-7,
                n_restarts_optimizer=20
            )
        elif self.task == Task.TIME_SERIES:
            self.model = None
        else:
            raise ValueError(f"Unsupported task: {self.task}")
    


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit either a GPC, GPR, ARIMA or SARIMA depending on the task.
        """
        if self.task == Task.REGRESSION or self.task == Task.CLASSIFICATION:
            self.model.fit(X, y)
        elif self.task == Task.TIME_SERIES:
            if y is None:
                raise ValueError("For time series forecasting, target data (y) must be provided.")
            else:
                params_sarima = TIME_SERIES_CONFIG["models"].get("SARIMA", {})
                params_arima = TIME_SERIES_CONFIG["models"].get("ARIMA", {})
                order_sarima = params_sarima.get("order")
                order_arima = params_arima.get("order")
                seasonal_order = params_sarima.get("seasonal_order")
                if seasonal_order is None:
                    self.model = ARIMA(y, order=order_arima)
                else:
                    self.model = SARIMAX(y, order=order_sarima, seasonal_order=seasonal_order)
                self.model.fit()


    def predict(self, X: np.ndarray, return_std: bool = False) -> tuple[Any, None] | tuple[Any, Any]:
        """
        - For regression (GPR): returns (mean, std) if return_std=True.
        - For classification (GPC):
            -> predict() returns discrete class labels,
               no built-in standard deviation concept.
               If return_std=True, raises NotImplementedError.
        """
        if self.task == Task.CLASSIFICATION:
            if return_std:
                raise NotImplementedError(
                    "GaussianProcessClassifier doesn't provide std dev directly. "
                    "Use predict_proba or a custom uncertainty measure if needed."
                )
            labels = self.model.predict(X)
            return labels, None
        elif self.task == Task.REGRESSION:
            if return_std:
                y_mean, y_std = self.model.predict(X, return_std=True)
                return y_mean, y_std
            else:
                y_mean = self.model.predict(X, return_std=False)
                return y_mean, None
        elif self.task == Task.TIME_SERIES:
            forecast = self.model.forecast(steps=10)
            return forecast, None
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        

if __name__ == "__main__":
    df = pd.read_csv(r'/Users/macbookpro/Desktop/university projects/Graduation Project/automl/Train.csv')
    print(df)
    pl = createPipeline(df, 'Sales_Quantity')
    df = pl.transform(df)
    gps = GaussianProcessSurrogate(task = Task.TIME_SERIES)

    # hpo = bayesian_optimizer_hyperband(task = Task.CLASSIFICATION, time_budget=150)
    X = df.drop(columns=['Sales_Quantity'])
    y = df['Sales_Quantity'].values
    gps.fit(X=None,y=y)
    forecast_steps = 10
    forecast = gps.predict(X=None, return_std=False)
    rmse = gps.get_metric_value()
    print(rmse)
    print(f"Forecasted values: {forecast}")

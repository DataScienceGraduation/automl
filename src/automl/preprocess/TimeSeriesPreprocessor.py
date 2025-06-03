from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, time_column=None, target_column=None, fill_method='ffill', date_format=None):
        self.time_column = time_column
        self.target_column = target_column
        self.fill_method = fill_method
        self.date_format = date_format

    def fit(self, X, y=None):
        if self.time_column is None:
            datetime_cols = X.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
            if not datetime_cols:
                for col in X.columns:
                    try:
                        parsed_col = pd.to_datetime(X[col], format=self.date_format, errors='coerce')
                        if parsed_col.notna().sum() > 0:
                            datetime_cols.append(col)
                    except (ValueError, TypeError):
                        continue
            if not datetime_cols:
                raise ValueError("No datetime column found. Please specify time_column manually.")
            self.time_column = datetime_cols[0]
        return self

    def transform(self, X):
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X[self.time_column]):
            X[self.time_column] = pd.to_datetime(X[self.time_column], format=self.date_format, errors='coerce')

        X = X.dropna(subset=[self.time_column])
        X = X.sort_values(by=self.time_column).reset_index(drop=True)

        if self.fill_method == 'ffill':
            X = X.ffill()
        elif self.fill_method == 'bfill':
            X = X.bfill()
        else:
            raise ValueError("fill_method must be either 'ffill' or 'bfill'")
        
        return X

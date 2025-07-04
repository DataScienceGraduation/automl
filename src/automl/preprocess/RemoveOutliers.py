from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest

class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        
    def fit(self, X, y=None):
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) == 0:
            # If no numerical columns, return self without fitting
            return self
        self.iso_forest_ = IsolationForest(contamination=self.contamination, random_state=42)
        self.iso_forest_.fit(X[numerical_cols])
        return self
    
    def transform(self, X):
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) == 0:
            # If no numerical columns, return X as is
            return X
        outliers = self.iso_forest_.predict(X[numerical_cols])
        return X[outliers != -1]
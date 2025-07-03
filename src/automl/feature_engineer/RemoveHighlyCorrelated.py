import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RemoveHighlyCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None, correlation_threshold=0.8):
        self.correlation_threshold = correlation_threshold
        self.target_variable = target_variable
    
    def fit(self, X, y=None):
        # Combine X and y for correlation calculation
        df = X.copy()

        if self.target_variable is not None and self.target_variable in df.columns:
            if not np.issubdtype(df[self.target_variable].dtype, np.number):
                df[self.target_variable] = df[self.target_variable].astype('category').cat.codes

        # Find numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'int32', 'int16', 'int8', 'float64', 'float32', 'float16']).columns

        # Correlation matrix
        corr_matrix = df[numerical_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        self.columns_to_drop_ = []

        # Find highly correlated pairs and drop one of them
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if upper_tri.loc[row, col] > self.correlation_threshold:
                    # For clustering, just drop the first column of the pair
                    if self.target_variable is None:
                        self.columns_to_drop_.append(col)
                    else:
                        # For other tasks, drop based on correlation with target
                        if abs(df[col].corr(df[self.target_variable])) < abs(df[row].corr(df[self.target_variable])):
                            self.columns_to_drop_.append(col)
                        else:
                            self.columns_to_drop_.append(row)

        # Deduplicate and ensure we don't drop the target variable itself
        self.columns_to_drop_ = list(set(self.columns_to_drop_))
        if self.target_variable is not None and self.target_variable in self.columns_to_drop_:
            self.columns_to_drop_.remove(self.target_variable)

        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')
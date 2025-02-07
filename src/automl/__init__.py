"""
automl package

This package provides automated machine learning functionality including feature engineering,
hyperparameter optimization, and data preprocessing.
"""

# Import from enums subpackage
from .enums import Task, Metric  # Alternatively, list specific names, e.g. from .enums.enums import MyEnum

# Import from feature_engineer subpackage
from .feature_engineer import CustomStandardScaler, RemoveHighlyCorrelated

# Import from optimizers subpackage
from .optimizers import BaseOptimizer, BayesianOptimizer, RandomSearchOptimizer
# And also import the subpackages for acquisition functions and surrogate models:
from .optimizers.acquisition_functions import BaseAcquistionFunction, ExpectedImprovement
from .optimizers.surrogate_models import BaseSurrogateModel, GaussianProcessSurrogate

# Import from preprocess subpackage
from .preprocess import (CustomLabelEncoder, DropHighCardinality, DropHighMissing,
                         DropNullRows, DropSingleValueColumns, RemoveDuplicates, RemoveOutliers)

# Import additional utility functions if desired
from .functions import createPipeline

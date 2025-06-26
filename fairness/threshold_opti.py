#%%
import pandas as pd
from fairlearn.metrics import (demographic_parity_difference,
                               demographic_parity_ratio)
from fairlearn.postprocessing import (ThresholdOptimizer,
                                      plot_threshold_optimizer)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from neural_net.preprocessing import get_train_test_data
from utils.graphs import compare
from utils.pipelines import Pipelines


#%%
def threshold_opti(X_train, X_test, y_train, sf_train, sf_test):
    threshold_optimizer = ThresholdOptimizer(
        estimator=Pipelines(X_train.columns).regressor(),
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        # prefit=True,
        predict_method="predict",
    )

    threshold_optimizer.fit(X_train, y_train, sensitive_features=sf_train)

    y_pred = threshold_optimizer.predict(X_test, sensitive_features=sf_test)
    plot_threshold_optimizer(threshold_optimizer) 
    
    return y_pred  
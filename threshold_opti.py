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

from utils.graphs import compare
from utils.pipelines import Pipelines

#%%
X_train_data = pd.read_csv("law_data.csv")
y_train_data = X_train_data.pop("first_pf")

# X_train_data_encoded = pd.get_dummies(X_train_data, drop_first=True)
A = X_train_data["race"]

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X_train_data, y_train_data, A, test_size=0.2, random_state=42
)

#%%
threshold_optimizer = ThresholdOptimizer(
    estimator=Pipelines(X_train_data.columns).regressor(),
    constraints="equalized_odds",
    objective="balanced_accuracy_score",
    # prefit=True,
    predict_method="predict",
)

#%%
threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)

print(threshold_optimizer)
#%%
y_pred = threshold_optimizer.predict(X_test, sensitive_features=A_test)

compare(
    [
        classification_report(y_test, y_pred, output_dict=True)
    ],
    label="Threshold Optimizer Results",
    model_names=["Threshold Optimizer"],
    print_output=True
)

print("Demographic Parity Metrics:  \nbased on ethinicity")
print(demographic_parity_difference(y_test, y_pred, sensitive_features=A_test))
print(demographic_parity_ratio(y_test, y_pred, sensitive_features=A_test))
#%%
plot_threshold_optimizer(threshold_optimizer)

#%%
ethnicities = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other", "Puertorican", "White"]

# compare fairness for each ethnicity

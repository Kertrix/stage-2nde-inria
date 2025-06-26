import pandas as pd
from fairlearn.preprocessing import CorrelationRemover
from sklearn.linear_model import LogisticRegression

from neural_net.preprocessing import get_train_test_data
from utils.pipelines import Pipelines


def cr():
    """
    This function demonstrates the use of CorrelationRemover to mitigate bias in a dataset.
    It trains a logistic regression model before and after applying CorrelationRemover,
    and returns the predictions.
    """
        
    X_train, X_test, y_train, y_test, sf_train, sf_test = get_train_test_data("law_data.csv", "first_pf", scale=False)

    print(type(X_train))

    # Train a logistic regressor before applying CorrelationRemover
    regressor = Pipelines(X_train.columns).regressor()
    regressor.fit(X_train, y_train)
    
    preds = regressor.predict(X_test)

    # Apply CorrelationRemover to remove correlation with sensitive features
    race_columns = [col for col in X_train.columns if col.startswith('race_')]

    corr_remover = CorrelationRemover(sensitive_feature_ids=race_columns)

    dcr = corr_remover.fit_transform(X_train)
    dcr = pd.DataFrame(dcr, columns=X_train.columns.drop(race_columns))
    dcr[race_columns] = X_train[race_columns].reset_index(drop=True)

    dcr_test = corr_remover.fit_transform(X_test)
    dcr_test = pd.DataFrame(dcr_test, columns=X_test.columns.drop(race_columns))
    dcr_test[race_columns] = X_test[race_columns].reset_index(drop=True)

    # Train a new logistic regression on the transformed data
    regressor_corr = Pipelines(X_train.columns).regressor()
    regressor_corr.fit(dcr, y_train)

    # Predict and evaluate
    pred_corr = regressor_corr.predict(dcr_test)

    return preds, pred_corr

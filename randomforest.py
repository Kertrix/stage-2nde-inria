#%%
import os
import sys

from utils.pipelines import Pipelines

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import pandas as pd
import torch
from fairlearn.metrics import (demographic_parity_difference,
                               demographic_parity_ratio,
                               equalized_odds_difference, equalized_odds_ratio,
                               false_negative_rate)
from fairlearn.preprocessing import CorrelationRemover
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from neural_net.preprocessing import prepare_data
from utils.graphs import compare, compare_fairness, compare_for_one_model
from utils.neural_utils import NeuralNetwork, predict

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ethnicities = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other", "Puertorican", "White"]

X_train_data = pd.read_csv("law_data.csv")
y_train_data = X_train_data.pop("first_pf")

A = X_train_data["race"]

X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(X_train_data, y_train_data, A, test_size=0.3, random_state=42, shuffle=True)

print(X_test.shape)
#%%
neural_network = NeuralNetwork(input_size=25)
neural_network.load_state_dict(torch.load('neural_net/model.pth', map_location=torch.device('cpu')))
#%%
rf = Pipelines(X_train.columns).random_forest()
rf.fit(X_train, y_train)

regressor = Pipelines(X_train.columns).regressor()
regressor.fit(X_train, y_train)

#%%
print(X_test.shape)

predictions = {
    "rf": rf.predict(X_test),
    "reg": regressor.predict(X_test),
    "nn": predict(neural_network, prepare_data(X_test))
}

# print(classification_report(y_test, predictions["nn"]))

compare(
    [classification_report(y_test, predictions["rf"], output_dict=True),
        classification_report(y_test, predictions["reg"], output_dict=True),
        classification_report(y_test, predictions["nn"], output_dict=True)
        ],
        model_names=["Random Forest", "Logistic Regression", "Neural Network"],
)
    
# %%
# Prediction per sex
results_sex = {
    "rf": {},
    "reg": {},
    "nn": {}
}

sex = X_test.groupby("sex")
for name, groups in sex:
    pred_rf = rf.predict(groups)
    pred_reg = regressor.predict(groups)
    pred_nn = predict(neural_network, prepare_data(groups))
    
    results_sex["rf"][name] = classification_report(y_test.loc[groups.index], pred_rf, output_dict=True)
    results_sex["reg"][name] = classification_report(y_test.loc[groups.index], pred_reg, output_dict=True)
    results_sex["nn"][name] = classification_report(y_test.loc[groups.index], pred_nn, output_dict=True)
    
    print("\n", name, groups.shape[0])
    # compare(
    #     [classification_report(y_test.loc[groups.index], pred_rf, output_dict=True),
    #     classification_report(y_test.loc[groups.index], pred_reg, output_dict=True),
    #     classification_report(y_test.loc[groups.index], pred_nn, output_dict=True)],
    #     model_names=["Random Forest", "Logistic Regression", "Neural Network"],
    #     label="Men" if name == 1 else "Women"
    # )


for model_name, results in results_sex.items():
    print(f"\nResults for {model_name}:")
    compare_fairness(results)
    
# %%
# Prediction per ethnicity

results_ethicities = {
    "rf": {},
    "reg": {},
    "nn": {}
}

for ethnicity in ethnicities:
    group = X_test.groupby("race_"+ethnicity)
    for name, groups in group:
        if name == True:
            pred_rf = rf.predict(groups)
            pred_reg = regressor.predict(groups)
            pred_nn = predict(neural_network, prepare_data(groups))
            
            results_ethicities["rf"][ethnicity] = classification_report(y_test.loc[groups.index], pred_rf, output_dict=True)
            results_ethicities["reg"][ethnicity] = classification_report(y_test.loc[groups.index], pred_reg, output_dict=True)
            results_ethicities["nn"][ethnicity] = classification_report(y_test.loc[groups.index], pred_nn, output_dict=True)
            
            print("\n", ethnicity, groups.shape[0])
            # print(false_negative_rate(y_test.loc[groups.index], pred_rf))
            
            # compare(
            #     [classification_report(y_test.loc[groups.index], pred_rf, output_dict=True),
            #     classification_report(y_test.loc[groups.index], pred_reg, output_dict=True),
            #     classification_report(y_test.loc[groups.index], pred_nn, output_dict=True)],
            #     model_names=["Random Forest", "Logistic Regression", "Neural Network"],
            #     label = ethnicity
            # )

for model_name, results in results_ethicities.items():
    print(f"\nResults for {model_name}:")
    compare_fairness(results)
# %%
# Prediction per region
results_regions = {
    "rf": {},
    "reg": {},
    "nn": {}
}

regions = ["FW","GL","MS","MW","Mt","NE","NG","NW","PO","SC","SE"]
for region in regions:
    group = X_test.groupby("region_first_"+region)
    for name, groups in group:
        if name == True:
            pred = rf.predict(groups)
            pred_reg = regressor.predict(groups)
            pred_nn = predict(neural_network, prepare_data(groups))
            
            results_regions["rf"][region] = classification_report(y_test.loc[groups.index], pred, output_dict=True)
            results_regions["reg"][region] = classification_report(y_test.loc[groups.index], pred_reg, output_dict=True)
            results_regions["nn"][region] = classification_report(y_test.loc[groups.index], pred_nn, output_dict=True)
            
            print("\n", region, groups.shape[0])
            
            # compare(
            #     [classification_report(y_test.loc[groups.index], pred, output_dict=True),
            #     classification_report(y_test.loc[groups.index], pred_reg, output_dict=True),
            #     classification_report(y_test.loc[groups.index], pred_nn, output_dict=True)],
            #     model_names=["Random Forest", "Logistic Regression", "Neural NetworkS"],
            #     label = region
            # )
            
for model_name, results in results_regions.items():
    print(f"\nResults for {model_name}:")
    compare_fairness(results)
            
# %%
#Grid search cv
# parameters = {"n_estimators": [100, 250, 500, 1000, 2000], "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
# rf_model = RandomForestClassifier(random_state=1)
# grid_search = GridSearchCV(rf_model, parameters, cv=5, n_jobs=-1, return_train_score=True, verbose=3)
# grid_search.fit(X_train_data, y_train_data)

#%%
# demographic parity 

print("Demographic Parity Metrics:  \nbased on ethinicity")
print(demographic_parity_difference(y_test, predictions["rf"], sensitive_features=sf_test))
print(demographic_parity_ratio(y_test, predictions["reg"], sensitive_features=sf_test))
# %%
# equalized odds
print("Equalized Odds Metrics:  \nbased on ethinicity")
print(equalized_odds_difference(y_test, predictions["reg"], sensitive_features=sf_data["race"]))
print(equalized_odds_ratio(y_test, predictions["reg"], sensitive_features=sf_data["race"]))

print("\nbased on sex")
print(equalized_odds_difference(y_test, predictions["reg"], sensitive_features=sf_data["sex"]))
print(equalized_odds_ratio(y_test, predictions["reg"], sensitive_features=sf_data["sex"]))
# %%
# false negative rate
ethnicities = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other", "Puertorican", "White"]

print("False Negative Rate Metrics:  \nbased on ethnicity")

for key, pred in predictions.items():
    print(f"\n{key} model:")
    for ethnicity in ethnicities:
        group_idx = sf_data[sf_data["race"] == ethnicity].index
        if len(group_idx) > 0:
            fnr = false_negative_rate(y_test.loc[group_idx], pd.Series(pred, index=y_test.index).loc[group_idx])
            print(f"{ethnicity}: {fnr}")
#%%
# --- CorrelationRemover for fairness on 'race' ---
def cr():
    race_columns = [col for col in X_train.columns if col.startswith('race_')]

    corr_remover = CorrelationRemover(sensitive_feature_ids=race_columns)

    dcr = corr_remover.fit_transform(X_train)
    dcr = pd.DataFrame(dcr, columns=X_train.columns.drop(race_columns))
    dcr[race_columns] = X_train[race_columns].reset_index(drop=True)

    dcr_test = corr_remover.fit_transform(X_test)
    dcr_test = pd.DataFrame(dcr_test, columns=X_test.columns.drop(race_columns))
    dcr_test[race_columns] = X_test[race_columns].reset_index(drop=True         )

    # Train a new logistic regression on the transformed data
    regressor_corr = LogisticRegression(max_iter=1000, random_state=42)
    regressor_corr.fit(dcr, y_train)

    # Predict and evaluate
    pred_corr = regressor_corr.predict(dcr_test)
    print("\nClassification report for Logistic Regression with CorrelationRemover:")
    print(classification_report(y_test, pred_corr))

    # Fairness metrics for the new model
    print("\nDemographic Parity Metrics (CorrelationRemover):  based on ethnicity")
    sf_data_corr = X_train_data[X_train_data['idx'].isin(X_test['idx'])]
    print(demographic_parity_difference(y_test, pred_corr, sensitive_features=sf_data_corr["race"]))
    print(demographic_parity_ratio(y_test, pred_corr, sensitive_features=sf_data_corr["race"]))

    print("\nDemographic Parity Metrics (CorrelationRemover):  based on sex")
    print(demographic_parity_difference(y_test, pred_corr, sensitive_features=sf_data_corr["sex"]))
    print(demographic_parity_ratio(y_test, pred_corr, sensitive_features=sf_data_corr["sex"]))

    print("\nEqualized Odds Metrics (CorrelationRemover):  based on ethnicity")
    print(equalized_odds_difference(y_test, pred_corr, sensitive_features=sf_data_corr["race"]))
    print(equalized_odds_ratio(y_test, pred_corr, sensitive_features=sf_data_corr["race"]))

    print("\nEqualized Odds Metrics (CorrelationRemover):  based on sex")
    print(equalized_odds_difference(y_test, pred_corr, sensitive_features=sf_data_corr["sex"]))
    print(equalized_odds_ratio(y_test, pred_corr, sensitive_features=sf_data_corr["sex"]))

    compare(
        [classification_report(y_test, pred_corr, output_dict=True)],
        model_names=["Logistic Regression with CorrelationRemover"],
        label="Logistic Regression with CorrelationRemover"
    )
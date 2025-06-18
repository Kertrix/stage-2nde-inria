#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

X_train_data = pd.read_csv("law_data.csv")
y_train_data = X_train_data.pop("first_pf")

X_train_data = pd.get_dummies(data=X_train_data)
X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

regressor = LogisticRegression(max_iter=1000, random_state=42)
regressor.fit(X_train, y_train)

#%%
def compare(rf, reg, print_output=True, label=None):
    print(label)
    
    if print_output:
        print("Random Forest \n", pd.DataFrame(rf).T.to_markdown(), "\n")
        print("Logistic Regression \n", pd.DataFrame(reg).T.to_markdown(), "\n")
    
    metrics = ["precision", "recall", "f1-score", "accuracy"]
    
    result_rf_failed = [rf["0.0"][k] for k in metrics[:-1]] + [rf["accuracy"]]
    result_log_failed = [reg["0.0"][k] for k in metrics[:-1]] + [reg["accuracy"]]
    
    result_rf_succeeded = [reg["1.0"][k] for k in metrics[:-1]] + [reg["accuracy"]]
    result_log_succeeded = [rf["1.0"][k] for k in metrics[:-1]] + [rf["accuracy"]]
    
    width = 0.35
    x = np.arange(len(metrics))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    fig.suptitle(label, fontsize=25)
    
    ax1.bar(x-0.2, result_rf_failed, width)
    ax1.bar(x+0.2, result_log_failed, width)
    
    ax1.set_ylabel("Score")
    ax1.set_title("Comparison of Random Forest and Logistic Regression for failed")
    ax1.set_xticks(x, metrics)
    ax1.set_xticklabels(metrics)
    ax1.legend(["Random Forest", "Logistic Regression"])
    
    ax2.bar(x-0.2, result_rf_succeeded, width)
    ax2.bar(x+0.2, result_log_succeeded, width)
    
    ax2.set_ylabel("Score")
    ax2.set_title("Comparison of Random Forest and Logistic Regression for suceeded")
    ax2.set_xticks(x, metrics)
    ax2.set_xticklabels(metrics)
    ax2.legend(["Random Forest", "Logistic Regression"])
#%%
predictions = model.predict(X_test)
predictions_reg = regressor.predict(X_test)
compare(classification_report(y_test, predictions, output_dict=True),
        classification_report(y_test, predictions_reg, output_dict=True))
    
# %%
# Prediction per sex
sex = X_test.groupby("sex")
for name, groups in sex:
    pred_rf = model.predict(groups)
    pred_reg = regressor.predict(groups)
    print("\n", name, groups.shape[0])
    compare(
        classification_report(y_test.loc[groups.index], pred_rf, output_dict=True),
        classification_report(y_test.loc[groups.index], pred_reg, output_dict=True),
        label="Men" if name == 1 else "Women"
    )

# %%
# Prediction per ethnicity
ethnicities = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other", "Puertorican", "White"]
# print(X_test)
for ethnicity in ethnicities:
    group = X_test.groupby("race_"+ethnicity)
    for name, groups in group:
        if name == True:
            pred_rf = model.predict(groups)
            pred_reg = regressor.predict(groups)
            print("\n", ethnicity, groups.shape[0])
            
            compare(
                classification_report(y_test.loc[groups.index], pred_rf, output_dict=True),
                classification_report(y_test.loc[groups.index], pred_reg, output_dict=True),
                label = ethnicity
            )

# %%
# Prediction per region
regions = ["FW","GL","MS","MW","Mt","NE","NG","NW","PO","SC","SE"]
for region in regions:
    group = X_test.groupby("region_first_"+region)
    for name, groups in group:
        if name == True:
            pred = model.predict(groups)
            pred_reg = regressor.predict(groups)
            print("\n", region, groups.shape[0])
            
            compare(
                classification_report(y_test.loc[groups.index], pred, output_dict=True),
                classification_report(y_test.loc[groups.index], pred_reg, output_dict=True),
                label = region
            )
            
# %%
#Grid search cv
parameters = {"n_estimators": [100, 250, 500, 1000, 2000], "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
rf_model = RandomForestClassifier(random_state=1)
grid_search = GridSearchCV(rf_model, parameters, cv=5, n_jobs=-1, return_train_score=True, verbose=3)
grid_search.fit(X_train_data, y_train_data)
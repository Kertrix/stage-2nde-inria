import torch
from sklearn.metrics import classification_report

from fairness.corr_remover import cr
from fairness.threshold_opti import threshold_opti
from neural_net.preprocessing import get_train_test_data
from utils.fairness_metrics import FairnessMetrics
from utils.graphs import compare
from utils.neural_utils import NeuralNetwork, predict
from utils.pipelines import Pipelines


def compare_fairness(preds, preds_fair, y_true, sf_data, method):
    fairness_before = FairnessMetrics(y_true=y_true, y_pred=preds, sf_data=sf_data)
    print(f"Fairness metrics before {method} fairness:")
    
    fairness_before.demographic_parity()
    
    print("\n")
    
    fairness_before.equalized_odds()
    
    print("\n" + "="*50 + "\n")
    
    fairness_after = FairnessMetrics(y_true=y_true, y_pred=preds_fair, sf_data=sf_data)
    print(f"Fairness metrics after {method} fairness:")
    fairness_after.demographic_parity()
    
    print("\n")
    
    fairness_after.equalized_odds()
    
    # Compare the performance of the two models
    compare(
        [
            classification_report(y_true, preds, output_dict=True),
            classification_report(y_true, preds_fair, output_dict=True),
        ],
        model_names=["Without Fairness", "With Fairness"],
        label="Fairness Comparison"
    )

def inprocessing():
    """
    Compares the difference of fairness when using in-processing fairness methods.
    """
    
    X_train, X_test, y_train, y_test, sf_train, sf_test = get_train_test_data("law_data.csv", "first_pf")
    
    model = NeuralNetwork(input_size=X_train.shape[1])
    model.load_state_dict(torch.load('neural_net/model.pth', map_location=torch.device('cpu')))
    
    model_fair = NeuralNetwork(input_size=X_train.shape[1])
    model_fair.load_state_dict(torch.load('neural_net/model_with_fairness.pth', map_location=torch.device('cpu')))
    
    preds = predict(model, X_test)
    preds_fair = predict(model_fair, X_test)
    
    compare_fairness(preds=preds, preds_fair=preds_fair, y_true=y_test, sf_data=sf_test, method="In-processing")
    
def preprocessing():
    X_train, X_test, y_train, y_test, sf_train, sf_test = get_train_test_data("law_data.csv", "first_pf")

    preds, preds_corr = cr()
    
    compare_fairness(preds=preds, preds_fair=preds_corr, y_true=y_test, sf_data=sf_test, method="Pre-processing")

def postprocessing():
    X_train, X_test, y_train, y_test, sf_train, sf_test = get_train_test_data("law_data.csv", "first_pf", encode=False)
    
    model = Pipelines(X_train.columns).regressor()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    preds_fair = threshold_opti(X_train, X_test, y_train, sf_train, sf_test)    
    
    compare_fairness(preds=preds, preds_fair=preds_fair, y_true=y_test, sf_data=sf_test, method="Post-processing")


if __name__ == "__main__":
    postprocessing()

    

    
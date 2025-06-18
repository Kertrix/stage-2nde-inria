import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def compare(results, model_names=None, print_output=True, label=None):
    """
    results: list of dicts (classification reports as dicts)
    model_names: list of str, names for each model
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(results))]
    if print_output:
        for name, res in zip(model_names, results):
            print(f"{name}\n", pd.DataFrame(res).T.to_markdown(), "\n")
    
    metrics = ["precision", "recall", "f1-score", "accuracy"]
    classes = ["0.0", "1.0"]
    print(classes)
    
    width = 0.8 / len(results)
    x = np.arange(len(metrics))
    
    fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 6))
    if len(classes) == 1:
        axes = [axes]
    fig.suptitle(label, fontsize=25)
    
    for idx, cls in enumerate(classes):
        ax = axes[idx]
        for i, (res, name) in enumerate(zip(results, model_names)):
            values = [res[cls][k] for k in metrics[:-1]] + [res["accuracy"]]
            ax.bar(x + (i - len(results)/2) * width + width/2, values, width, label=name)
        ax.set_ylabel("Score")
        ax.set_title(f"Comparison for class {cls}")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
    plt.tight_layout()
    plt.show()
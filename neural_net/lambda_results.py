#%%
import numpy as np
from matplotlib import pyplot as plt

from neural_net.results import *

#%%
λ = np.linspace(1e-3, 1, 10)

accuracies_mean = np.reshape(accuracies_all, ((10, 20))).mean(axis=1)
accuracies = [accuracies_all[k * 20] for k in range(10)]

# Average accuracies of each epoch for each λ
plt.bar(list(map(str, list(λ))), accuracies_mean)
plt.ylim(min(accuracies_mean) - 0.001, max(accuracies_mean) + 0.001)  # Zoom in on the top of the bars
plt.ylabel("Accuracy")
plt.xlabel("λ")
plt.title("Accuracy vs λ (zoomed on top)")
plt.show()

#%%
# Last accuracy of each λ
plt.bar(list(map(str, list(λ))), accuracies)
plt.ylim(min(accuracies) - 0.001, max(accuracies) + 0.001)  # Zoom in on the top of the bars
plt.ylabel("Accuracy")
plt.xlabel("λ")
plt.title("Accuracy vs λ (zoomed on top)")
plt.show()

#%%
# We try to achieve the highest accuracy while minimizing the demographic parity difference
dems = np.reshape(dems_all, ((10, 20))).mean(axis=1)
plt.bar(list(map(str, list(λ))), accuracies_mean / dems)
# plt.ylim(min(dems) - 0.001, max(dems) + 0.001)  # Zoom in on the top of the bars
plt.ylabel("Demographic Parity Difference")
plt.xlabel("λ")
plt.title("Demographic Parity vs λ (zoomed on top)")
plt.show()
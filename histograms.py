import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from data import *

fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    n, bins, patches = axes[i].hist(D_male[param], bins=30, edgecolor='black', density=True)
    axes[i].set_title(f'Histogram of {param} for male abalones')

    x = np.linspace(D_male[param].min(), D_male[param].max(), 100)
    y = stats.norm.pdf(x, np.mean(D_male[param]), np.std(D_male[param]))
    axes[i].plot(x, y, 'r')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    n, bins, patches = axes[i].hist(D_female[param], bins=30, edgecolor='black', density=True)
    axes[i].set_title(f'Histogram of {param} for female abalones')

    x = np.linspace(D_female[param].min(), D_female[param].max(), 100)
    y = stats.norm.pdf(x, np.mean(D_female[param]), np.std(D_female[param]))
    axes[i].plot(x, y, 'r')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    n, bins, patches = axes[i].hist(D_infant[param], bins=30, edgecolor='black', density=True)
    axes[i].set_title(f'Histogram of {param} for infant abalones')

    x = np.linspace(D_infant[param].min(), D_infant[param].max(), 100)
    y = stats.norm.pdf(x, np.mean(D_infant[param]), np.std(D_infant[param]))
    axes[i].plot(x, y, 'r')

plt.tight_layout()
plt.show()
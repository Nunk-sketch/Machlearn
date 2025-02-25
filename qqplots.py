import matplotlib.pyplot as plt
import statsmodels.api as sm
from data import *

# QQ plots for each parameter for abalone males
parameters = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    sm.qqplot(D_male[param], line='s', ax=axes[i])
    axes[i].set_title(f'QQ plot of {param} for male abalones')

plt.tight_layout()
plt.savefig('qqplot_male_abalones.png')
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    sm.qqplot(D_female[param], line='s', ax=axes[i])
    axes[i].set_title(f'QQ plot of {param} for female abalones')

plt.tight_layout()
plt.savefig('qqplot_female_abalones.png')
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    sm.qqplot(D_infant[param], line='s', ax=axes[i])
    axes[i].set_title(f'QQ plot of {param} for infant abalones')

plt.tight_layout()
plt.savefig('qqplot_infant_abalones.png')
plt.show()


fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    sm.qqplot(D_clean[param], line='s', ax=axes[i])
    axes[i].set_title(f'QQ plot of {param} for all abalones')

plt.tight_layout()
plt.savefig('qqplot_abalones.png')
plt.show()


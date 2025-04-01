import matplotlib.pyplot as plt
from data import *

fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    axes[i].boxplot([D_male[param], D_female[param], D_infant[param]], labels=['Male', 'Female', 'Infant'], vert = False)
    axes[i].set_title(f'Abalones - {param}')

plt.tight_layout()
plt.savefig('boxplot.png')
plt.show()
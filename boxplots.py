import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';')
# D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';') der hvor i har filen
# D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';') der hvor i har filen

Male = D["Sex"]== "M"
Female = D["Sex"] == "F"
Infant = D["Sex"] == "I"

# "Sex";"Length";"Diameter";"Height";"Whole weight";"Shucked weight";"Viscera weight";"Shell weight";"Rings"
# parameters

# Assign values based on sex
D_male = D[Male]
D_female = D[Female]
D_infant = D[Infant]

#Stats about abalones
maleinfo = D_male.describe()
femaleinfo = D_female.describe()
infantinfo = D_infant.describe()
datainfo = D.describe()

parameters = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
# Boxplots
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, param in enumerate(parameters):
    axes[i].boxplot([D_male[param], D_female[param], D_infant[param]], labels=['Male', 'Female', 'Infant'])
    axes[i].set_title(f'Abalones - {param}')

plt.tight_layout()
plt.show()
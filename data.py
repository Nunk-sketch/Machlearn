import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';')
D = pd.read_csv('abalone/abalone.data', sep=';')
# D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';') der hvor i har filen

N = len(D)

Male = D["Sex"]== "M"
Female = D["Sex"] == "F"
Infant = D["Sex"] == "I"

# "Sex";"Length";"Diameter";"Height";"Whole weight";"Shucked weight";"Viscera weight";"Shell weight";"Rings"
# parameters

# Assign values based on sex
D_male = D[Male]
D_female = D[Female]
D_infant = D[Infant]

D_clean = D.drop(columns=["Sex"])

#Stats about abalones
maleinfo = D_male.describe()
femaleinfo = D_female.describe()
infantinfo = D_infant.describe()
datainfo = D.describe()

parameters = D.columns[1:]
C = len(D.columns) - 1 # amount of attributes (8)
classDict = dict(zip(parameters, range(C))) # dictionary of the parameters and their indices
print(classDict)

y = np.asarray([classDict[param] for param in parameters]) # indices of the parameters
print(y)

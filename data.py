import pandas as pd

#D = pd.read_csv('MachineLearningOgDataMining/Projekt/Machlearn/abalone/abalone.data', sep=';')
D = pd.read_csv('abalone/abalone.data', sep=';')
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
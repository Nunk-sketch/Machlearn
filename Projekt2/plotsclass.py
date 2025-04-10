import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from abaclass import *
import json
from sklearn.dummy import DummyClassifier


# 1. Indlæs data fra JSON-filen
with open("crossvalidation_results2.json", "r") as f:
    data = json.load(f)

results = data["results"]
models = list(results.keys())
best_model = data["best_model"]["name"] if isinstance(data["best_model"], dict) else data["best_model"]
best_params = data["best_params"] if isinstance(data["best_params"], dict) else {}


# 2. Udregn middel og std
mean_errors = [np.mean(results[model]) for model in models]
std_errors = [np.std(results[model]) for model in models]

# 3. Barplot med error ± std
plt.figure(figsize=(10, 6))
plt.bar(models, mean_errors, yerr=std_errors, capsize=5, color='cornflowerblue')
plt.title("Mean error pr. model (± std)", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.ylim(0, max(mean_errors) + 0.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("model_errors_barplot.png")
# plt.show()  # Removed due to non-interactive backend

# 4. Boxplot for fejlfordeling
plt.figure(figsize=(10, 6))
plt.boxplot([results[model] for model in models], tick_labels=models, patch_artist=True,
            boxprops=dict(facecolor='lightgreen', color='green'),
            medianprops=dict(color='black'))
plt.title("Error distribution over 10 folds", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("model_errors_boxplot.png")
# plt.show()  # Commented out because the 'Agg' backend is non-interactive


plt.figure(figsize=(8, 5))
for model in ["CT", "MN", "BC"]:
    plt.plot(results[model], label=model)
plt.title("Error over folds")
plt.xlabel("Fold")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("error_over_folds.png")


# Instantiate the best model using its name and parameters
model_mapping = {
    "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha = 0.001),
    "CT": DecisionTreeClassifier(max_depth = 4),
    "KNN": KNeighborsClassifier(n_neighbors=11),
    "MN": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=10),
    "BC": DummyClassifier(strategy =  "most_frequent"),
    "NB": MultinomialNB(alpha=0.1)
}
if best_model in model_mapping:
    best_model_instance = model_mapping[best_model]
else:
    raise KeyError(f"Model '{best_model}' not found in model_mapping.")

# Fit the model and generate predictions
best_model_instance.fit(x_clas, y_clas)
y_pred = best_model_instance.predict(x_clas)

ConfusionMatrixDisplay.from_predictions(y_clas, y_pred, display_labels=["Child", "Adult", "Old"])
plt.title("Confusion Matrix for MN")
plt.savefig("confusion_matrix.png")


############################################


# 1. Indlæs data fra JSON-filen
with open("crossvalidation_resultsNY.json", "r") as f:
    data = json.load(f)

results = data["results"]
models = list(results.keys())
best_model = data["best_model"]["name"] if isinstance(data["best_model"], dict) else data["best_model"]
best_params = data["best_params"] if isinstance(data["best_params"], dict) else {}


# 2. Udregn middel og std
mean_errors = [np.mean(results[model]) for model in models]
std_errors = [np.std(results[model]) for model in models]

# 3. Barplot med error ± std
plt.figure(figsize=(10, 6))
plt.bar(models, mean_errors, yerr=std_errors, capsize=5, color='cornflowerblue')
plt.title("Mean error pr. model (± std) NY", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.ylim(0, max(mean_errors) + 0.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("model_errors_barplot_NY.png")
# plt.show()  # Removed due to non-interactive backend

# 4. Boxplot for fejlfordeling
plt.figure(figsize=(10, 6))
plt.boxplot([results[model] for model in models], tick_labels=models, patch_artist=True,
            boxprops=dict(facecolor='lightgreen', color='green'),
            medianprops=dict(color='black'))
plt.title("Error distribution over 10 folds NY", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("model_errors_boxplot_NY.png")
# plt.show()  # Commented out because the 'Agg' backend is non-interactive


plt.figure(figsize=(8, 5))
for model in ["CT", "MN", "BC"]:
    plt.plot(results[model], label=model)
plt.title("Error over folds NY")
plt.xlabel("Fold")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("error_over_folds_NY.png")


# Instantiate the best model using its name and parameters
model_mapping = {
    "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha = 0.001),
    "CT": DecisionTreeClassifier(max_depth = 4),
    "KNN": KNeighborsClassifier(n_neighbors=11),
    "MN": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=10),
    "BC": DummyClassifier(strategy =  "most_frequent"),
    "NB": MultinomialNB(alpha=0.1)
}
if best_model in model_mapping:
    best_model_instance = model_mapping[best_model]
else:
    raise KeyError(f"Model '{best_model}' not found in model_mapping.")

# Fit the model and generate predictions
best_model_instance.fit(x_clas, y_clas)
y_pred = best_model_instance.predict(x_clas)

ConfusionMatrixDisplay.from_predictions(y_clas, y_pred, display_labels=["Child", "Adult", "Old"])
plt.title("Confusion Matrix for MN NY")
plt.savefig("confusion_matrix_NY.png")

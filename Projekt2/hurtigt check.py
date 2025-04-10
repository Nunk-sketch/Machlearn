import json
import scipy.stats as stats

# Load data from JSON file
with open('crossvalidation_results2.json', 'r') as file:
    data = json.load(file)


ct_scores = data['results']['CT']
mn_scores = data['results']['MN']
bc_scores = data['results']['BC']


def perform_ttest(group1, group2, label1, label2):
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    print(f"{label1} vs {label2} - T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        print(f"There is a significant difference between {label1} and {label2}.")
    else:
        print(f"There is no significant difference between {label1} and {label2}.")


model_names = list(data['results'].keys())

# Perform pairwise comparisons for all models
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model1 = model_names[i]
        model2 = model_names[j]
        scores1 = data['results'][model1]
        scores2 = data['results'][model2]
        perform_ttest(scores1, scores2, model1, model2)

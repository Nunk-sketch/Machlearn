import numpy as np
import scipy.stats as stats
import json
from scipy.stats import ttest_ind
from abaclass import *

# Load the JSON file
with open("crossvalidation_results.json", "r") as file:
    data = json.load(file)

    # Extract errors for each model
    errors = {model: np.array(values) for model, values in data["results"].items()}

    # Perform data analysis
    analysis_results = {}
    for model, error_values in errors.items():
        confidence_level = 0.95
        degrees_freedom = len(error_values) - 1
        sample_mean = np.mean(error_values)
        sample_standard_error = stats.sem(error_values)
        confidence_interval = stats.t.interval(
            confidence_level, degrees_freedom, loc=sample_mean, scale=sample_standard_error
        )
        
        analysis_results[model] = {
            "mean_error": sample_mean,
            "std_error": np.std(error_values),
            "min_error": np.min(error_values),
            "max_error": np.max(error_values),
            "confidence_interval": confidence_interval,
        }

    # Compare models using t-tests with Bonferroni correction
    comparison_results = {}
    models = list(errors.keys())
    num_comparisons = len(models) * (len(models) - 1) // 2  # Total pairwise comparisons
    bonferroni_alpha = 0.05 / num_comparisons  # Adjusted alpha for Bonferroni correction

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            t_stat, p_value = ttest_ind(errors[model1], errors[model2], equal_var=False)
            adjusted_p_value = min(p_value * num_comparisons, 1.0)  # Bonferroni adjustment
            comparison_results[f"{model1} vs {model2}"] = {
                "t_stat": t_stat,
                "p_value": p_value,
                "adjusted_p_value": adjusted_p_value,  # Ensure p-value does not exceed 1
                "confidence_interval": (
                    (np.mean(errors[model1]) - np.mean(errors[model2])) - 
                    stats.t.ppf(0.975, df=(np.var(errors[model1], ddof=1) / len(errors[model1]) + 
                                           np.var(errors[model2], ddof=1) / len(errors[model2]))**2 /
                                ((np.var(errors[model1], ddof=1) / len(errors[model1]))**2 / 
                                 (len(errors[model1]) - 1) + 
                                 (np.var(errors[model2], ddof=1) / len(errors[model2]))**2 / 
                                 (len(errors[model2]) - 1))) * 
                    np.sqrt(np.var(errors[model1], ddof=1) / len(errors[model1]) + 
                            np.var(errors[model2], ddof=1) / len(errors[model2])),
                    (np.mean(errors[model1]) - np.mean(errors[model2])) + 
                    stats.t.ppf(0.975, df=(np.var(errors[model1], ddof=1) / len(errors[model1]) + 
                                           np.var(errors[model2], ddof=1) / len(errors[model2]))**2 /
                                ((np.var(errors[model1], ddof=1) / len(errors[model1]))**2 / 
                                 (len(errors[model1]) - 1) + 
                                 (np.var(errors[model2], ddof=1) / len(errors[model2]))**2 / 
                                 (len(errors[model2]) - 1))) * 
                    np.sqrt(np.var(errors[model1], ddof=1) / len(errors[model1]) + 
                            np.var(errors[model2], ddof=1) / len(errors[model2]))
                )
            }

    # Print analysis results
    print("Analysis Results:")
    for model, stats in analysis_results.items():
        print(f"{model}: {stats}")

    # Print comparison results
    print("\nComparison Results:")
    for comparison, stats in comparison_results.items():
        print(f"{comparison}: {stats}")

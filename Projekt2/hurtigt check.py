import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
# Load the data directly from the provided dictionary
data = {
    "CT": [
        0.1578947368421053,
        0.11004784688995217,
        0.12440191387559807,
        0.12440191387559807,
        0.09090909090909094,
        0.11961722488038273,
        0.13397129186602874,
        0.1435406698564593,
        0.09134615384615385,
        0.1298076923076923
    ],
    "MN": [
        0.14832535885167464,
        0.09569377990430628,
        0.12440191387559807,
        0.1291866028708134,
        0.09569377990430628,
        0.1291866028708134,
        0.14832535885167464,
        0.14832535885167464,
        0.09615384615384615,
        0.13942307692307687
    ]
}

# Extract the relevant columns for CT and MN
ct_scores = data['CT']
mn_scores = data['MN']
# Perform a Welch's t-test to check for significant difference
t_stat, p_value = stats.ttest_ind(ct_scores, mn_scores, equal_var=False)

# Print the results
print(f"T-statistic: {t_stat}, P-value: {p_value}")
if p_value < 0.05:
    print("There is a significant difference between CT and MN.")
else:
    print("There is no significant difference between CT and MN.")
    # Base case (BC) scores
    bc_scores = [
        0.1722488038277512,
            0.1866028708133971,
            0.16507177033492826,
            0.14593301435406703,
            0.22248803827751196,
            0.1578947368421053,
            0.18181818181818177,
            0.17985611510791366,
            0.14148681055155876,
            0.14388489208633093
    ]

    # Perform Welch's t-test for CT vs BC
    t_stat_ct, p_value_ct = stats.ttest_ind(ct_scores, bc_scores, equal_var=False)
    print(f"CT vs BC - T-statistic: {t_stat_ct}, P-value: {p_value_ct}")
    if p_value_ct < 0.05:
        print("CT is significantly different from BC.")
    else:
        print("CT is not significantly different from BC.")

    # Perform Welch's t-test for MN vs BC
    t_stat_mn, p_value_mn = stats.ttest_ind(mn_scores, bc_scores, equal_var=False)
    print(f"MN vs BC - T-statistic: {t_stat_mn}, P-value: {p_value_mn}")
    if p_value_mn < 0.05:
        print("MN is significantly different from BC.")
    else:
        print("MN is not significantly different from BC.")
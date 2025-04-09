from scipy.stats import ttest_rel, t
import numpy as np

# Per-fold MSEs for each model
mse_ann = np.array([2.191, 3.066, 3.198, 2.688, 2.403, 2.942, 2.041, 2.040, 2.187, 2.351])
mse_rlr = np.array([2.936, 3.716, 3.654, 3.222, 2.700, 3.329, 2.714, 2.657, 2.784, 3.358])
mse_base = np.array([13.478, 13.943, 12.727, 9.297, 10.167, 11.766, 11.316, 9.789, 11.216, 10.630])


def paired_t_test_and_ci(model1, model2, name1, name2):
    diffs = model1 - model2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    n = len(diffs)
    conf_int = t.interval(0.95, df=n-1, loc=mean_diff, scale=std_diff / np.sqrt(n))
    t_stat, p_val = ttest_rel(model1, model2)

    print(f"\n--- {name1} vs {name2} ---")
    print(f"Mean difference ({name1} - {name2}): {mean_diff:.4f}")
    print(f"95% confidence interval: {conf_int}")
    print(f"Paired t-test p-value: {p_val:.4f}")

# Run comparisons
paired_t_test_and_ci(mse_ann, mse_rlr, "ANN", "RLR")
paired_t_test_and_ci(mse_ann, mse_base, "ANN", "BASE")
paired_t_test_and_ci(mse_rlr, mse_base, "RLR", "BASE")


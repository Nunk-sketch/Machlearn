from scipy.stats import ttest_rel, t
import numpy as np

# Per-fold MSEs for each model
mse_ann = np.array([0.411, 0.446, 0.404, 0.387, 0.509, 0.418, 0.520, 0.417, 0.311, 0.368])
mse_rlr = np.array([0.460, 0.497, 0.436, 0.438, 0.562, 0.484, 0.636, 0.443, 0.360, 0.430])
mse_base = np.array([1.082, 1.001, 0.853, 0.883, 1.212, 1.025, 1.208, 1.091, 0.808, 0.842])



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


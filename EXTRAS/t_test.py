from scipy.stats import ttest_1samp

data = [10, 12, 14, 16, 18]
t_stat, p_value = ttest_1samp(data, popmean=15)  # Test mean=15
print("t-statistic:", t_stat, "p-value:", p_value)
#completed
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
n, p = 10, 0.5  # Trials, Probability of success
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)
plt.bar(x, pmf)
plt.title("Binomial Distribution")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, loc=0, scale=1)  
cdf = norm.cdf(x, loc=0, scale=1)  

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, pdf, label="PDF", color="blue")
plt.plot(x, cdf, label="CDF", color="orange")
plt.title("PDF and CDF of a Standard Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density / Cumulative Probability")
plt.legend()
plt.grid()
plt.show()

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-4, 4, 100)
pdf = norm.pdf(x, loc=0, scale=1)  # Mean=0, Std Dev=1
plt.plot(x, pdf, label="PDF")
plt.title("Normal Distribution")
plt.show()

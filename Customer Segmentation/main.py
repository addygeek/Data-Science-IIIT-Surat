import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset (replace 'customers.csv' with your dataset)
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Annual Income (k$)": [15, 16, 17, 20, 22, 24, 25, 29, 35, 40],
    "Spending Score (1-100)": [39, 81, 6, 77, 40, 76, 94, 3, 72, 14]
}
df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# Feature Selection
X = df.iloc[:, 1:].values

# Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add Cluster Labels to Dataset
df['Cluster'] = y_kmeans

print("\nClustered Dataset:\n", df)

# Visualize Clusters using PCA
pca = PCA(2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', s=100, alpha=0.7)
plt.title("Customer Segments", fontsize=16)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster")
plt.show()

# Save the results
df.to_csv("customer_segments.csv", index=False)
print("\nClustered data saved to 'customer_segments.csv'.")

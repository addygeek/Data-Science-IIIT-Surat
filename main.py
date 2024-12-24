import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load Dataset
def load_data(file_path):
    """
    Load the dataset from the given CSV file path.
    """
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")
    return data

# 2. Preprocessing
def preprocess_data(data):
    """
    Preprocess the data by handling missing values, scaling, and selecting features for clustering.
    """
    # Handle missing values by filling with the mean (you can change the strategy)
    data.fillna(data.mean(), inplace=True)
    
    # Select features relevant for clustering (excluding non-numerical columns like 'CustomerID' or 'Name')
    features = data.drop(columns=['CustomerID', 'Name'], errors='ignore')
    
    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("Data preprocessing complete!")
    return scaled_features

# 3. Finding the Optimal Number of Clusters using Elbow Method
def find_optimal_clusters(X):
    """
    Use the Elbow method to find the optimal number of clusters.
    """
    distortions = []
    for k in range(1, 11):  # Trying between 1 and 10 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    # Plot the Elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), distortions, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()

# 4. Clustering with KMeans
def perform_clustering(X, n_clusters=3):
    """
    Perform KMeans clustering on the data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    print(f"Clustering done with {n_clusters} clusters!")
    return clusters, kmeans

# 5. Visualize Clusters (2D or 3D)
def visualize_clusters(X, clusters):
    """
    Visualize the clusters in a 2D plot.
    """
    df_clusters = pd.DataFrame(X, columns=['Feature1', 'Feature2'])  # Adjust according to features
    df_clusters['Cluster'] = clusters
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Feature1', y='Feature2', data=df_clusters, hue='Cluster', palette='viridis')
    plt.title('Customer Segmentation (2D)')
    plt.show()

# 6. Evaluate Clustering Performance (Silhouette Score)
def evaluate_clustering(X, clusters):
    """
    Evaluate the quality of the clustering using silhouette score.
    """
    score = silhouette_score(X, clusters)
    print(f'Silhouette Score: {score:.3f}')
    
# 7. Main Function to Execute the Project
def main():
    # Load the dataset
    file_path = 'customer_data.csv'  # Replace with the path to your dataset
    data = load_data(file_path)
    
    # Preprocess the data
    X = preprocess_data(data)
    
    # Find the optimal number of clusters using the Elbow method
    find_optimal_clusters(X)
    
    # Perform clustering with the chosen number of clusters (e.g., 3 clusters)
    clusters, kmeans_model = perform_clustering(X, n_clusters=3)
    
    # Visualize the clusters
    visualize_clusters(X, clusters)
    
    # Evaluate clustering performance
    evaluate_clustering(X, clusters)
    
    # Optionally, you can add the cluster labels to your o

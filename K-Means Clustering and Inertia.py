
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set random seed for reproducible results
np.random.seed(42)

# Create two groups of random data
X = np.vstack([
    np.random.randn(50, 2) + np.array([0, 0]),
    np.random.randn(50, 2) + np.array([5, 5])
])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)

# Get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="x",
    color="red",
    s=100,
    label="Cluster Centers"
)

plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Display Inertia (WCSS)
print("Inertia (WCSS):", round(kmeans.inertia_, 3))

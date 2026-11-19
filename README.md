## AIM :






Apply KMeans to a synthetic 2-cluster dataset; compute and visualize inertia. 






## ALGORITHM :





Lloyd’s algorithm: assign to nearest centroid → recompute centroids until convergence. 






KMeans minimizes within-cluster variance; relates to vector quantization. 





## INPUT

import numpy as np

import matplotlib.pyplot as plt

from sklearn. cluster import KMeans

np.random.seed(42)

x = np.vstack([np.random.randn(50,2)+np.array([0,0]),

np.random.randn(50,2)+np.array([5,5])])

kmeans = KMeans (n_clusters=2, random_state=42).fit(x)

labels = kmeans.labels_

centers = kmeans.cluster_centers_

plt.scatter(x[:,0], x[:,1], c=labels, cmap='viridis');

plt.scatter(centers[:,0],centers[:,1], marker='x', color='red', s=100); plt.title('KMeans Clusters');plt.show()

print('Inerita (WCSS):', round(kmeans.inertia_,3))



## OUTPUT:

<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/70985b57-3018-4ced-b5e8-14b6e75835de" />


Inerita (WCSS): 170.203

## RESULT



Students observe cluster structure and effect of centroid positions. Note sensitivity to 
initialization & scaling. 

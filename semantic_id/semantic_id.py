import numpy as np
from sklearn.cluster import KMeans

def generate_semantic_ids(embeddings, codebook_sizes=[32, 32, 32]):
    """
    Converts ad embeddings into discrete token sequences.
    """
    current_residual = embeddings
    semantic_ids = []
    
    for k in codebook_sizes:
        # Cluster the current residuals into 'k' groups
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(current_residual)
        
        # Capture the cluster index (the 'token') for each ad
        tokens = kmeans.labels_
        semantic_ids.append(tokens)
        
        # Calculate residual for the next hierarchy level: E_next = E_current - Centroid
        centroids = kmeans.cluster_centers_
        current_residual = current_residual - centroids[tokens]
        
    # Return as a matrix where each row is an ad's 'word' (e.g., [12, 5, 29])
    return np.stack(semantic_ids, axis=1)

# Example: 1000 ads with 512-dim features
ad_embeddings = np.random.randn(1000, 512)
tokens = generate_semantic_ids(ad_embeddings)

for id in range(10):
    print(f"Ad 0 Semantic ID: {tokens[id]}") # Output looks like: [14 2 31]
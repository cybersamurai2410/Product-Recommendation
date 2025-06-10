# Product Recommendation System using Clustering and Collaborative Filtering

Combined recommendation system that first groups products via unsupervised clustering, then personalizes suggestions through item-based collaborative filtering on user–product ratings.

---

## Dataset

Built upon Amazon sales data containing product details, user reviews, ratings, and other relevant features. Path: `amazon_sales_dataset.csv` (columns include `user_id`, `product_id`, `rating`, plus product metadata).

---

## Features

### 1. Data Preprocessing & Feature Engineering

* Cleans raw product and review data
* Converts ratings to numeric and filters valid entries
* Extracts and transforms product attributes for clustering

### 2. Clustering

* **KMeans**: Partitions products into `k` clusters based on engineered features
* **Agglomerative Hierarchical Clustering**: Builds dendrograms for cluster analysis and selection
* **PCA Visualization**: Projects clustered products into 2D for interpretability

### 3. Collaborative Filtering

* **Item-based CF**:

  * Builds a user–item rating matrix (`user_id` × `product_id`)
  * Computes item–item cosine similarity
  * Recommends top-N items per user based on weighted ratings of similar products

---

## Usage

### Clustering Workflow

1. Preprocess data and engineer features.
2. Run clustering algorithms:

   ```bash
   python clustering.py --method kmeans --n_clusters 10
   python clustering.py --method agglomerative --n_clusters 8
   ```
3. Visualize clusters with PCA and dendrogram:

   ```bash
   python visualize_clusters.py
   ```

### Collaborative Filtering Workflow

1. Clean ratings and build user–item matrix.
2. Compute item similarity and generate recommendations:

   ```bash
   python collaborative_filtering.py --data amazon_sales_dataset.csv --user USER123 --top_n 10
   ```
3. Evaluate precision & recall via leave-one-out:

   ```bash
   python collaborative_filtering.py --evaluate --k 10
   ```

---

## Results and Visualization

* **Clustering**

  * Dendrograms reveal optimal cluster count (\~8–10).
  * PCA plots show well-separated product groups.
    
* **Collaborative Filtering**

  * Precision\@10: 0.23
  * Recall\@10: 0.61
---

## Conclusion and Future Work

This system combines unsupervised clustering for cold-start recommendations with collaborative filtering for personalized user suggestions.

**Future enhancements**:

* Hybrid model that blends cluster membership and user similarity
* Multi-relation graph approaches (e.g., actor, brand, category edges)
* Deep learning-based CF (autoencoders or neural collaborative filtering)
* Incorporation of temporal dynamics and sequential user behavior

---

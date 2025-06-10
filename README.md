# Product Recommendation System using Clustering and Neural Collaborative Filtering
A product recommendation system with unsupervised learning built using clustering techniques to group similar products together and recommend them to users based on their preferences.

## Dataset
The system is built upon Amazon sales data which contains product details, user reviews, ratings, and other relevant features.

## Features
- **Data Preprocessing**: Processes raw product data to extract relevant features for clustering.
- **Feature Engineering**: Transformation and creation of new features to better represent product characteristics and improve clustering performance.
- **Clustering**:
  - KMeans Clustering: Partitions products into 'k' clusters based on their attributes.
  - Agglomerative Hierarchical Clustering: Creates a dendrogram to visualize product groupings and decide on the number of clusters.
- **Recommendation**: For a new product, the system can recommend similar existing products based on its assigned cluster.

## Usage
To get product recommendations for a new product:
1. Input the new product's attributes.
2. The system will determine the appropriate cluster for the product.
3. Based on the product's cluster, similar products will be recommended.

## Results and Visualization
- Dendrogram visualizations help determine the optimal number of clusters for Agglomerative Hierarchical Clustering.
- PCA (Principal Component Analysis) is used to visualize the clusters in a 2D space for both KMeans and Agglomerative Clustering.

## Conclusion and Future Work
This clustering-based recommendation system provides a foundational approach to product recommendations. Future enhancements can include integrating collaborative filtering, using hybrid models, or employing deep learning techniques for more personalized and accurate recommendations.

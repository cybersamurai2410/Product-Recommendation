import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""
Collaborative Filtering (Item-based) on Amazon product ratings.

This script:
1. Loads and cleans the dataset
2. Builds a user–item matrix
3. Computes item–item similarity using cosine similarity
4. Recommends top-N products to a given user based on their rating history
"""

def load_and_clean(path):
    """
    Load dataset and convert ratings to float.
    Filters to only include valid rows with numeric ratings.
    """
    df = pd.read_csv(path)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    return df[['user_id', 'product_id', 'rating']]

def build_user_item_matrix(df):
    """
    Build user-item rating matrix (users as rows, products as columns).
    Missing values are filled with 0 (unrated).
    """
    return df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

def compute_item_similarity(user_item_matrix):
    """
    Compute cosine similarity between items.
    Returns a DataFrame where each row and column is a product_id.
    """
    item_matrix = user_item_matrix.T
    sim_matrix = cosine_similarity(item_matrix)
    return pd.DataFrame(sim_matrix, index=item_matrix.index, columns=item_matrix.index)

def recommend_items(user_id, user_item_matrix, item_sim_matrix, top_n=5):
    """
    Recommend top-N products to a user based on weighted item similarity.

    Steps:
    1. For each product not yet rated by the user,
       compute a weighted score based on similarity to rated items.
    2. Return top-N product IDs ranked by score.
    """
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    scores = {}

    for item in item_sim_matrix.columns:
        if item in rated_items:
            continue
        sim_scores = item_sim_matrix[item][rated_items]
        user_scores = user_ratings[rated_items]
        if sim_scores.sum() > 0:
            weighted_score = np.dot(sim_scores, user_scores) / sim_scores.sum()
            scores[item] = weighted_score

    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked_items[:top_n]]

# Example usage
if __name__ == "__main__":
    # Path to your file (replace if needed)
    data_path = "amazon.csv"

    df = load_and_clean(data_path)
    user_item_matrix = build_user_item_matrix(df)
    item_sim_matrix = compute_item_similarity(user_item_matrix)

    example_user = user_item_matrix.index[0]
    recommendations = recommend_items(example_user, user_item_matrix, item_sim_matrix, top_n=10)

    print(f"Top 10 recommendations for user {example_user}:")
    print(recommendations)

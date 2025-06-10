import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

"""
Collaborative Filtering (Item-based) on Amazon product ratings.

This script:
1. Loads and cleans the dataset
2. Builds a user–item matrix
3. Computes item–item similarity using cosine similarity
4. Recommends top-N products to a given user based on their rating history
5. Evaluates recommendations using Precision and Recall 
"""

def load_and_clean(path):
    """Load dataset and convert ratings to float. Drop invalid entries."""
    df = pd.read_csv(path)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    return df[['user_id', 'product_id', 'rating']]

def build_user_item_matrix(df):
    """Build user-item rating matrix with 0 for missing entries."""
    return df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

def compute_item_similarity(user_item_matrix):
    """Compute cosine similarity between items."""
    item_matrix = user_item_matrix.T
    sim_matrix = cosine_similarity(item_matrix)
    return pd.DataFrame(sim_matrix, index=item_matrix.index, columns=item_matrix.index)

def recommend_items(user_id, user_item_matrix, item_sim_matrix, top_n=10):
    """Recommend top-N products for a given user using weighted similarity."""
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    scores = {}
    for item in item_sim_matrix.columns:
        if item in rated_items:
            continue
        sims = item_sim_matrix[item][rated_items]
        ratings = user_ratings[rated_items]
        if sims.sum() > 0:
            scores[item] = np.dot(sims, ratings) / sims.sum()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_n]]

def leave_one_out_split(df):
    """Split dataset into train and test by leaving one rating out per user."""
    train_rows = []
    test_rows = []
    grouped = df.groupby('user_id')

    for user, group in grouped:
        if len(group) < 2:
            continue
        group = group.sample(frac=1, random_state=42)
        test_rows.append(group.iloc[0])
        train_rows.extend(group.iloc[1:])

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, test_df
    
def evaluate_model_precision_recall(train_df, test_df, k=10):
    """
    Evaluate collaborative filtering using only Precision@k and Recall@k.
    One test item is held out per user (leave-one-out evaluation).
    """
    user_item_train = build_user_item_matrix(train_df)
    item_sim = compute_item_similarity(user_item_train)

    precision_scores = []
    recall_scores = []
    total_users = 0

    for _, row in test_df.iterrows():
        user = row['user_id']
        true_item = row['product_id']
        if user not in user_item_train.index:
            continue

        recs = recommend_items(user, user_item_train, item_sim, top_n=k)
        total_users += 1

        if true_item in recs:
            precision_scores.append(1 / k)
            recall_scores.append(1)
        else:
            precision_scores.append(0)
            recall_scores.append(0)

    print(f"\nEvaluated on {total_users} users:")
    print(f"Precision@{k}: {np.mean(precision_scores):.4f}")
    print(f"Recall@{k}:    {np.mean(recall_scores):.4f}\n")


if __name__ == "__main__":
    data_path = "amazon_sales_dataset.csv"  
    df = load_and_clean(data_path)

    # Evaluation block
    train_df, test_df = leave_one_out_split(df)
    evaluate_model(train_df, test_df, k=10)

    # Example recommendation
    user_item_matrix = build_user_item_matrix(train_df)
    item_sim_matrix = compute_item_similarity(user_item_matrix)
    example_user = user_item_matrix.index[0]
    recommendations = recommend_items(example_user, user_item_matrix, item_sim_matrix, top_n=10)

    print(f"Top 10 recommendations for user {example_user}:")
    print(recommendations)

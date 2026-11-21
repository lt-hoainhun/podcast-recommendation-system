
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from nlp_pipeline import run_nlp_pipeline

from recommender import (
    popularity_recommendations,
    recommend_for_user_content,
    item_based_cf
)


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# 1. Train/Test Split
def create_train_test_split(ratings_df, test_size=1, random_state=42):

    rng = np.random.default_rng(random_state)

    remaining_counts = ratings_df["item_id"].value_counts().to_dict()

    test_indices = []

    for user_id, group in ratings_df.groupby("user_id"):

        candidates = [
            idx for idx in group.index
            if remaining_counts[group.loc[idx, "item_id"]] > 1
        ]

        if not candidates:
            raise ValueError(
                f"Cannot create test split for user {user_id}"
            )

        selected = rng.choice(candidates, size=test_size, replace=False)

        for idx in selected:
            test_indices.append(idx)
            item_id = ratings_df.loc[idx, "item_id"]
            remaining_counts[item_id] -= 1

    test_df = ratings_df.loc[test_indices].copy()
    train_df = ratings_df.drop(test_indices).copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# 2. Precision@K
def precision_at_k(recommended_items, relevant_items, k=5):

    recommended_items = list(recommended_items)[:k]
    relevant_items = set(relevant_items)

    hits = len(set(recommended_items) & relevant_items)

    return hits / k


# 3. Evaluate recommendation methods
def evaluate_methods(items_df, ratings_df, tfidf_matrix, vectorizer):

    train_df, test_df = create_train_test_split(ratings_df)

    print("\nTRAIN DATA:", len(train_df))
    print("TEST DATA:", len(test_df))

    methods = {
        "Popularity": [],
        "Content-Based": [],
        "Item-Based CF": []
    }

    for user_id in sorted(ratings_df["user_id"].unique()):

        user_test = test_df[
            test_df["user_id"] == user_id
        ]

        relevant_items = user_test[
            user_test["rating"] >= 4
        ]["item_id"].tolist()

        # Popularity
        pop_results = popularity_recommendations(
            user_id,
            items_df,
            train_df,
            top_n=5,
            min_ratings=2
        )

        pop_items = pop_results["item_id"].tolist()

        # Content-Based
        content_results = recommend_for_user_content(
            user_id,
            items_df,
            train_df,
            tfidf_matrix,
            vectorizer,
            top_n=5
        )

        content_items = content_results["item_id"].tolist()

        # Item-Based Collaborative Filtering
        cf_results = item_based_cf(
            user_id,
            items_df,
            train_df,
            top_n=5
        )

        cf_items = cf_results["item_id"].tolist()

        # Calculate Precision@5
        methods["Popularity"].append(
            precision_at_k(pop_items, relevant_items, 5)
        )

        methods["Content-Based"].append(
            precision_at_k(content_items, relevant_items, 5)
        )

        methods["Item-Based CF"].append(
            precision_at_k(cf_items, relevant_items, 5)
        )

    summary = pd.DataFrame({
        "Method": list(methods.keys()),
        "Precision@5": [
            np.mean(scores)
            for scores in methods.values()
        ]
    })

    return summary, train_df, test_df


# 4. Rating distribution chart
def plot_rating_distribution(ratings_df):

    rating_counts = (
        ratings_df["rating"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        rating_counts.index.astype(str),
        rating_counts.values
    )

    plt.title("Podcast Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.xticks([1, 2, 3, 4, 5])

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "rating_distribution.png",
        dpi=300
    )

    plt.close()


# 5. Recommendation comparison chart
def plot_evaluation_comparison(summary):

    plt.figure(figsize=(9, 5))

    plt.bar(
        summary["Method"],
        summary["Precision@5"]
    )

    plt.title("Recommendation Methods Comparison")
    plt.xlabel("Recommendation Method")
    plt.ylabel("Precision@5")

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "evaluation_comparison.png",
        dpi=300
    )

    plt.close()


# 6. Main program
if __name__ == "__main__":

    print("===== LOADING DATA =====")

    items_df = pd.read_csv(DATA_DIR / "items.csv")
    ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")

    print("Episodes:", len(items_df))
    print("Users:", ratings_df["user_id"].nunique())
    print("Ratings:", len(ratings_df))

    print("\n===== BUILDING NLP MODEL =====")

    vectorizer, tfidf_matrix, keywords = run_nlp_pipeline(
        items_df
    )

    print("TF-IDF shape:", tfidf_matrix.shape)

    print("\n===== EVALUATION =====")

    summary, train_df, test_df = evaluate_methods(
        items_df,
        ratings_df,
        tfidf_matrix,
        vectorizer
    )

    print("\n===== PRECISION@5 RESULTS =====")

    print(summary.to_string(index=False))

    summary.to_csv(
        RESULTS_DIR / "evaluation_results.csv",
        index=False
    )

    print("\n===== GENERATING CHARTS =====")

    plot_rating_distribution(ratings_df)

    plot_evaluation_comparison(summary)

    print("\n===== SAVING RESULTS =====")

    print("Saved: results/evaluation_results.csv")
    print("Saved: results/rating_distribution.png")
    print("Saved: results/evaluation_comparison.png")

    print("\nEVALUATION COMPLETED SUCCESSFULLY!")

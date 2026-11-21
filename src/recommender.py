
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# 1. Popularity Baseline
def popularity_recommendations(
    user_id,
    items_df,
    ratings_df,
    top_n=5,
    min_ratings=2
):
    rated_items = ratings_df[
        ratings_df["user_id"] == user_id
    ]["item_id"].tolist()

    popularity = (
        ratings_df.groupby("item_id")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    popularity = popularity[
        popularity["rating_count"] >= min_ratings
    ]

    popularity = popularity[
        ~popularity["item_id"].isin(rated_items)
    ]

    results = popularity.merge(
        items_df,
        on="item_id"
    )

    results = results.sort_values(
        by=["average_rating", "rating_count"],
        ascending=False
    )

    return results[
        [
            "item_id",
            "episode_title",
            "show_name",
            "average_rating",
            "rating_count"
        ]
    ].head(top_n)


# 2. Get matching keywords

def get_matching_keywords(
    query_vector,
    item_vector,
    vectorizer,
    top_n=5
):
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(query_vector, "toarray"):
        query_array = query_vector.toarray().flatten()
    else:
        query_array = np.asarray(query_vector).flatten()

    if hasattr(item_vector, "toarray"):
        item_array = item_vector.toarray().flatten()
    else:
        item_array = np.asarray(item_vector).flatten()

    matching_scores = query_array * item_array

    top_indices = matching_scores.argsort()[::-1]

    keywords = [
        feature_names[i]
        for i in top_indices
        if matching_scores[i] > 0
    ]

    return keywords[:top_n]



# 3. NLP Content-Based: Text Query
def recommend_by_query(
    query,
    items_df,
    tfidf_matrix,
    vectorizer,
    top_n=5,
    exclude_item_ids=None
):
    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    exclude_item_ids = exclude_item_ids or []

    results = []

    for i, score in enumerate(similarity_scores):

        item_id = items_df.iloc[i]["item_id"]

        if item_id in exclude_item_ids:
            continue

        keywords = get_matching_keywords(
            query_vector,
            tfidf_matrix[i],
            vectorizer
        )

        results.append({
            "item_id": item_id,
            "episode_title": items_df.iloc[i]["episode_title"],
            "show_name": items_df.iloc[i]["show_name"],
            "similarity_score": float(score),
            "matching_keywords": ", ".join(keywords)
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        "similarity_score",
        ascending=False
    )

    return results.head(top_n)


# 4. NLP Content-Based: Similar Episodes
def recommend_similar_items(
    item_id,
    items_df,
    tfidf_matrix,
    vectorizer,
    top_n=5
):
    matches = items_df.index[
        items_df["item_id"] == item_id
    ].tolist()

    if not matches:
        raise ValueError(f"Episode {item_id} not found")

    item_index = matches[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[item_index],
        tfidf_matrix
    ).flatten()

    results = []

    for i, score in enumerate(similarity_scores):

        if i == item_index:
            continue

        keywords = get_matching_keywords(
            tfidf_matrix[item_index],
            tfidf_matrix[i],
            vectorizer
        )

        results.append({
            "item_id": items_df.iloc[i]["item_id"],
            "episode_title": items_df.iloc[i]["episode_title"],
            "show_name": items_df.iloc[i]["show_name"],
            "similarity_score": float(score),
            "matching_keywords": ", ".join(keywords)
        })

    results = pd.DataFrame(results)

    return results.sort_values(
        "similarity_score",
        ascending=False
    ).head(top_n)


# 5. NLP Content-Based: User Profile
def recommend_for_user_content(
    user_id,
    items_df,
    ratings_df,
    tfidf_matrix,
    vectorizer,
    top_n=5
):
    user_ratings = ratings_df[
        ratings_df["user_id"] == user_id
    ].copy()

    if user_ratings.empty:
        raise ValueError(f"User {user_id} has no ratings")

    positive_ratings = user_ratings[
        user_ratings["rating"] >= 4
    ]

    if positive_ratings.empty:
        positive_ratings = user_ratings

    item_indices = []
    weights = []

    for _, row in positive_ratings.iterrows():

        matches = items_df.index[
            items_df["item_id"] == row["item_id"]
        ].tolist()

        if matches:
            item_indices.append(matches[0])
            weights.append(float(row["rating"]))

    if not item_indices:
        raise ValueError("No matching episodes found")

    profile = tfidf_matrix[item_indices].multiply(
        np.array(weights)[:, None]
    ).sum(axis=0)

    profile = np.asarray(profile).reshape(1, -1)

    similarity_scores = cosine_similarity(
        profile,
        tfidf_matrix
    ).flatten()

    rated_items = user_ratings["item_id"].tolist()

    results = []

    for i, score in enumerate(similarity_scores):

        item_id = items_df.iloc[i]["item_id"]

        if item_id in rated_items:
            continue

        keywords = get_matching_keywords(
            profile,
            tfidf_matrix[i],
            vectorizer
        )

        results.append({
            "item_id": item_id,
            "episode_title": items_df.iloc[i]["episode_title"],
            "show_name": items_df.iloc[i]["show_name"],
            "similarity_score": float(score),
            "matching_keywords": ", ".join(keywords)
        })

    results = pd.DataFrame(results)

    return results.sort_values(
        "similarity_score",
        ascending=False
    ).head(top_n)


# 6. Item-Based Collaborative Filtering
def item_based_cf(
    user_id,
    items_df,
    ratings_df,
    top_n=5
):
    rating_matrix = ratings_df.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating"
    )

    if user_id not in rating_matrix.index:
        raise ValueError(f"User {user_id} not found")

    item_similarity = cosine_similarity(
        rating_matrix.fillna(0).T
    )

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=rating_matrix.columns,
        columns=rating_matrix.columns
    )

    user_ratings = rating_matrix.loc[user_id].dropna()

    rated_items = user_ratings.index.tolist()

    candidates = [
        item for item in rating_matrix.columns
        if item not in rated_items
    ]

    predictions = []

    for candidate in candidates:

        similarities = item_similarity_df.loc[
            candidate, rated_items
        ].values

        scores = user_ratings.values

        denominator = np.abs(similarities).sum()

        if denominator == 0:
            predicted_rating = 0
        else:
            predicted_rating = np.dot(
                similarities,
                scores
            ) / denominator

        predictions.append({
            "item_id": candidate,
            "predicted_rating": float(predicted_rating)
        })

    results = pd.DataFrame(predictions)

    results = results.merge(
        items_df,
        on="item_id"
    )

    results = results.sort_values(
        "predicted_rating",
        ascending=False
    )

    return results[
        [
            "item_id",
            "episode_title",
            "show_name",
            "predicted_rating"
        ]
    ].head(top_n)


# 7. Test all recommendation methods
if __name__ == "__main__":

    from nlp_pipeline import run_nlp_pipeline

    items = pd.read_csv("data/items.csv")
    ratings = pd.read_csv("data/ratings.csv")

    vectorizer, tfidf_matrix, keywords = run_nlp_pipeline(
        items
    )

    user_id = "U001"

    print("\n========== POPULARITY BASELINE ==========")

    print(
        popularity_recommendations(
            user_id,
            items,
            ratings
        ).to_string(index=False)
    )

    print("\n========== CONTENT-BASED QUERY ==========")

    query = "time management deep focus student study habits"

    print(
        recommend_by_query(
            query,
            items,
            tfidf_matrix,
            vectorizer
        ).to_string(index=False)
    )

    print("\n========== SIMILAR EPISODES ==========")

    print(
        recommend_similar_items(
            "P001",
            items,
            tfidf_matrix,
            vectorizer
        ).to_string(index=False)
    )

    print("\n========== USER CONTENT PROFILE ==========")

    print(
        recommend_for_user_content(
            user_id,
            items,
            ratings,
            tfidf_matrix,
            vectorizer
        ).to_string(index=False)
    )

    print("\n========== ITEM-BASED COLLABORATIVE FILTERING ==========")

    print(
        item_based_cf(
            user_id,
            items,
            ratings
        ).to_string(index=False)
    )

    print("\nRECOMMENDATION ENGINE COMPLETED!")


import pandas as pd
from pathlib import Path
from scipy.sparse import vstack

from nlp_pipeline import run_nlp_pipeline, transform_query

from recommender import (
    recommend_by_query,
    recommend_similar_items,
    popularity_recommendations
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def save_output(text):
    output_path = RESULTS_DIR / "sample_outputs.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

    print("\nSaved:", output_path)


if __name__ == "__main__":

    output = []

    items = pd.read_csv(DATA_DIR / "items.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")

    vectorizer, tfidf_matrix, keywords = run_nlp_pipeline(items)

    # TEST CASE 1: Topic Query
    print("\n========== TEST CASE 1 ==========")
    print("Topic Query: time management, deep focus, and student study habits")

    query = "time management deep focus student study habits"

    query_results = recommend_by_query(
        query,
        items,
        tfidf_matrix,
        vectorizer,
        top_n=5
    )

    print(query_results.to_string(index=False))

    output.append("TEST CASE 1: TOPIC QUERY")
    output.append(query_results.to_string(index=False))

    # TEST CASE 2: Next Episode
    print("\n========== TEST CASE 2 ==========")
    print("Favorite Episode: P001 - Deep Focus for Better Study")

    similar_results = recommend_similar_items(
        "P001",
        items,
        tfidf_matrix,
        vectorizer,
        top_n=5
    )

    print(similar_results.to_string(index=False))

    output.append("\nTEST CASE 2: NEXT EPISODE")
    output.append(similar_results.to_string(index=False))

    # TEST CASE 3: Cold Start
    print("\n========== TEST CASE 3 ==========")
    print("Adding a new episode with zero ratings.")

    new_episode = pd.DataFrame([{
        "item_id": "P036",
        "episode_title": "Deep Focus and Effective Study Planning",
        "show_name": "Student Podcast",
        "duration_min": 32,
        "show_notes": (
            "Learn deep focus techniques, time management, "
            "study planning, concentration, and productive "
            "learning habits for university students."
        ),
        "topic_tags": "study focus time management productivity"
    }])

    new_text = (
        new_episode["show_notes"].fillna("").astype(str)
        + " "
        + new_episode["topic_tags"].fillna("").astype(str)
    ).iloc[0]

    new_vector = transform_query(new_text, vectorizer)

    updated_items = pd.concat(
        [items, new_episode],
        ignore_index=True
    )

    updated_matrix = vstack(
        [tfidf_matrix, new_vector]
    ).tocsr()

    cold_start_query = (
        "time management deep focus student study habits"
    )

    cold_results = recommend_by_query(
        cold_start_query,
        updated_items,
        updated_matrix,
        vectorizer,
        top_n=5
    )

    print(cold_results.to_string(index=False))

    new_rating_count = ratings[
        ratings["item_id"] == "P036"
    ].shape[0]

    print("\nNew episode rating count:", new_rating_count)

    appears = "P036" in cold_results["item_id"].values

    print("New episode appears in recommendations:", appears)

    output.append("\nTEST CASE 3: COLD START")
    output.append(cold_results.to_string(index=False))
    output.append(f"\nNew episode rating count: {new_rating_count}")
    output.append(f"New episode appears: {appears}")

    # Additional popularity test
    print("\n========== USER RECOMMENDATIONS ==========")

    user_results = popularity_recommendations(
        "U001",
        items,
        ratings,
        top_n=5
    )

    print(user_results.to_string(index=False))

    output.append("\nUSER RECOMMENDATIONS")
    output.append(user_results.to_string(index=False))

    save_output("\n".join(output))

    print("\nALL DEMO TESTS COMPLETED!")

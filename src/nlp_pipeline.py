
import re
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and",
    "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "this", "that", "these",
    "those", "it", "its", "be", "been", "being", "can",
    "could", "would", "should", "will", "may", "might",
    "how", "what", "when", "where", "why", "who",
    "your", "you", "their", "they", "them", "we", "our",
    "about", "into", "through", "during", "while"
}


# 1. Clean text
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# 2. Tokenization
def tokenize_text(text):
    text = clean_text(text)

    tokens = re.findall(r"\b[a-z0-9]+\b", text)

    return [
        word for word in tokens
        if word not in STOP_WORDS and len(word) > 1
    ]


# 3. Combine episode text
def prepare_episode_text(items_df):
    texts = (
        items_df["show_notes"].fillna("").astype(str)
        + " "
        + items_df["topic_tags"].fillna("").astype(str)
    )

    return texts.tolist()


# 4. Build TF-IDF
def build_tfidf(items_df):
    texts = prepare_episode_text(items_df)

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=1000
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix


# 5. Extract top keywords
def extract_keywords(tfidf_matrix, vectorizer, top_n=5):
    feature_names = vectorizer.get_feature_names_out()

    all_keywords = []

    for row in range(tfidf_matrix.shape[0]):
        scores = tfidf_matrix[row].toarray().flatten()

        top_indices = scores.argsort()[::-1][:top_n]

        keywords = [
            feature_names[i]
            for i in top_indices
            if scores[i] > 0
        ]

        all_keywords.append(keywords)

    return all_keywords


# 6. Transform new query
def transform_query(query, vectorizer):
    return vectorizer.transform([query])


# 7. Run NLP pipeline
def run_nlp_pipeline(items_df):
    vectorizer, tfidf_matrix = build_tfidf(items_df)

    keywords = extract_keywords(
        tfidf_matrix,
        vectorizer,
        top_n=5
    )

    results = items_df[
        ["item_id", "episode_title"]
    ].copy()

    results["top_keywords"] = keywords

    return vectorizer, tfidf_matrix, results


# 8. Test program
if __name__ == "__main__":

    data_path = Path("data/items.csv")

    items_df = pd.read_csv(data_path)

    vectorizer, tfidf_matrix, results = run_nlp_pipeline(items_df)

    print("===== NLP PIPELINE RESULTS =====")

    print("Total episodes:", len(items_df))
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

    print("\n===== TOP 5 KEYWORDS =====")

    for _, row in results.iterrows():
        print("\n", row["item_id"], "-", row["episode_title"])
        print("Keywords:", ", ".join(row["top_keywords"]))

    print("\n===== QUERY TEST =====")

    query = "time management deep focus student study habits"

    query_vector = transform_query(query, vectorizer)

    print("Query:", query)
    print("Query vector shape:", query_vector.shape)

    print("\nNLP PIPELINE COMPLETED SUCCESSFULLY!")

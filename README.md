# Podcast Recommendation System

## Introduction
A podcast recommendation system using Natural Language Processing
and recommendation algorithms.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SciPy

## Dataset
- 35 podcast episodes
- 12 users
- 60 ratings

The dataset is synthetic and used for demonstration.

## Algorithms
1. Popularity-Based Recommendation
2. Content-Based Recommendation
3. Item-Based Collaborative Filtering

## Evaluation

Metric: Precision@5

| Method | Precision@5 |
|---|---:|
| Popularity | 0.000000 |
| Content-Based | 0.150000 |
| Item-Based CF | 0.083333 |

## How to Run

Install dependencies:

    python -m pip install -r requirements.txt

Run NLP:

    python src/nlp_pipeline.py

Run recommendation:

    python src/recommender.py

Run evaluation:

    python src/evaluate.py

Run demo:

    python src/main.py

## Results
Evaluation charts and demonstration outputs are stored
in the results folder.
# Project Testing Report

## 1. Dataset Validation

- 35 podcast episodes
- 12 users
- 60 ratings

## 2. NLP Testing

- Text cleaning
- Tokenization
- TF-IDF feature extraction
- Keyword extraction

TF-IDF matrix shape: (35, 750)

## 3. Recommendation Testing

- Popularity-Based Recommendation
- Content-Based Recommendation
- Item-Based Collaborative Filtering

## 4. Evaluation Results

| Algorithm | Precision@5 |
|---|---:|
| Popularity | 0.000000 |
| Content-Based | 0.150000 |
| Item-Based CF | 0.083333 |

## 5. Demo Testing

- Search-based recommendation
- Similar episode recommendation
- Cold-start recommendation

Output: results/sample_outputs.txt
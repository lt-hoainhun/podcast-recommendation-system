
import pandas as pd

items = pd.read_csv("data/items.csv")
ratings = pd.read_csv("data/ratings.csv")

print("===== ITEMS DATA =====")
print(items.head())
print("Total episodes:", len(items))

print("\n===== RATINGS DATA =====")
print(ratings.head())
print("Total ratings:", len(ratings))

print("\n===== USERS =====")
print(ratings["user_id"].nunique())

print("\n===== DATA CHECK =====")
print("Missing values:")
print(items.isnull().sum())
print(ratings.isnull().sum())

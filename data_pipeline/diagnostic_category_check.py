# diagnostic.py — run once, not part of the pipeline
import pandas as pd

df = pd.read_csv("data_pipeline/raw_books.csv")
clean = pd.read_csv("data_pipeline/clean_books.csv")

bad = clean[clean["category"].isin(["Add a comment", "Default"])]
print(bad[["title", "category"]])

# cross-reference to get the detail_url for the "Add a comment" one(s)
merged = df.merge(bad[["title"]], on="title")
print(merged[["title", "detail_url"]])
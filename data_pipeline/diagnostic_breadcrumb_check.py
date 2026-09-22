import pandas as pd
import requests
from bs4 import BeautifulSoup

# 1. Find the actual URL that failed from your data
df_raw = pd.read_csv("data_pipeline/raw_books.csv")
df_clean = pd.read_csv("data_pipeline/clean_books.csv")

bad_title = df_clean[df_clean["category"] == "Add a comment"]["title"].iloc[0]
bad_url = df_raw[df_raw["title"] == bad_title]["detail_url"].values[0]

print(f"Testing URL: {bad_url}")

# 2. Fetch and inspect it
resp = requests.get(bad_url)
resp.encoding = "utf-8"
soup = BeautifulSoup(resp.text, "html.parser")

crumb_list = soup.select("ul.breadcrumb li")
print(f"Found {len(crumb_list)} breadcrumb items:")
for li in crumb_list:
    print(repr(li.get_text(strip=True)), "| has <a>:", li.find("a") is not None)
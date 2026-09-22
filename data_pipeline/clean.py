"""
Cleans raw_books.csv into typed columns:
  price_gbp   float   (strip '£')
  rating      int     (word -> 1-5)
  in_stock    bool    (from availability text)
  price_inr   float   (price_gbp * 105.50, fixed project-defined constant)

Decision: rows that fail to parse (e.g. unexpected rating word, malformed
price) are DROPPED rather than median-imputed. Justification: on this
source every field is machine-generated from a fixed template, so a parse
failure signals a genuinely corrupt/unexpected row rather than normal
missing data -- imputing a fabricated price/rating for a bad scrape would
misrepresent that book, whereas dropping a handful of rows out of ~100
has negligible effect on meeting the >=60-row requirement.
"""
import pandas as pd

GBP_TO_INR = 105.50  # fixed project-defined constant, not a live rate

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

INPUT_CSV = "data_pipeline/raw_books.csv"
OUTPUT_CSV = "data_pipeline/clean_books.csv"


def parse_price(raw: str):
    try:
        return float(raw.replace("£", "").strip())
    except (ValueError, AttributeError):
        return None


def parse_rating(raw: str):
    return RATING_WORDS.get(raw)  # None if unrecognized word


def parse_in_stock(raw: str):
    if raw is None:
        return None
    return "In stock" in raw  # page text is e.g. "In stock (22 available)"


def main():
    df = pd.read_csv(INPUT_CSV)

    df["price_gbp"] = df["price_raw"].apply(parse_price)
    df["rating"] = df["star_rating_raw"].apply(parse_rating)
    df["in_stock"] = df["availability_raw"].apply(parse_in_stock)

    before = len(df)
    bad_rows = df[df["price_gbp"].isna() | df["rating"].isna() | df["in_stock"].isna()]
    if len(bad_rows) > 0:
        print(f"Dropping {len(bad_rows)} unparseable rows:")
        print(bad_rows[["title", "price_raw", "star_rating_raw", "availability_raw"]])

    df = df.dropna(subset=["price_gbp", "rating", "in_stock"]).copy()
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    keep_cols = ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    df = df[keep_cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Kept {len(df)}/{before} rows -> {OUTPUT_CSV}")
    print(f"Categories: {df['category'].nunique()} unique -> {sorted(df['category'].unique())}")


if __name__ == "__main__":
    main()
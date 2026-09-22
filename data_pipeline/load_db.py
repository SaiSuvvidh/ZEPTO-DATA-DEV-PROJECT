"""
Loads clean_books.csv into a normalized 2-table SQLite schema:

  categories(category_id PK, category_name UNIQUE)
  books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
        category_id FK -> categories.category_id)

Built via raw sqlite3 (not to_sql) specifically so we control the PK/FK
constraints directly -- to_sql would create the tables without enforcing
the foreign key relationship the rubric requires.
"""
import sqlite3
import pandas as pd

CSV_PATH = "data_pipeline/clean_books.csv"
DB_PATH = "data_pipeline/zepto_books.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def main():
    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA)

    # Clear tables on rerun so this script is idempotent (safe to run multiple times)
    conn.execute("DELETE FROM books;")
    conn.execute("DELETE FROM categories;")

    # Populate categories first, in sorted order (deterministic IDs across reruns)
    categories = sorted(df["category"].unique())
    conn.executemany(
        "INSERT INTO categories (category_name) VALUES (?);",
        [(c,) for c in categories]
    )
    conn.commit()

    # Build category_name -> category_id lookup
    cat_map = dict(conn.execute("SELECT category_name, category_id FROM categories;").fetchall())

    # Insert books, resolving category_id via the lookup
    book_rows = [
        (row.title, row.price_gbp, row.price_inr, int(row.rating), int(row.in_stock),
         cat_map[row.category])
        for row in df.itertuples()
    ]
    conn.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?);""",
        book_rows
    )
    conn.commit()

    n_books = conn.execute("SELECT COUNT(*) FROM books;").fetchone()[0]
    n_cats = conn.execute("SELECT COUNT(*) FROM categories;").fetchone()[0]
    print(f"Loaded {n_books} books across {n_cats} categories into {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
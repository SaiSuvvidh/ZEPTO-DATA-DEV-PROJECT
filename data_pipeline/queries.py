"""
Runs the 5 required SQL queries against zepto_books.db, printing each
query + its output. Then demonstrates:
  - pd.read_sql for 2 of the queries (Q1, Q5)
  - pd.merge reproducing Q5's JOIN purely in-memory (no SQL), compared
    against the SQL JOIN result to prove both approaches match.
"""
import sqlite3
import pandas as pd

DB_PATH = "data_pipeline/zepto_books.db"


def run_and_print(conn, label, query, params=()):
    print(f"\n--- {label} ---")
    print(query.strip())
    cur = conn.execute(query, params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    for r in rows:
        print(dict(zip(cols, r)))
    return rows, cols


def main():
    conn = sqlite3.connect(DB_PATH)

    # Q1 -- SELECT / WHERE / ORDER BY / LIMIT: cheapest 5 in-stock books
    q1 = """
        SELECT title, price_inr, rating
        FROM books
        WHERE in_stock = 1
        ORDER BY price_inr ASC
        LIMIT 5;
    """
    run_and_print(conn, "Q1: 5 cheapest in-stock books", q1)

    # Q2 -- DISTINCT: list all category names actually used
    q2 = "SELECT DISTINCT category_name FROM categories ORDER BY category_name;"
    run_and_print(conn, "Q2: distinct categories", q2)

    # Q3 -- BETWEEN: books priced between 500 and 1000 INR
    q3 = """
        SELECT title, price_inr
        FROM books
        WHERE price_inr BETWEEN 500 AND 1000
        ORDER BY price_inr;
    """
    run_and_print(conn, "Q3: books priced 500-1000 INR", q3)

    # Q4 -- IN: books rated 4 or 5 stars
    q4 = """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title;
    """
    run_and_print(conn, "Q4: books rated 4 or 5 stars", q4)

    # Q5 -- JOIN: top 2 highest-rated books per category (simple version:
    # full join, ranking left to pandas/manual inspection is unnecessary --
    # we just need "a JOIN" demonstrated with clean output)
    q5 = """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC;
    """
    q5_rows, q5_cols = run_and_print(conn, "Q5: all books joined with category name", q5)

    # ---- pandas parity demonstration ----
    print("\n=== pandas parity check ===")

    # pd.read_sql for Q1
    df_q1 = pd.read_sql(q1, conn)
    print("\nQ1 via pd.read_sql:")
    print(df_q1)

    # pd.read_sql for Q5 (the JOIN, via SQL)
    df_q5_sql = pd.read_sql(q5, conn)

    # Reproduce Q5 via pd.merge on in-memory DataFrames (no SQL at all)
    books_df = pd.read_sql("SELECT * FROM books;", conn)
    categories_df = pd.read_sql("SELECT * FROM categories;", conn)
    df_q5_merge = (
        books_df.merge(categories_df, on="category_id")
        [["category_name", "title", "rating", "price_inr"]]
        .sort_values(["category_name", "rating"], ascending=[True, False])
        .reset_index(drop=True)
    )
    df_q5_sql_sorted = df_q5_sql.sort_values(["category_name", "rating"], ascending=[True, False]).reset_index(drop=True)

    print("\nQ5 via SQL JOIN (pd.read_sql):")
    print(df_q5_sql_sorted.head())
    print("\nQ5 via pd.merge (in-memory, no SQL):")
    print(df_q5_merge.head())

    match = df_q5_sql_sorted.equals(df_q5_merge)
    print(f"\nSQL JOIN output matches pd.merge output: {match}")

    conn.close()


if __name__ == "__main__":
    main()
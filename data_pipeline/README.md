# Data Pipeline Module

Scrapes book catalog data from books.toscrape.com, cleans and types it, loads it into a normalized SQLite database, and demonstrates SQL + pandas querying.

## Run
```bash
python scrape.py      # -> raw_books.csv (100 rows, 5 listing pages)
python clean.py       # -> clean_books.csv (95 rows after dropping malformed)
python load_db.py     # -> zepto_books.db
python queries.py     # -> query_results.txt
```

## Design Decisions

**Scraping scope:** First 5 "All products" listing pages (100 books) rather than fixed categories, guaranteeing ≥60 rows and category diversity (28 unique categories) automatically.

**Encoding fix:** `books.toscrape.com` doesn't declare its encoding in HTTP headers, so `requests` mis-guesses it and corrupts the `£` symbol (`£` → `Â£`). Fixed by explicitly setting `resp.encoding = "utf-8"` after each request.

**Row-drop policy:** 5 rows were dropped where `category` came back as `"Add a comment"` — a confirmed breadcrumb-generator artifact on a subset of the site's pages (verified by direct HTML inspection), not a real category. Treated identically to any other malformed/unparseable field per the stated drop policy: since every field here is machine-generated from a fixed template, a parse failure signals genuine corruption rather than normal missing data, so dropping (rather than imputing a fabricated category) was chosen. Final dataset: **95/100 rows kept**, 28 legitimate categories (including "Default", which is a real category on this site, distinct from the dropped artifact).

**Currency conversion:** `price_inr = price_gbp * 105.50` — a fixed, project-defined constant, not a live/historical rate.

**Schema:** Normalized 2-table design —
```sql
categories(category_id PK, category_name UNIQUE)
books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK -> categories)
```
Built via raw `sqlite3` (not `pandas.to_sql`) to explicitly enforce the FK constraint.

## SQL Queries (see `queries.py` / `query_results.txt` for full output)
1. SELECT/WHERE/ORDER BY/LIMIT — 5 cheapest in-stock books
2. DISTINCT — all category names
3. BETWEEN — books priced 500–1000 INR
4. IN — books rated 4 or 5 stars
5. JOIN — all books joined with category name, ordered by category then rating

`pd.read_sql` was used to reproduce queries 1 and 5 as DataFrames; `pd.merge` reproduced query 5's JOIN entirely in-memory from the `books`/`categories` DataFrames with no SQL — both outputs matched exactly (confirmed via `.equals()`).
# Data Pipeline Module

Scrapes book catalog data from books.toscrape.com, cleans and types it, loads it into a normalized SQLite database, and demonstrates SQL + pandas querying.

## Run
```bash
python scrape.py      # -> raw_books.csv
python clean.py       # -> clean_books.csv
python load_db.py     # -> zepto_books.db
python queries.py     # -> query_results.txt
```

## How I Built This

The task allowed either scraping fixed categories or covering the first 5 listing pages of the "All products" catalogue — "either scope is acceptable," as long as I cleared 60 books. I went with the listing pages, since it gets me both the row count and whatever category spread the site naturally has in one pass, rather than picking categories upfront and hoping they add up to enough books. That gave me 100 books across the site's category breadcrumbs.

**The encoding bug.** My first real problem showed up not during scraping but during cleaning — when I ran `clean.py` the first time, it dropped all 100 rows as unparseable. Digging in, the issue was that `books.toscrape.com` doesn't declare its character encoding in the HTTP response headers, so `requests` was guessing an encoding that mangled the `£` symbol into `Â£`. My price parser was doing a plain `.replace("£", "")`, which obviously couldn't find `£` anymore once it had been corrupted, so every price came back unparseable. I fixed it two ways: explicitly setting `resp.encoding = "utf-8"` right after each request in `scrape.py`, and rewriting `parse_price` in `clean.py` to pull the numeric value out with a regex instead of relying on stripping a specific currency character — that way it's robust even if something like this happens again. I had to re-run the scraper after this fix, since the corruption was already baked into the raw CSV.

**The category bug.** After the encoding was sorted, cleaning ran fine and kept all 100 rows — but I noticed the category list included `"Add a comment"`, which is obviously not a real book category. I wrote a couple of small diagnostic scripts (`diagnostic_category_check.py` and `diagnostic_breadcrumb_check.py`, kept in this folder as a record of the investigation) to trace which books had this and pull their actual page HTML. It turned out to be a genuine quirk of the site itself: for a subset of books, the breadcrumb navigation on the product page literally says "Add a comment" in the slot where the category link should be — it's not something my scraper selector did wrong, it's how the site generated those specific pages. I confirmed this by checking the raw breadcrumb HTML directly. Since I already had a rule that unparseable/corrupted fields get dropped rather than guessed at, I extended that same rule to cover this: any row where `category` came back as `"Add a comment"` gets dropped, exactly like a bad price or rating would. That took me from 100 down to 95 clean rows, still comfortably over the 60-row minimum, with 28 legitimate categories remaining (including "Default," which — unlike "Add a comment" — really is a genuine category on this site for a batch of its books).

**Currency conversion.** `price_inr = price_gbp * 105.50` — the fixed rate I was given for this project, not a live lookup.

**Database schema.** Loading the data in could've gone through either raw `sqlite3` or `pandas.DataFrame.to_sql`. I used `sqlite3` directly, since it gave me predictable control over the foreign key relationship between the two tables — `to_sql` builds tables straight from a DataFrame's own dtypes, and I didn't want to leave the FK constraint to chance:
```sql
categories(category_id PK, category_name UNIQUE)
books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK -> categories)
```

## SQL Queries (full output in `query_results.txt`)
1. SELECT/WHERE/ORDER BY/LIMIT — 5 cheapest in-stock books
2. DISTINCT — all category names
3. BETWEEN — books priced 500–1000 INR
4. IN — books rated 4 or 5 stars
5. JOIN — all books joined with category name, ordered by category then rating

I reproduced queries 1 and 5 via `pd.read_sql`, and separately rebuilt query 5's join purely in-memory with `pd.merge` on the `books`/`categories` DataFrames — no SQL involved in that second version. I compared the two outputs with `.equals()` and confirmed they matched exactly.

[query 1 result added here for Preview:
--- Q1: 5 cheapest in-stock books ---
SELECT title, price_inr, rating
        FROM books
        WHERE in_stock = 1
        ORDER BY price_inr ASC
        LIMIT 5;
{'title': 'Patience', 'price_inr': 1071.88, 'rating': 3}
{'title': 'In Her Wake', 'price_inr': 1354.62, 'rating': 1}
{'title': 'Princess Between Worlds (Wide-Awake Princess #5)', 'price_inr': 1407.37, 'rating': 5}
{'title': 'Princess Jellyfish 2-in-1 Omnibus, Vol. 01 (Princess Jellyfish 2-in-1 Omnibus #1)', 'price_inr': 1435.855, 'rating': 5}
{'title': 'Starving Hearts (Triangular Trade Trilogy, #1)', 'price_inr': 1475.945, 'rating': 2}
]
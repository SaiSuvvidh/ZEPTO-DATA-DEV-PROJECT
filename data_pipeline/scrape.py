"""
Scrapes books.toscrape.com: walks the first 5 'All products' listing pages,
then visits each book's detail page once to pull its category (not shown
on the listing page). Output is RAW/uncleaned — cleaning happens in a
separate script (clean.py) so scraping and cleaning stay independently
testable/rerunnable.
"""
import time
import csv
import requests
from bs4 import BeautifulSoup

BASE = "http://books.toscrape.com/catalogue/"
LISTING_URL = BASE + "page-{}.html"
OUTPUT_CSV = "data_pipeline/raw_books.csv"

# Maps book-page CSS star-rating class -> nothing yet; kept as raw text here,
# converted to int only in clean.py (raw scrape should stay a 1:1 mirror
# of what's on the page, not do parsing logic).
def scrape_listing_page(page_num: int, session: requests.Session) -> list[dict]:
    url = LISTING_URL.format(page_num)
    resp = session.get(url, timeout=10)
    resp.encoding = "utf-8"
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    books = []
    for article in soup.select("article.product_pod"):
        title = article.h3.a["title"]
        price_text = article.select_one(".price_color").text  # e.g. "£51.77"
        rating_text = article.select_one("p.star-rating")["class"][1]  # e.g. "Three"
        availability_text = article.select_one(".availability").text.strip()
        detail_href = article.h3.a["href"]  # relative, needs BASE join

        books.append({
            "title": title,
            "price_raw": price_text,
            "star_rating_raw": rating_text,
            "availability_raw": availability_text,
            "detail_url": BASE + detail_href.replace("../../../", ""),
        })
    return books


def fetch_category(detail_url: str, session: requests.Session) -> str:
    resp = session.get(detail_url, timeout=10)
    resp.encoding = "utf-8"
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Breadcrumb: Home > Books > <Category> > <Title>
    crumbs = soup.select("ul.breadcrumb li a")
    return crumbs[-1].text.strip()  # last <a> before the (non-link) title crumb


def main():
    session = requests.Session()
    all_books = []

    for page in range(1, 6):  # pages 1..5
        print(f"Scraping listing page {page}...")
        all_books.extend(scrape_listing_page(page, session))
        time.sleep(0.5)  # polite delay; this is a practice site but good habit regardless

    print(f"Found {len(all_books)} books. Fetching categories...")
    for i, book in enumerate(all_books):
        book["category"] = fetch_category(book["detail_url"], session)
        if i % 20 == 0:
            print(f"  {i}/{len(all_books)} categories fetched")
        time.sleep(0.2)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_books[0].keys()))
        writer.writeheader()
        writer.writerows(all_books)

    print(f"Saved {len(all_books)} raw rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
# Zepto Data & AI Platform — Capstone Project Problem

This is my submission for the AI/ML capstone — an end-to-end platform built as, If I were to join Zepto's analytics, covering three linked pieces: a scraping-based data pipeline, a full EDA + modeling pipeline on the Titanic dataset, and a RAG-based GenAI support assistant. All three live in this one repo, built and committed in that order.

## Repository Structure

/data_pipeline/ - Scraping, cleaning, SQLite loading, SQL queries
/analytics/ - Titanic EDA + classification/regression modeling
/support_assistant/ - LangGraph + ChromaDB + FastAPI RAG service
requirements.txt - Single consolidated dependency file for all 3 modules


## Setup

```bash
git clone https://github.com/SaiSuvvidh/ZEPTO-DATA-DEV-PROJECT.git
cd ZEPTO-DATA-DEV-PROJECT
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

The Project description left the requirements file open — either one per module or a single consolidated file, my choice to make. I went with one file for all three, since the whole project runs in one shared environment for me and there wasn't a real dependency split between modules that would've justified breaking it apart. That single file ended up mattering more than I expected — a mismatch between my terminal's `pip` and my venv's own `pip` left it silently missing half of Module 3's dependencies at one point, which I only caught once the Docker build failed on it (Full explanation extended in the support_assistant README).

## Running Each Module

### 1. Data Pipeline
```bash
python data_pipeline/scrape.py      # scrapes books.toscrape.com -> raw_books.csv
python data_pipeline/clean.py       # cleans + types + computes price_inr -> clean_books.csv
python data_pipeline/load_db.py     # loads into normalized SQLite schema -> zepto_books.db
python data_pipeline/queries.py     # runs 5 required SQL queries + pandas parity check
```

### 2. Analytics
```bash
python analytics/_01_eda.py         # profiling, cleaning, EDA story, saves titanic.csv
python analytics/_02_modeling.py    # stratified split, 3 classifiers, tuning, regression, saves best_pipeline.joblib
```
(I named these with a leading underscore — `_01_eda.py`, `_02_modeling.py` — since Python doesn't allow module names to start with a digit. Keeps the intended run order visible without breaking imports.)

### 3. Support Assistant
```bash
python support_assistant/setup_corpus.py    # writes the 8 policy doc files
python support_assistant/embed_corpus.py    # embeds + stores in ChromaDB
uvicorn support_assistant.main:app --reload --port 8000   # run the API locally

# Docker alternative (build from repo root):
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```
`MOCK_LLM` defaults to `1`, which is the mode I built and tested everything against — no API key or network access to an LLM provider needed.

## Design Decisions

**Data Pipeline.** The scope here was also left open — either scrape fixed categories or cover the first 5 listing pages of the full catalogue, as long as I cleared 60 books. I went with the listing-page route, since it gets both the row count and the category spread in one pass rather than me hand-picking categories and hoping they add up. That gave me 100 books across 28 categories. Five of those rows had `"Add a comment"` sitting where the category should be — I dug into this (full story in the data_pipeline README) and confirmed it's a genuine artifact from the site's own breadcrumb generator on a handful of pages, not something my scraper broke. I dropped those rows rather than guess at a category, landing on 95 clean rows. Currency conversion uses the fixed constant as per statement of Question Description: **1 GBP = 105.50 INR**, nothing live.

**Analytics.** I followed the missing-value threshold rule as given: dropped `embarked`/`embark_town` (0.22% missing each), median-imputed `age` grouped by pclass+sex (19.87% missing), and for `deck` (77.22% missing) — past the point where imputation is reliable — the Question description stated to either drop the column or encode the gaps as their own category and justify the pick in writing. I encoded it as "Unknown" instead of dropping it, since whether a cabin got recorded at all is plausibly tied to class and fare, and dropping the column felt like it would throw away a real signal along with the noise. For modeling, everything ran through a `ColumnTransformer` + `Pipeline` so I couldn't accidentally leak test data into training. Of the three classifiers, Random Forest came out ahead overall (0.816 accuracy, 0.744 F1, tuned OOB 0.823), so that's what I'd deploy, even though Logistic Regression edged it slightly on AUC. Neither `class_weight='balanced'` nor SMOTE beat the baseline F1 — with only mild imbalance in this dataset, I wasn't expecting either to move the needle much, and they didn't. The fare-prediction regression side-task was a separate exercise — R²=0.40 and a clearly heteroscedastic residual plot told me it's a useful exploratory model, not something I'd trust for production.

**Support Assistant.** I built the RAG pipeline over the 8 provided Zepto policy documents, embedding them locally with `all-MiniLM-L6-v2` and storing them in ChromaDB — no API key involved anywhere in retrieval. The LangGraph router uses a plain keyword heuristic to decide if a query needs policy retrieval or not. Everything I actually tested and got working runs under `MOCK_LLM=1` — no real LLM call anywhere in my graded path; the structured prompt template exists in the repo as required but only actually gets exercised if I ever flip on the optional real-LLM extension. I did hit a real snag getting the Dockerfile working (detailed in the support_assistant README) that turned out to trace back to `requirements.txt`, not the Dockerfile itself.

## Git Workflow

All of this was built on `feature/zepto-development`, committed incrementally as I finished each piece, then merged into `main`. I noticed partway through that my first merge attempt fast-forwarded instead of leaving an actual merge commit (since `main` had no commits of its own past the branch point) — so I added an empty commit on the feature branch and redid the merge with `--no-ff` to get an explicit merge commit showing the branch history clearly in `git log --graph --all`.

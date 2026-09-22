# Zepto Data & AI Platform — Capstone Project

An end-to-end AI/ML platform built for Zepto's analytics guild, spanning three linked modules in a single repository: a web-scraping data pipeline, a full EDA + predictive modeling pipeline, and a RAG-based GenAI support assistant.

## Repository Structure
/data_pipeline/       - Scraping, cleaning, SQLite loading, SQL queries
/analytics/           - Titanic EDA + classification/regression modeling
/support_assistant/   - LangGraph + ChromaDB + FastAPI RAG service
requirements.txt      - Single consolidated dependency file for all 3 modules

## Setup

```bash
git clone https://github.com/SaiSuvvidh/ZEPTO-DATA-DEV-PROJECT.git
cd ZEPTO-DATA-DEV-PROJECT
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

A single consolidated `requirements.txt` is used for all three modules rather than per-module files, since the whole project runs in one shared environment.

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

### 3. Support Assistant
```bash
python support_assistant/setup_corpus.py    # writes the 8 policy doc files
python support_assistant/embed_corpus.py    # embeds + stores in ChromaDB
uvicorn support_assistant.main:app --reload --port 8000   # run the API locally

# Docker alternative (build from repo root):
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```
`MOCK_LLM` defaults to `1` (mock mode) — this is the graded baseline and requires no API key or network access to any LLM provider.

## Design Decisions Summary

**Data Pipeline:** Scraped the first 5 catalogue listing pages (100 books, 28 categories) rather than picking fixed categories, to guarantee both row count and category diversity automatically. Malformed rows (5 books with a breadcrumb-generator artifact "Add a comment" instead of a real category) were dropped rather than imputed, since on this machine-generated source a parse failure indicates a corrupt row, not normal missingness. Fixed conversion rate: **1 GBP = 105.50 INR** (project-defined constant, not a live rate). Schema: normalized 2-table `categories`/`books` with PK/FK relationship, built via raw `sqlite3` rather than `to_sql` to control the FK constraint directly.

**Analytics:** Missing values handled per the percentage-threshold rule: `embarked`/`embark_town` (0.22%) dropped, `age` (19.87%) median-imputed grouped by pclass+sex, `deck` (77.22%) encoded as its own "Unknown" category rather than dropped, since whether a cabin was recorded likely correlates with survival. Modeling used a `ColumnTransformer`+`Pipeline` fit strictly on the training split. Random Forest was the best classifier (Accuracy 0.816, F1 0.744, tuned OOB 0.823) and is the recommended deployment choice over Logistic Regression (higher AUC but lower F1) and Decision Tree (weakest on all metrics). Imbalance handling (`class_weight='balanced'`, SMOTE) did not improve on baseline F1, consistent with this dataset's only-mild imbalance (61.6%/38.4%). The fare-prediction regression side-task (R²=0.40) showed clear heteroscedasticity in its residuals.

**Support Assistant:** RAG pipeline over 8 Zepto policy documents, embedded locally with `all-MiniLM-L6-v2` and stored in ChromaDB — no API key needed for embeddings or retrieval. A 3-node LangGraph (`classify_intent` → `retrieve_and_answer` / `direct_answer`) routes queries via a keyword heuristic. The graded baseline runs entirely offline via `MOCK_LLM=1` (default): no real LLM call is made anywhere in the graded path — answers are either a canned template built from the top retrieved chunk, or a fixed refusal string for non-policy questions. The optional `MOCK_LLM=0` extension (not required for grading) would route generation through a real LLM using the structured prompt template in `prompt_template.py`.

## Git Workflow
All development happened on `feature/zepto-development`, committed incrementally through each module, then merged into `main`.
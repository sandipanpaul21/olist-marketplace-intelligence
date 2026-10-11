# Olist Marketplace Intelligence

This project combines exploratory analysis of the Olist Brazilian e-commerce dataset with a policy-focused retrieval-augmented generation (RAG) workflow. The goal is to understand marketplace operations and performance from raw transaction data, then answer policy questions using a searchable and traceable knowledge base grounded in the policy text.

## Project overview

The repository is structured around two complementary tasks:

1. Data analysis and marketplace intelligence
   - Inspect order, payment, seller, customer, product, and review data
   - Identify operational patterns and quality issues in marketplace behavior
   - Explore how commercial performance varies by geography, category, and seller attributes

2. Policy question answering with RAG
   - Break policy text into individual clauses and sections
   - Index clauses using embeddings and keyword retrieval
   - Retrieve the most relevant policy passages for a user question
   - Generate answers with citations and abstain when evidence is weak
   - Validate retrieval quality and answer faithfulness

This solution is intended to be practical and reproducible: it starts from raw e-commerce data, moves into insight generation, and then adds a retrieval layer that can help answer policy-related questions with source-backed responses.

## Solution approach

The solution follows a standard end-to-end workflow from raw data to decision-support outputs:

- Data loading and profiling: review the structure, missingness, and quality of the Olist datasets
- Exploratory analysis: assess sales patterns, order flow, customer behavior, product segmentation, and seller performance
- Policy processing: parse policy clauses into structured chunks with section and reference metadata
- Hybrid retrieval: combine semantic embeddings and BM25 rankings to find the most relevant policy passages
- Answer generation: use the retrieved policy context to answer questions while grounding responses in the cited clauses
- Faithfulness and abstention checks: reject unsupported outcomes and validate that the answer stays aligned with the retrieved evidence
- Evaluation: measure retrieval quality with recall@k and review answer quality through faithfulness checks

A more detailed summary of the methodology is available in the repository root document `Solution Approach.pdf` (this is the project document referenced in materials as `Solution_approach.pdf`).

## Repository structure

```text
data/raw/             Raw Olist CSV files (kept out of Git)
notebooks/EDA/        Sequential exploratory analysis notebooks
notebooks/RAG/        Policy indexing, retrieval, generation, and evaluation notebooks
src/policy_rag.py     Shared policy loading, retrieval, generation, and evaluation logic
requirements.txt      Python dependencies for the project
Solution Approach.pdf Summary of the project methodology and solution design
```

## Data assets

The project is based on the Olist Brazilian E-Commerce dataset. The raw source files should be placed in `data/raw/` before running the notebooks:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_geolocation_dataset.csv`
- `product_category_name_translation.csv`

These files are intentionally not committed to the repository; the folder is kept as a local working directory for the dataset inputs.

## Workflow and notebooks

### Exploratory data analysis

The EDA notebooks in `notebooks/EDA/` are designed to be run in sequence. They progressively examine the raw data, check data quality, and build a narrative around marketplace activity. This stage helps surface operational issues such as incomplete records, timing anomalies, payment behaviors, and variation across seller and customer segments.

### Policy RAG workflow

The notebooks in `notebooks/RAG/` follow the policy retrieval pipeline in order:

- Load and process the policy text
- Create clause-level chunks and reference structures
- Build semantic and keyword indexes
- Retrieve candidate clauses for each question
- Generate grounded answers with citations
- Evaluate retrieval performance and answer abstention behavior

The shared logic in `src/policy_rag.py` centralizes the indexing and retrieval logic so that the notebooks stay focused on experimentation and assessment rather than duplicated implementation details.

## Getting started

Use Python 3.12 and set up a virtual environment from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

From the Jupyter interface, begin with the EDA notebooks in `notebooks/EDA/` and then continue to the RAG notebooks in `notebooks/RAG/` once the raw data is available.

## Configuration notes

The retrieval and generation workflow relies on local embedding models and optionally the Anthropic API for answer generation and verification. If the generation notebooks are used, set the API key in the shell before launching Jupyter:

```powershell
$env:ANTHROPIC_API_KEY = "your-api-key"
```

The project also supports an override for the LLM model through the environment variable `RAG_LLM_MODEL` when needed. Keep secrets out of Git and avoid committing API keys or local dataset files.

## Expected outcome

The final result is a repository that can:

- analyze marketplace patterns from Olist transaction data,
- explain seller and customer dynamics from the dataset,
- answer structured policy questions with policy-grounded evidence,
- and evaluate the reliability of retrieval and generation behavior.

This combination of business analytics and retrieval-augmented policy reasoning makes the project useful as both an analytical workflow and a practical RAG implementation example.

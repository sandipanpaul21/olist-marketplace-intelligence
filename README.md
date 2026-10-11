# Olist Marketplace Intelligence

Exploratory analysis of the Olist Brazilian e-commerce dataset, plus a policy-focused retrieval-augmented generation (RAG) workflow. The notebooks cover data quality and marketplace operations, then demonstrate policy retrieval, answer generation, and faithfulness checks.

## Project structure

```text
data/raw/             Place the source CSV files here (kept out of Git)
notebooks/EDA/        Sequential exploratory data analysis notebooks
notebooks/RAG/        Policy RAG and retrieval-evaluation notebooks
src/policy_rag.py     Shared policy loading, retrieval, generation, and evaluation code
requirements.txt      Pinned Python dependencies
```

## Getting started

Use Python 3.12, then create and activate a virtual environment from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

Open notebooks from the Jupyter interface. For the EDA workflow, start with `notebooks/EDA/eda_00_raw_data.ipynb` and continue through the numbered notebooks.

## Raw data

Download the Olist Brazilian E-Commerce dataset and place its nine CSV files in `data/raw/`:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_geolocation_dataset.csv`
- `product_category_name_translation.csv`

Raw data files are ignored by Git; only an empty placeholder is tracked so the folder is present after cloning.

## Policy RAG notebooks

Run the notebooks in `notebooks/RAG/` in numeric order. The embedding workflow downloads the configured Sentence Transformers model on first use. The generation and faithfulness workflows call the Anthropic API; set the key in your PowerShell session before launching Jupyter:

```powershell
$env:ANTHROPIC_API_KEY = "your-api-key"
```

Keep API keys and other secrets out of the repository.

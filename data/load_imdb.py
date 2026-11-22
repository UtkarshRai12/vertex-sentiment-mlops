# Load a fixed IMDB sample into BigQuery.
#
# The full set is 50,000 labelled movie reviews. We keep 15,000 so later
# training stays cheap. The seed is fixed so every re-run loads the same rows.
#
# label: 1 = positive, 0 = negative
# split: train (10,000) or test (5,000)
#
# Run from the repo root, with the venv active and gcloud config p1:
#   python -m data.load_imdb

import pandas as pd
from datasets import load_dataset
from google.cloud import bigquery

import config

# Public IMDB dataset on Hugging Face. First run downloads it.
ds = load_dataset("stanfordnlp/imdb")

# sample() without a seed would change the rows every run.
train = ds["train"].to_pandas().sample(10_000, random_state=42).assign(split="train")
test = ds["test"].to_pandas().sample(5_000, random_state=42).assign(split="test")

# The source column is `text`. Every later phase expects `review`.
df = pd.concat([train, test]).rename(columns={"text": "review"})

bq = bigquery.Client(project=config.PROJECT)
table_id = f"{config.PROJECT}.{config.BQ_DATASET}.reviews"

# WRITE_TRUNCATE replaces the table, so a second run does not stack rows.
job = bq.load_table_from_dataframe(
    df,
    table_id,
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
)
job.result()

print("rows loaded:", bq.get_table(table_id).num_rows)
